"""
Tabular detector — silent corruption in CSV, TSV and spreadsheet exports.

Tabular data fails quietly and often. The file opens, the row count looks right, the
dataframe has the columns you expected, and some of the values are wrong. Nothing
raises. These are the failures worth catching. I have seen all of them in real data:

  MOJIBAKE            UTF-8 bytes decoded as Latin-1 or CP1252. "café" becomes
                      "cafÃ©", "don't" becomes "don't". Sorts fine, joins fine,
                      is wrong.

  NUMERIC_AS_TEXT     A column of numbers stored as strings. Aggregates silently
                      return nonsense, or the column is dropped from a numeric
                      summary and nobody notices the gap.

  RAGGED_ROWS         Rows whose field count differs from the header. Usually an
                      unescaped delimiter inside a quoted value. Every column after
                      the break is shifted, so values land under the wrong names.

  TRUNCATION          Values clustered at exactly 255, 100 or 50 characters: the
                      signature of a VARCHAR limit upstream. The data is cut and
                      the file cannot tell you.

  NULL_AS_STRING      "NULL", "N/A", "-", "None" stored as text. Counts as a value,
                      passes a not-null check, poisons every average.

  EXCEL_DATE          The famous one. Excel converts gene symbols and identifiers to
                      dates: SEPT2 -> 2-Sep, MARCH1 -> 1-Mar. It got bad enough that
                      the HGNC renamed the affected genes rather than keep fighting
                      the spreadsheet.

  LEADING_ZERO_LOSS   Zip codes, account numbers and part IDs read as integers.
                      "01234" becomes 1234 and the join fails, or worse, matches
                      the wrong row.

Design
------
Column-wise, not file-wise. Corruption almost always sits in particular columns, and
a whole-file score averages it into invisibility.

No pandas dependency. It takes rows of strings, so it works on a csv.reader, a
database cursor, a list of lists, or a dataframe you converted with
.astype(str).values.tolist().
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, Sequence

from ..core import AuditResult, Finding, Severity, plural

# --- thresholds, deliberately visible and refittable by the pipeline -------------

MOJIBAKE_SUSPECT = 0.002      # fraction of cells showing mojibake signatures
MOJIBAKE_CORRUPT = 0.02

NUMERIC_AS_TEXT_SUSPECT = 0.95  # fraction of a text column that parses as a number
RAGGED_SUSPECT = 0.001        # fraction of rows with wrong field count
RAGGED_CORRUPT = 0.01

TRUNCATION_SUSPECT = 0.03     # fraction of values sitting exactly on a limit
# Of the values sitting exactly at a limit, at least this fraction must be DISTINCT.
# Below it, they are one repeated legitimate value rather than many cut ones.
TRUNCATION_MIN_DIVERSITY = 0.30
NULL_STRING_SUSPECT = 0.01
EXCEL_DATE_SUSPECT = 2        # absolute count; even one is worth knowing about
LEADING_ZERO_SUSPECT = 0.5

MIN_ROWS_FOR_STATS = 30

# Sequences that are overwhelmingly UTF-8-read-as-Latin-1/CP1252 rather than real
# text. Each is a real multi-byte character misread one byte at a time.
_MOJIBAKE = (
    "Ã¡", "Ã©", "Ã­", "Ã³", "Ãº", "Ã±", "Ã¼", "Ã¤", "Ã¶", "Ã ", "Ã¨",
    "â€™", "â€œ", "â€", "â€“", "â€”", "â€¦", "Â£", "Â©", "Â®", "Â°",
    "Â ", "Ã‰", "Ã–", "Ãœ", "ï»¿",
)

_NUMERIC = re.compile(r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?([eE][+-]?\d+)?$")
_NULLISH = frozenset({
    "null", "NULL", "Null", "n/a", "N/A", "NA", "na", "none", "None", "NONE",
    "-", "--", "?", "nan", "NaN", "NULL()", "#N/A", "#NULL!", "\\N", "",
})
# Excel turns these into dates. Pattern: 1-3 letters (month abbrev) + digits, or
# the date form Excel produces.
_EXCEL_DATE_OUT = re.compile(
    r"^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", re.I
)
_LIKELY_ID = re.compile(r"(id|code|zip|postal|account|sku|part|ref)$", re.I)
_TRUNCATION_LIMITS = (50, 100, 128, 200, 255, 256, 500, 512, 1000, 1024)


def _mojibake_hits(value: str) -> list[str]:
    return [m for m in _MOJIBAKE if m in value]


def _is_numeric(value: str) -> bool:
    return bool(_NUMERIC.match(value.strip()))


# A number wearing display formatting: thousands separators, a currency symbol, a
# trailing percent, or parenthesised negatives from a spreadsheet export.
_FORMATTED_NUMBER = re.compile(
    r"""^\s*(
        [-+]?[\$£€¥]\s?[\d,]+(\.\d+)?      |   # $1,234.56
        [-+]?\d{1,3}(,\d{3})+(\.\d+)?      |   # 1,234,567
        \(\s*[\$£€¥]?[\d,]+(\.\d+)?\s*\)   |   # (1,234)  negative
        [-+]?[\d,]+(\.\d+)?\s?%                # 12.5%
    )\s*$""",
    re.X,
)


def _is_formatted_number(value: str) -> bool:
    """True for a number formatted for display rather than storage.

    This distinction is what makes the check useful. A CSV holding `1234` is just an
    untyped file. A CSV holding `1,234` has a display format baked into the data, and
    every cast downstream will fail on it.
    """
    return bool(_FORMATTED_NUMBER.match(value))


def audit_column(
    name: str, values: Sequence[str], thresholds: dict[str, float] | None = None
) -> AuditResult:
    """Audit one column. Returns findings with the column named as the location."""
    t = {**_defaults(), **(thresholds or {})}
    findings: list[Finding] = []
    n = len(values)
    if n == 0:
        return AuditResult(subject=name, units=0, judged=False)

    non_empty = [v for v in values if v.strip()]
    judged = n >= MIN_ROWS_FOR_STATS

    # --- mojibake
    hits: Counter[str] = Counter()
    affected = 0
    for v in values:
        h = _mojibake_hits(v)
        if h:
            affected += 1
            hits.update(h)
    if affected:
        rate = affected / n
        sev = (
            Severity.CORRUPT if rate >= t["mojibake_rate_corrupt"]
            else Severity.SUSPECT if rate >= t["mojibake_rate"]
            else None
        )
        if sev:
            findings.append(Finding(
                mode="mojibake", severity=sev, location=f"column {name!r}",
                metric=rate, threshold=t["mojibake_rate"],
                detail=(
                    f"{affected} of {n} values contain UTF-8-read-as-Latin-1 "
                    f"sequences. This column was decoded with the wrong encoding"
                ),
                samples=tuple(s for s, _ in hits.most_common(5)),
            ))

    # --- numbers carrying display formatting
    #
    # NOT "numbers stored as text". In a CSV every value is text, because a CSV has
    # no types. Flagging every numeric column would fire on every ID and every amount
    # in every file ever written. That is noise, not a finding.
    #
    # What is worth reporting is a number carrying DISPLAY FORMATTING: thousands
    # separators, a currency symbol, a trailing percent. Those mean the value was
    # formatted for a person to read, and any downstream cast will fail or quietly
    # produce nulls. A bare "1234" needs no warning. "1,234" does.
    if judged and non_empty:
        formatted = [v for v in non_empty if _is_formatted_number(v)]
        numeric = sum(1 for v in non_empty if _is_numeric(v))
        ratio = len(formatted) / len(non_empty)
        if ratio >= t["numeric_as_text"] and len(formatted) >= MIN_ROWS_FOR_STATS:
            example = formatted[0]
            findings.append(Finding(
                mode="numeric_as_text", severity=Severity.SUSPECT,
                location=f"column {name!r}", metric=ratio,
                threshold=t["numeric_as_text"],
                detail=(
                    f"{ratio:.0%} of values are numbers carrying display formatting "
                    f"(e.g. {example!r}). A cast to a numeric type will fail or "
                    f"silently produce nulls, and aggregations will be wrong"
                ),
                samples=tuple(formatted[:5]),
            ))
        elif numeric == len(non_empty) and numeric >= MIN_ROWS_FOR_STATS:
            # Every value is a clean number. Worth knowing when the source HAS types
            # and chose text anyway; not worth saying about a CSV, which cannot.
            pass

    # --- truncation at a VARCHAR limit
    #
    # Diversity matters as much as the count. Real truncation cuts MANY DIFFERENT
    # values to the same length. One value repeated thousands of times at exactly 50
    # characters is a long name, not damage.
    #
    # Two checks, and real data showed I needed both:
    #
    #   max length   If any value is longer than the candidate limit, nothing is
    #                being cut at that limit.
    #   diversity    Real truncation cuts MANY DIFFERENT values to one length.
    #
    # Both came from running this on NYC 311 open data:
    #   agency_name           7,655 values at exactly 50 chars, all the SAME string
    #                         ("Department of Housing Preservation and Development",
    #                         a complete agency name). Caught by diversity; max
    #                         length was also 50, so the length check would not have.
    #   taxi_pick_up_location 24 DISTINCT values at 50 chars, all complete addresses.
    #                         Passed diversity; caught by max length, which is 60.
    if judged and non_empty:
        lengths = Counter(len(v) for v in non_empty)
        for limit in _TRUNCATION_LIMITS:
            at_limit = lengths.get(limit, 0)
            if at_limit < 3:
                continue
            # If anything in the column is LONGER than the candidate limit, then
            # nothing is being cut at that limit. Decisive, and cheap.
            if max(lengths) > limit:
                continue
            values_at_limit = [v for v in non_empty if len(v) == limit]
            distinct = len(set(values_at_limit))
            if distinct / max(1, at_limit) < TRUNCATION_MIN_DIVERSITY:
                continue
            rate = at_limit / len(non_empty)
            if rate >= t["truncation"]:
                findings.append(Finding(
                    mode="truncation", severity=Severity.SUSPECT,
                    location=f"column {name!r}", metric=rate,
                    threshold=t["truncation"],
                    detail=(
                        f"{at_limit} values are exactly {limit} characters. That is "
                        f"the signature of a VARCHAR({limit}) limit upstream, and "
                        f"these values are cut off"
                    ),
                    samples=tuple(
                        v[:40] + "..." for v in non_empty if len(v) == limit
                    )[:3],
                ))
                break

    # --- null-like strings
    if judged:
        nullish = [v for v in values if v in _NULLISH and v != ""]
        if nullish and len(nullish) / n >= t["null_string"]:
            findings.append(Finding(
                mode="null_as_string", severity=Severity.SUSPECT,
                location=f"column {name!r}", metric=len(nullish) / n,
                threshold=t["null_string"],
                detail=(
                    f"{len(nullish)} values are null-like strings rather than real "
                    f"nulls. They pass not-null checks and corrupt aggregates"
                ),
                samples=tuple(s for s, _ in Counter(nullish).most_common(5)),
            ))

    # --- Excel date corruption
    date_like = [v for v in non_empty if _EXCEL_DATE_OUT.match(v.strip())]
    if len(date_like) >= t["excel_date"] and len(date_like) < len(non_empty) * 0.5:
        findings.append(Finding(
            mode="excel_date_corruption", severity=Severity.CORRUPT,
            location=f"column {name!r}", metric=float(len(date_like)),
            threshold=t["excel_date"],
            detail=(
                f"{len(date_like)} values look like Excel auto-converted them to "
                f"dates (SEPT2 -> 2-Sep). The original identifiers are unrecoverable "
                f"from this file"
            ),
            samples=tuple(date_like[:5]),
        ))

    # --- leading zeros lost from an ID-like column
    if judged and _LIKELY_ID.search(name) and non_empty:
        numeric_vals = [v for v in non_empty if v.isdigit()]
        if numeric_vals:
            lengths = {len(v) for v in numeric_vals}
            has_zero_pad = any(v.startswith("0") for v in numeric_vals)
            if len(lengths) > 1 and not has_zero_pad:
                ratio = len(numeric_vals) / len(non_empty)
                if ratio >= t["leading_zero"]:
                    findings.append(Finding(
                        mode="leading_zero_loss", severity=Severity.SUSPECT,
                        location=f"column {name!r}", metric=ratio,
                        threshold=t["leading_zero"],
                        detail=(
                            f"identifier column with varying digit lengths "
                            f"({sorted(lengths)[:5]}) and no zero padding. Leading "
                            f"zeros were probably stripped by a numeric read. Joins "
                            f"on this column may silently fail or match the wrong row"
                        ),
                        samples=tuple(numeric_vals[:5]),
                    ))

    return AuditResult(
        findings=findings, units=n, judged=judged, subject=f"column {name}"
    )


def audit_rows(
    rows: Sequence[Sequence[str]],
    header: Sequence[str] | None = None,
    thresholds: dict[str, float] | None = None,
) -> list[AuditResult]:
    """Audit a whole table: structural checks, then every column.

    Pass `header` explicitly, or the first row gets used. Ragged rows are reported at
    table level because you cannot pin them on any single column. That is the damage:
    values shift out from under their names.
    """
    t = {**_defaults(), **(thresholds or {})}
    rows = [list(r) for r in rows]
    if not rows:
        return []

    if header is None:
        header, rows = list(rows[0]), rows[1:]
    header = list(header)
    results: list[AuditResult] = []

    # --- structural: ragged rows
    width = len(header)
    ragged = [(i, len(r)) for i, r in enumerate(rows) if len(r) != width]
    if ragged:
        rate = len(ragged) / max(1, len(rows))
        sev = (
            Severity.CORRUPT if rate >= t["ragged_corrupt"]
            else Severity.SUSPECT if rate >= t["ragged"]
            else None
        )
        if sev:
            results.append(AuditResult(
                subject="table structure", units=len(rows), judged=True,
                findings=[Finding(
                    mode="ragged_rows", severity=sev, metric=rate,
                    threshold=t["ragged"], location="table",
                    detail=(
                        f"{len(ragged)} of {len(rows)} rows have a field count "
                        f"different from the {width}-column header. Almost always an "
                        f"unescaped delimiter inside a quoted value. Every column "
                        f"after the break is shifted"
                    ),
                    samples=tuple(
                        f"row {i}: {c} fields" for i, c in ragged[:5]
                    ),
                )],
            ))

    # --- duplicate header names, which silently overwrite on load
    dupes = [n for n, c in Counter(header).items() if c > 1]
    if dupes:
        results.append(AuditResult(
            subject="table header", units=len(header), judged=True,
            findings=[Finding(
                mode="duplicate_columns", severity=Severity.SUSPECT,
                location="header", detail=(
                    f"{plural(len(dupes), 'column name')} used more than once. "
                    f"Most loaders keep only the last one and drop the rest without "
                    f"warning"
                ),
                samples=tuple(dupes[:5]),
            )],
        ))

    # --- per column
    for idx, name in enumerate(header):
        col = [r[idx] if idx < len(r) else "" for r in rows]
        results.append(audit_column(str(name), [str(v) for v in col], t))

    return results


def _defaults() -> dict[str, float]:
    return {
        "mojibake_rate": MOJIBAKE_SUSPECT,
        "mojibake_rate_corrupt": MOJIBAKE_CORRUPT,
        "numeric_as_text": NUMERIC_AS_TEXT_SUSPECT,
        "ragged": RAGGED_SUSPECT,
        "ragged_corrupt": RAGGED_CORRUPT,
        "truncation": TRUNCATION_SUSPECT,
        "null_string": NULL_STRING_SUSPECT,
        "excel_date": float(EXCEL_DATE_SUSPECT),
        "leading_zero": LEADING_ZERO_SUSPECT,
    }


class TabularDetector:
    """Pipeline-compatible wrapper. See `pipeline.Detector`."""

    name = "tabular"

    def features(self, artifact: Any) -> dict[str, float]:
        """Roll a table into scalar features so the pipeline can learn thresholds."""
        rows = artifact if isinstance(artifact, list) else list(artifact)
        if not rows:
            return {"units": 0, "judged": 0}

        header, body = list(rows[0]), rows[1:]
        cells = [str(v) for r in body for v in r]
        n = max(1, len(cells))

        mojibake = sum(1 for c in cells if _mojibake_hits(c)) / n
        nullish = sum(1 for c in cells if c in _NULLISH and c) / n
        ragged = (
            sum(1 for r in body if len(r) != len(header)) / max(1, len(body))
        )
        excel = sum(1 for c in cells if _EXCEL_DATE_OUT.match(c.strip()))

        return {
            "units": len(body),
            "judged": 1 if len(body) >= MIN_ROWS_FOR_STATS else 0,
            "mojibake_rate": mojibake,
            "null_string": nullish,
            "ragged": ragged,
            "excel_date": float(excel),
        }

    def findings(
        self, artifact: Any, feats: dict[str, float], thresholds: dict[str, float]
    ) -> list[Finding]:
        rows = artifact if isinstance(artifact, list) else list(artifact)
        out: list[Finding] = []
        for r in audit_rows(rows, thresholds=thresholds):
            out.extend(r.findings)
        return out

    def default_thresholds(self) -> dict[str, float]:
        return _defaults()

    def directions(self) -> dict[str, Literal["above_is_bad", "below_is_bad"]]:
        return {k: "above_is_bad" for k in _defaults()}
