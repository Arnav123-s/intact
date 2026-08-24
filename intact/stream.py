"""
Stream — constant memory, however large the input.

`solve()` reads a file into memory. That is fine for an export you could open in a
spreadsheet and useless for the ones you cannot. Those are exactly the files where
silent corruption hurts most. Nobody eyeballs a ten-gigabyte file, so nothing catches
the column that stopped being a number two million rows in.

This module does the same work in bounded memory.

The trick is that nothing here needs the data twice
----------------------------------------------------
Every statistic the detectors use is an aggregate, and every aggregate has an
incremental form:

    "12% of values are fragments"        -> two counters
    "18 values sit at exactly 255 chars" -> a Counter over lengths
    "digits appear in 0.03% of tokens"   -> two counters

So a column's verdict comes from counters that grow with the number of *distinct
lengths*, not the number of rows. A billion-row file and a thousand-row file use the
same memory.

Two things really do need bounded state, and both are capped:

    examples          first N per finding, so a report can show evidence
    length histogram  distinct lengths only, capped and then coarsened

What it costs
-------------
One pass, no random access, so anything needing a second look at an earlier row is
out. In practice that rules out cross-row deduplication and any "compare this value
to the median" rule. Everything here is single-pass and any future detector should be
too. That constraint is what keeps this usable on files you cannot hold.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence, TextIO

from .core import AuditResult, Finding, Severity, plural
from .repair import fix_mojibake, fix_null_string, fix_numeric_text
from .solve import GENE_DATE_MAP, _MOJIBAKE_MARKERS, detect_dialect, detect_encoding

# Bytes read from the head of a file to decide encoding and dialect. Big enough to
# be representative, small enough to be free.
SNIFF_BYTES = 256 * 1024

# Cap on distinct lengths tracked per column. Beyond this the histogram is coarsened
# into buckets, which preserves the truncation signal (a spike at exactly 255) while
# bounding memory on columns of free text.
MAX_DISTINCT_LENGTHS = 4096

MAX_EXAMPLES = 5

_NUMERIC = re.compile(r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?([eE][+-]?\d+)?$")
_DIGIT = re.compile(r"\d")
_NULLISH = frozenset({
    "null", "NULL", "Null", "n/a", "N/A", "NA", "na", "none", "None", "NONE",
    "-", "--", "?", "nan", "NaN", "#N/A", "#NULL!", "\\N",
})
_EXCEL_DATE = re.compile(
    r"^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", re.I
)
_TRUNCATION_LIMITS = (50, 100, 128, 200, 255, 256, 500, 512, 1000, 1024)


@dataclass
class ColumnStats:
    """Everything known about one column, in memory that does not grow with rows."""

    name: str
    seen: int = 0
    non_empty: int = 0
    mojibake: int = 0
    numeric: int = 0
    nullish: int = 0
    excel_dates: int = 0
    control_chars: int = 0
    total_chars: int = 0
    digit_bearing: int = 0
    lengths: Counter[int] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)
    # Distinct values seen at each candidate truncation limit. Real truncation cuts
    # many different values to one length. A single repeated long name does not.
    # Capped, so a genuinely truncated column cannot grow this without bound.
    at_limit: dict[int, set[str]] = field(default_factory=dict)

    def _example(self, mode: str, value: str) -> None:
        bucket = self.examples.setdefault(mode, [])
        if len(bucket) < MAX_EXAMPLES and value not in bucket:
            bucket.append(value[:60])

    def update(self, value: str) -> None:
        """Fold one cell in. O(len(value)), no allocation proportional to rows."""
        self.seen += 1
        v = value.strip()
        self.total_chars += len(value)

        if v:
            self.non_empty += 1
            self.lengths[len(value)] += 1
            if len(value) in _TRUNCATION_LIMITS:
                seen = self.at_limit.setdefault(len(value), set())
                if len(seen) < 64:
                    seen.add(value)
            if len(self.lengths) > MAX_DISTINCT_LENGTHS:
                self._coarsen()

        if any(m in value for m in _MOJIBAKE_MARKERS):
            self.mojibake += 1
            self._example("mojibake", value)

        if v and _NUMERIC.match(v):
            self.numeric += 1

        if v in _NULLISH and v:
            self.nullish += 1
            self._example("null_as_string", v)

        if v and _EXCEL_DATE.match(v):
            self.excel_dates += 1
            self._example("excel_date_corruption", v)

        if _DIGIT.search(value):
            self.digit_bearing += 1

        for ch in value:
            if ch in "\t\n\r":
                continue
            if unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn"):
                self.control_chars += 1
                self._example("missing_glyph", f"U+{ord(ch):04X}")

    def _coarsen(self) -> None:
        """Keep exact counts for the lengths that matter; bucket the rest.

        Truncation shows up as a spike at a specific round length, so those are
        preserved exactly. Everything else is rounded into buckets of 10, which
        bounds memory without losing the signal we are looking for.
        """
        kept: Counter[int] = Counter()
        for length, n in self.lengths.items():
            if length in _TRUNCATION_LIMITS:
                kept[length] += n
            else:
                kept[(length // 10) * 10] += n
        self.lengths = kept

    def finalise(self, min_rows: int = 30) -> AuditResult:
        """Turn accumulated counters into findings."""
        findings: list[Finding] = []
        judged = self.seen >= min_rows
        n = max(1, self.seen)

        if self.mojibake:
            rate = self.mojibake / n
            sev = Severity.CORRUPT if rate >= 0.02 else Severity.SUSPECT
            findings.append(Finding(
                mode="mojibake", severity=sev, metric=rate, threshold=0.002,
                location=f"column {self.name!r}",
                detail=(
                    f"{self.mojibake} of {self.seen} values contain "
                    f"UTF-8-read-as-Latin-1 sequences"
                ),
                samples=tuple(self.examples.get("mojibake", [])),
            ))

        if judged and self.non_empty:
            ratio = self.numeric / self.non_empty
            if ratio >= 0.95 and self.numeric >= min_rows:
                findings.append(Finding(
                    mode="numeric_as_text", severity=Severity.SUSPECT,
                    metric=ratio, threshold=0.95,
                    location=f"column {self.name!r}",
                    detail=(
                        f"{ratio:.0%} of values are numbers carrying display "
                        f"formatting. A cast to a numeric type will fail or silently "
                        f"produce nulls"
                    ),
                ))

            longest = max(self.lengths) if self.lengths else 0
            for limit in _TRUNCATION_LIMITS:
                at = self.lengths.get(limit, 0)
                distinct = len(self.at_limit.get(limit, ()))
                # Anything longer than the limit means nothing is cut at it.
                if longest > limit:
                    continue
                # One repeated value at the limit is a long name, not truncation.
                if at >= 3 and distinct < 2:
                    continue
                if at >= 3 and at / self.non_empty >= 0.03:
                    findings.append(Finding(
                        mode="truncation", severity=Severity.SUSPECT,
                        metric=at / self.non_empty, threshold=0.03,
                        location=f"column {self.name!r}",
                        detail=(
                            f"{at} values are exactly {limit} characters, the "
                            f"signature of a VARCHAR({limit}) limit upstream"
                        ),
                    ))
                    break

            if self.nullish and self.nullish / n >= 0.01:
                findings.append(Finding(
                    mode="null_as_string", severity=Severity.SUSPECT,
                    metric=self.nullish / n, threshold=0.01,
                    location=f"column {self.name!r}",
                    detail=(
                        f"{self.nullish} values are null-like strings rather than "
                        f"real nulls, so they pass not-null checks"
                    ),
                    samples=tuple(self.examples.get("null_as_string", [])),
                ))

        if self.excel_dates >= 2 and self.excel_dates < self.non_empty * 0.5:
            findings.append(Finding(
                mode="excel_date_corruption", severity=Severity.CORRUPT,
                metric=float(self.excel_dates), threshold=2.0,
                location=f"column {self.name!r}",
                detail=(
                    f"{self.excel_dates} values look like Excel auto-converted "
                    f"identifiers to dates (SEPT2 -> 2-Sep)"
                ),
                samples=tuple(self.examples.get("excel_date_corruption", [])),
            ))

        if self.total_chars:
            rate = (self.control_chars / self.total_chars) * 1000
            if rate >= 0.5:
                findings.append(Finding(
                    mode="missing_glyph",
                    severity=Severity.CORRUPT if rate >= 3.0 else Severity.SUSPECT,
                    metric=rate, threshold=0.5,
                    location=f"column {self.name!r}",
                    detail=f"{rate:.2f} undecodable characters per 1000",
                    samples=tuple(self.examples.get("missing_glyph", [])),
                ))

        return AuditResult(
            findings=findings, units=self.seen, judged=judged,
            subject=f"column {self.name}",
        )


@dataclass
class StreamResult:
    header: list[str] = field(default_factory=list)
    encoding: str = ""
    delimiter: str = ""
    rows_in: int = 0
    rows_out: int = 0
    rows_quarantined: int = 0
    repairs: Counter[str] = field(default_factory=Counter)
    quarantine_reasons: Counter[str] = field(default_factory=Counter)
    audits: list[AuditResult] = field(default_factory=list)
    peak_columns: int = 0

    @property
    def recovered_fraction(self) -> float:
        return self.rows_out / self.rows_in if self.rows_in else 1.0

    @property
    def report(self) -> str:
        lines = [
            f"read as     : {self.encoding}, delimiter {self.delimiter!r}",
            f"rows in     : {self.rows_in:,}",
            f"rows out    : {self.rows_out:,}",
            f"quarantined : {self.rows_quarantined:,}",
            f"recovered   : {self.recovered_fraction:.1%}",
            "",
        ]
        if self.repairs:
            lines.append("Fixed:")
            for mode, n in self.repairs.most_common():
                lines.append(f"  {mode}: {n:,}")
        if self.quarantine_reasons:
            lines.append("")
            lines.append("Held back:")
            for reason, n in self.quarantine_reasons.most_common():
                lines.append(f"  {plural(n, 'row')}: {reason}")
        return "\n".join(lines)


def solve_stream(
    source: str | Path,
    out_clean: str | Path,
    out_quarantine: str | Path | None = None,
    chunk_rows: int = 50_000,
    fix_genes: bool = True,
    progress: Callable[[int], None] | None = None,
) -> StreamResult:
    """Repair a file of any size in one pass, writing output as it goes.

    Memory is bounded by the number of columns and their distinct value lengths,
    not by the number of rows. A billion-row file uses the same memory as a
    thousand-row one.

    `progress` is called with the running row count every `chunk_rows` rows, so a
    long job can report without this module knowing anything about how you display it.
    """
    src = Path(source)
    result = StreamResult()

    # Sniff a bounded head of the file rather than reading it whole.
    with src.open("rb") as fh:
        head = fh.read(SNIFF_BYTES)
    encoding, _ = detect_encoding(head)
    delimiter, quote = detect_dialect(head.decode(encoding, errors="replace"))
    result.encoding, result.delimiter = encoding, delimiter

    clean_path = Path(out_clean)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_fh: TextIO | None = None
    quarantine_writer = None

    stats: dict[str, ColumnStats] = {}

    with src.open("r", encoding=encoding, errors="replace", newline="") as fin, \
            clean_path.open("w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter=delimiter, quotechar=quote)
        writer = csv.writer(fout)

        try:
            header = next(reader)
        except StopIteration:
            return result

        result.header = header
        result.peak_columns = len(header)
        writer.writerow(header)
        for name in header:
            stats[name] = ColumnStats(name=name)

        width = len(header)

        for row in reader:
            result.rows_in += 1
            if progress and result.rows_in % chunk_rows == 0:
                progress(result.rows_in)

            if len(row) != width:
                result.rows_quarantined += 1
                result.quarantine_reasons[
                    f"field count {len(row)}, expected {width} (unescaped delimiter)"
                ] += 1
                if out_quarantine:
                    if quarantine_writer is None:
                        quarantine_fh = Path(out_quarantine).open(
                            "w", encoding="utf-8", newline=""
                        )
                        quarantine_writer = csv.writer(quarantine_fh)
                        quarantine_writer.writerow(header + ["__reason"])
                    quarantine_writer.writerow(
                        list(row) + ["field count mismatch"]
                    )
                continue

            out_row = list(row)
            for i, value in enumerate(out_row):
                stats[header[i]].update(value)

                fixed = fix_mojibake(value)
                if fixed is not None:
                    out_row[i] = fixed
                    result.repairs["mojibake"] += 1
                    continue
                fixed = fix_numeric_text(value)
                if fixed is not None:
                    out_row[i] = fixed
                    result.repairs["numeric_as_text"] += 1
                    continue
                fixed = fix_null_string(value)
                if fixed is not None and value.strip():
                    out_row[i] = fixed
                    result.repairs["null_as_string"] += 1
                    continue
                if fix_genes:
                    sym = GENE_DATE_MAP.get(value.strip())
                    if sym:
                        out_row[i] = sym
                        result.repairs["excel_date_corruption"] += 1

            writer.writerow(out_row)
            result.rows_out += 1

    if quarantine_fh is not None:
        quarantine_fh.close()

    result.audits = [s.finalise() for s in stats.values()]
    return result
