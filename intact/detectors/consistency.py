"""
Consistency — learn a file's own conventions, then flag what breaks them.

Every other detector here applies a rule written in advance: 255 characters is
suspicious, these strings mean null, this regex is a date. Rules written in advance
are wrong in ways their author cannot see coming. That is how my truncation rule
produced two false positives on real NYC data before I added the guards.

This one carries no rules about what corruption looks like. It learns what *this
column* looks like and reports what does not fit.

    Every date in this column reads 2026-09-02. One cell says 2-Sep.
    That cell breaks this file's own convention.

Why it is worth having separately
----------------------------------
1. **No labels, no training, no prior knowledge.** It works on the first file it ever
   sees, because the thing it compares against is the file itself.

2. **It catches things nobody thought of.** A fixed rule only finds corruption its
   author knew about. This finds anything that breaks a pattern, including failure
   modes that do not have a name yet.

3. **It can tell apart what fixed rules cannot.** The Excel-date detector cannot tell
   a mangled gene symbol from a real September date, because both match the same
   regex. This one can. If every other value in the column is an ISO date, `2-Sep`
   does not belong. If the column is full of `%d-%b` dates, it does.

4. **It stays quiet on consistent data, however strange.** 7,655 identical agency
   names are perfectly consistent, so it reports nothing. That is the right answer,
   and it is the one my length-based rule got wrong.

How it works
------------
Each value gets reduced to a **shape**: its structural skeleton with the specific
characters thrown away.

    "2026-09-02"   ->  "dddd-dd-dd"
    "2-Sep"        ->  "d-Mmm"
    "(555) 010-9"  ->  "(ddd) ddd-d"

Then it looks at the shape distribution. A column where one shape covers almost
everything has a convention, and values outside it are anomalies. A column with no
dominant shape has no convention, so nothing gets reported. Free text is not an
error.

The dominance threshold is deliberately high. A column split 60/40 between two shapes
has two legitimate formats, not 40% corruption.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from ..core import AuditResult, Finding, Severity, plural

# A column needs at least this fraction on one shape before it is treated as having a
# convention at all. Set high on purpose: two competing formats is not corruption.
DOMINANCE = 0.90

# And the minority must be small enough to read as exceptions rather than a variant.
MAX_ANOMALY_RATE = 0.05

# Below this many values, shape statistics are noise.
MIN_VALUES = 40

# A shape seen this many times or more is a format, not an anomaly, even when rare.
MIN_ANOMALY_ISOLATION = 3

_MONTHS = (
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
)


def shape(value: str, keep_length: bool = False) -> str:
    """Reduce a value to its structural skeleton.

    Digits become `d`, letters become `a` or `A`, and runs are collapsed so "12" and
    "123456" do not read as different formats. Month abbreviations survive as their
    own token, because `2-Sep` and `2-XYZ` are structurally identical but only one of
    them is a date, and that difference is the whole point of this detector.

    Punctuation is kept verbatim. It carries the format.
    """
    v = value.strip()
    if not v:
        return ""

    out: list[str] = []
    i = 0
    n = len(v)

    while i < n:
        ch = v[i]

        if ch.isdigit():
            j = i
            while j < n and v[j].isdigit():
                j += 1
            run = j - i
            out.append("d" * run if keep_length else ("d" * min(run, 4)))
            i = j

        elif ch.isalpha():
            j = i
            while j < n and v[j].isalpha():
                j += 1
            word = v[i:j]
            if len(word) == 3 and word.lower() in _MONTHS:
                out.append("Mmm")
            elif word.isupper():
                out.append("A" * min(len(word), 4))
            elif word[0].isupper():
                out.append("A" + "a" * min(len(word) - 1, 3))
            else:
                out.append("a" * min(len(word), 4))
            i = j

        elif ch.isspace():
            out.append(" ")
            while i < n and v[i].isspace():
                i += 1

        else:
            out.append(ch)
            i += 1

    return "".join(out)


def audit_column(
    name: str, values: Sequence[str], dominance: float = DOMINANCE
) -> AuditResult:
    """Report values that break their column's own dominant convention."""
    non_empty = [v for v in values if v and v.strip()]
    if len(non_empty) < MIN_VALUES:
        return AuditResult(
            subject=f"column {name}", units=len(values), judged=False
        )

    shapes = [shape(v) for v in non_empty]
    counts = Counter(shapes)
    top_shape, top_n = counts.most_common(1)[0]
    share = top_n / len(shapes)

    # No dominant shape means no convention. Free text is not an error.
    if share < dominance:
        return AuditResult(subject=f"column {name}", units=len(values), judged=True)

    anomalies = [
        (v, s) for v, s in zip(non_empty, shapes) if s != top_shape
    ]
    if not anomalies:
        return AuditResult(subject=f"column {name}", units=len(values), judged=True)

    rate = len(anomalies) / len(non_empty)
    if rate > MAX_ANOMALY_RATE:
        # Enough of them to be a second legitimate format rather than damage.
        return AuditResult(subject=f"column {name}", units=len(values), judged=True)

    by_shape = Counter(s for _, s in anomalies)

    # Rare shapes are the interesting ones. A shape appearing many times is a variant
    # the column tolerates. A shape appearing twice is an accident.
    isolated = {s for s, c in by_shape.items() if c < MIN_ANOMALY_ISOLATION}
    examples = [v for v, s in anomalies if s in isolated][:5]
    if not examples:
        examples = [v for v, _ in anomalies[:5]]

    severity = Severity.SUSPECT if rate < 0.01 else Severity.CORRUPT

    detail = (
        f"{share:.0%} of values follow the pattern {top_shape!r}. "
        f"{plural(len(anomalies), 'value')} did not. Nothing here knows what this "
        f"column is meant to hold. These values just do not match what the rest of "
        f"the column does"
    )

    return AuditResult(
        subject=f"column {name}",
        units=len(values),
        judged=True,
        findings=[Finding(
            mode="convention_break",
            severity=severity,
            metric=rate,
            threshold=MAX_ANOMALY_RATE,
            location=f"column {name!r}",
            detail=detail,
            samples=tuple(f"{v!r} (shape {shape(v)!r})" for v in examples),
        )],
    )


def audit_rows(
    rows: Sequence[Sequence[str]], header: Sequence[str] | None = None
) -> list[AuditResult]:
    """Run the consistency check over every column of a table."""
    rows = [list(r) for r in rows]
    if not rows:
        return []
    if header is None:
        header, rows = list(rows[0]), rows[1:]
    header = list(header)

    out: list[AuditResult] = []
    for i, name in enumerate(header):
        col = [str(r[i]) if i < len(r) else "" for r in rows]
        out.append(audit_column(str(name), col))
    return out


def describe_conventions(
    rows: Sequence[Sequence[str]], header: Sequence[str] | None = None
) -> str:
    """Report what conventions were learned, not just what broke them.

    Useful on its own. Being told what a file's columns actually look like often
    tells you more than being told nothing is wrong. It also makes the anomaly
    findings readable, because you can see the pattern they broke.
    """
    rows = [list(r) for r in rows]
    if not rows:
        return "no rows"
    if header is None:
        header, rows = list(rows[0]), rows[1:]
    header = list(header)

    lines = ["Learned conventions (nothing was told to this detector in advance):", ""]
    width = max(len(str(h)) for h in header)

    for i, name in enumerate(header):
        col = [str(r[i]) for r in rows if i < len(r) and str(r[i]).strip()]
        if len(col) < MIN_VALUES:
            lines.append(f"  {str(name).ljust(width)}  (too few values to judge)")
            continue
        counts = Counter(shape(v) for v in col)
        top, n = counts.most_common(1)[0]
        share = n / len(col)
        if share >= DOMINANCE:
            lines.append(
                f"  {str(name).ljust(width)}  {share:>4.0%} follow {top!r}"
                + (f"  ({plural(len(counts) - 1, 'other shape')})"
                   if len(counts) > 1 else "")
            )
        else:
            lines.append(
                f"  {str(name).ljust(width)}  no dominant pattern "
                f"({len(counts)} shapes, top {share:.0%}): free text, not checked"
            )
    return "\n".join(lines)
