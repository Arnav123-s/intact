"""
Solve — hand it a broken file, get back usable data.

The rest of this library reports. This module fixes. That difference matters more
than it sounds: a tool that says "your export is corrupted, ask your vendor to send
it again" has given you a second problem, not solved the first. Nobody wants a report.
They want the data to work.

So the primary output here is a working dataset. The report is a by-product you read
only if you care why.

    result = solve("vendor-export.csv")
    result.rows          # usable data, repaired
    result.quarantined   # the rows that could not be saved, kept separately
    result.report        # what was done, if you want it

What it actually fixes, rather than reports
--------------------------------------------
ENCODING       Does not tell you the encoding was wrong. Tries the plausible
               candidates, scores each by how much mojibake it produces, and reads
               the file correctly.

DELIMITER      Does not tell you rows are ragged. Sniffs the real delimiter and
               quote character, re-parses, and checks whether that fixed the row
               widths. Usually it does — ragged rows are nearly always a parse
               problem, not a data problem.

TRUNCATION     Cannot restore cut text. Can isolate the affected rows so the other
               94% flows through instead of blocking the whole load.

EXCEL DATES    For gene symbols this is genuinely reversible — the mangling is a
               finite known mapping, and the reverse table is published. Where a
               domain mapping exists, it is applied; where it does not, the rows are
               quarantined rather than guessed at.

The rule
--------
**Fix what can be fixed. Isolate what cannot. Never guess into the output.**

Anything uncertain goes to `quarantined`, which is a dataset in its own right — the
same columns, the same order, just held back. That is the difference between
solving and refusing: the good data moves, and the bad data is somewhere specific
rather than somewhere unknown.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .core import Finding, Severity
from .detectors.tabular import audit_rows
from .repair import fix_mojibake, fix_null_string, fix_numeric_text

# Encodings worth trying, in the order they are worth trying. utf-8-sig first because
# a BOM is common in exports and produces a visible artifact if read as plain utf-8.
_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1", "utf-16")

_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "ï»¿", "�")

# Excel mangles gene symbols into dates. The mapping is finite and the reverse is
# published (Ziemann et al. 2016 documented the scale of it; the HGNC eventually
# renamed the genes). Only symbols whose date form is unambiguous are listed — a
# reverse map that guesses is worse than one that abstains.
GENE_DATE_MAP: dict[str, str] = {
    "1-Mar": "MARCH1", "2-Mar": "MARCH2", "3-Mar": "MARCH3", "4-Mar": "MARCH4",
    "5-Mar": "MARCH5", "6-Mar": "MARCH6", "7-Mar": "MARCH7", "8-Mar": "MARCH8",
    "9-Mar": "MARCH9", "10-Mar": "MARCH10", "11-Mar": "MARCH11",
    "1-Sep": "SEPT1", "2-Sep": "SEPT2", "3-Sep": "SEPT3", "4-Sep": "SEPT4",
    "5-Sep": "SEPT5", "6-Sep": "SEPT6", "7-Sep": "SEPT7", "8-Sep": "SEPT8",
    "9-Sep": "SEPT9", "10-Sep": "SEPT10", "11-Sep": "SEPT11", "12-Sep": "SEPT12",
    "14-Sep": "SEPT14", "15-Sep": "SEPT15",
    "1-Dec": "DEC1", "2-Dec": "DEC2",
    "1-Oct": "OCT1", "2-Oct": "OCT2", "3-Oct": "OCT3", "4-Oct": "OCT4",
    "6-Oct": "OCT6", "11-Oct": "OCT11",
}


@dataclass
class Action:
    """One thing that was actually done, not one thing that was observed."""

    what: str
    detail: str
    affected: int = 0

    def __str__(self) -> str:
        n = f" ({self.affected})" if self.affected else ""
        return f"{self.what}{n}: {self.detail}"


@dataclass
class Solution:
    """Working data, plus whatever could not be saved, plus what was done."""

    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    quarantined: list[list[str]] = field(default_factory=list)
    quarantine_reasons: list[str] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    encoding: str = ""
    delimiter: str = ""

    @property
    def recovered_fraction(self) -> float:
        total = len(self.rows) + len(self.quarantined)
        return len(self.rows) / total if total else 1.0

    @property
    def report(self) -> str:
        lines = [
            f"read as        : {self.encoding}, delimiter {self.delimiter!r}",
            f"usable rows    : {len(self.rows)}",
            f"quarantined    : {len(self.quarantined)}",
            f"recovered      : {self.recovered_fraction:.1%}",
            "",
        ]
        if self.actions:
            lines.append("Fixed:")
            lines.extend(f"  {a}" for a in self.actions)
        else:
            lines.append("Nothing needed fixing.")
        if self.quarantined:
            lines.append("")
            lines.append(f"Held back ({len(self.quarantined)} rows):")
            for r in Counter(self.quarantine_reasons).most_common():
                lines.append(f"  {r[1]} rows — {r[0]}")
            lines.append("")
            lines.append(
                "These are in `.quarantined` with the same columns. The rest of the "
                "data is usable now; nothing is blocked waiting on them."
            )
        return "\n".join(lines)


# --- detection that leads to an action, not a message ---------------------------


def _looks_like_utf16(raw: bytes) -> bool:
    """UTF-16 text is full of null bytes; ASCII and UTF-8 are not.

    This guard exists because of a real bug. Scoring encodings purely by mojibake
    markers let UTF-16 win on plain ASCII: decoding ASCII as UTF-16 produces
    CJK-looking characters that contain no mojibake markers at all, so it scored
    zero — a perfect score — and the file was silently turned into garbage.

    A byte-order mark is decisive. Failing that, real UTF-16 of mostly-Latin text has
    a null byte for roughly every other character; anything under a third is not
    UTF-16 whatever it decodes to.
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return True
    sample = raw[:4096]
    if not sample:
        return False
    return sample.count(0) / len(sample) > 0.30


def detect_encoding(raw: bytes) -> tuple[str, int]:
    """Pick the encoding that decodes with the least damage.

    Scored by mojibake markers and replacement characters rather than by whether the
    decode raises — latin-1 decodes anything without error and is wrong most of the
    time, so "it didn't throw" is not evidence.

    Ties go to the earlier candidate, because `_ENCODINGS` is ordered by how likely
    each is in practice.
    """
    # A byte-order mark is not a hint, it is a declaration. Nothing else gets a vote,
    # because every 8-bit codec will happily decode UTF-16 bytes into scoreless
    # garbage and win on a tie.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16", 0
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig", 0

    best, best_score = "utf-8", 10**9
    utf16_plausible = _looks_like_utf16(raw)

    for enc in _ENCODINGS:
        # utf-8-sig is only meaningful with a BOM, which is handled above; without
        # one it is just utf-8 and reporting it would misdescribe the file.
        if enc == "utf-8-sig":
            continue
        if enc == "utf-16" and not utf16_plausible:
            continue
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        score = sum(text.count(m) for m in _MOJIBAKE_MARKERS)
        if score < best_score:
            best, best_score = enc, score
    return best, best_score


def detect_dialect(text: str) -> tuple[str, str]:
    """Find the delimiter and quote char that produce consistent row widths.

    csv.Sniffer is tried first and usually right. When it is not, candidates are
    scored by how many rows come out with the modal width — the correct dialect is
    the one that makes the table rectangular.
    """
    sample = "\n".join(text.splitlines()[:50])
    try:
        d = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return d.delimiter, d.quotechar or '"'
    except csv.Error:
        pass

    best, best_score = ",", -1
    for delim in (",", ";", "\t", "|"):
        try:
            rows = list(csv.reader(io.StringIO(sample), delimiter=delim))
        except csv.Error:
            continue
        if not rows:
            continue
        widths = Counter(len(r) for r in rows)
        modal, count = widths.most_common(1)[0]
        score = count if modal > 1 else 0
        if score > best_score:
            best, best_score = delim, score
    return best, '"'


# --- the main entry point --------------------------------------------------------


def solve(
    source: str | Path | bytes,
    fix_genes: bool = True,
    quarantine_truncated: bool = True,
) -> Solution:
    """Read a broken CSV and return usable data.

    Does not ask you to fix anything upstream. Reads it correctly, repairs what is
    reversible, applies domain mappings where one exists, and isolates the rest.
    """
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    sol = Solution()

    # 1. Read it correctly rather than reporting that it was read wrongly.
    encoding, damage = detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")
    sol.encoding = encoding
    if encoding != "utf-8":
        sol.actions.append(Action(
            "encoding", f"detected {encoding} and read with it "
            f"(utf-8 would have produced mojibake)",
        ))

    # 2. Parse it correctly rather than reporting ragged rows.
    delim, quote = detect_dialect(text)
    sol.delimiter = delim
    rows = list(csv.reader(io.StringIO(text), delimiter=delim, quotechar=quote))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return sol

    header, body = rows[0], rows[1:]
    sol.header = header
    width = len(header)

    if delim != ",":
        sol.actions.append(Action(
            "delimiter", f"detected {delim!r}, not a comma — parsed with it",
        ))

    ragged = [r for r in body if len(r) != width]
    if ragged:
        sol.actions.append(Action(
            "ragged rows", f"{len(ragged)} rows had the wrong field count after "
            f"parsing; held back rather than shifted into the wrong columns",
            len(ragged),
        ))

    keep: list[list[str]] = []
    for r in body:
        if len(r) == width:
            keep.append(list(r))
        else:
            sol.quarantined.append(list(r))
            sol.quarantine_reasons.append(
                f"field count {len(r)}, expected {width} — unescaped delimiter"
            )

    # 3. Repair cell by cell. Only reversible transforms touch the output.
    counts: Counter[str] = Counter()
    for row in keep:
        for i, v in enumerate(row):
            fixed = fix_mojibake(v)
            if fixed is not None:
                row[i] = fixed
                counts["mojibake"] += 1
                continue
            fixed = fix_numeric_text(v)
            if fixed is not None:
                row[i] = fixed
                counts["numeric_as_text"] += 1
                continue
            fixed = fix_null_string(v)
            if fixed is not None and v.strip():
                row[i] = fixed
                counts["null_as_string"] += 1

    for mode, n in counts.items():
        sol.actions.append(Action(
            mode,
            {
                "mojibake": "re-decoded mis-encoded characters back to the original",
                "numeric_as_text": "removed thousands separators so values parse as numbers",
                "null_as_string": "replaced null-like strings with real empty values",
            }[mode],
            n,
        ))

    # 4. Domain recovery where a real mapping exists.
    if fix_genes:
        recovered = 0
        for row in keep:
            for i, v in enumerate(row):
                sym = GENE_DATE_MAP.get(v.strip())
                if sym:
                    row[i] = sym
                    recovered += 1
        if recovered:
            sol.actions.append(Action(
                "excel_date_corruption",
                "reversed Excel's gene-symbol mangling using the published mapping "
                "(2-Sep -> SEPT2). Only unambiguous symbols were restored",
                recovered,
            ))

    # 5. Isolate what genuinely cannot be recovered.
    if quarantine_truncated:
        audits = audit_rows([header] + keep)
        truncated_cols = [
            a.subject.replace("column ", "")
            for a in audits
            if any(f.mode == "truncation" for f in a.findings)
        ]
        if truncated_cols:
            idxs = [header.index(c) for c in truncated_cols if c in header]
            lengths = {
                i: Counter(len(r[i]) for r in keep if i < len(r)) for i in idxs
            }
            limits = {
                i: max(
                    (L for L, n in c.items()
                     if n >= 3 and L in (50, 100, 128, 200, 255, 256, 500, 512)),
                    default=None,
                )
                for i, c in lengths.items()
            }
            survivors: list[list[str]] = []
            for r in keep:
                cut = next(
                    (header[i] for i, lim in limits.items()
                     if lim and i < len(r) and len(r[i]) == lim),
                    None,
                )
                if cut:
                    sol.quarantined.append(r)
                    sol.quarantine_reasons.append(
                        f"value in {cut!r} truncated at a length limit — the rest of "
                        f"the text is not in this file"
                    )
                else:
                    survivors.append(r)
            if len(survivors) != len(keep):
                sol.actions.append(Action(
                    "truncation",
                    f"isolated rows with cut-off values so the remaining "
                    f"{len(survivors)} rows are usable now",
                    len(keep) - len(survivors),
                ))
            keep = survivors

    sol.rows = keep
    return sol


def solve_to_files(
    source: str | Path, out_dir: str | Path = "."
) -> tuple[Path, Path | None, Path]:
    """Solve a file and write three outputs: clean data, quarantine, report.

    Three files rather than one because they have three different readers — the
    pipeline consumes the clean data, whoever owns the source deals with the
    quarantine, and a human reads the report once.
    """
    src = Path(source)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sol = solve(src)

    clean = out / f"{src.stem}.clean.csv"
    with clean.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(sol.header)
        w.writerows(sol.rows)

    quarantine: Path | None = None
    if sol.quarantined:
        quarantine = out / f"{src.stem}.quarantine.csv"
        with quarantine.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(sol.header) + ["__reason"])
            for row, reason in zip(sol.quarantined, sol.quarantine_reasons):
                w.writerow(list(row) + [reason])

    report = out / f"{src.stem}.report.txt"
    report.write_text(sol.report, encoding="utf-8")
    return clean, quarantine, report
