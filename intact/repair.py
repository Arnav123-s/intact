"""
Repair — fix what is provably fixable, and refuse the rest.

The useful question about corrupted data is not "can you clean it?" It is:

    Is the original information still present in this file, or is it gone?

Those need opposite responses. Mojibake is a reversible byte-level mistake: the
original characters are still there, encoded wrongly, and one decode reverses it
exactly. An Excel-mangled gene symbol is not: SEPT2 became 2-Sep, and no amount of
cleverness recovers which of SEPT2 or a September date it was. One of those you fix.
The other you must go back to the source for, and any tool that "cleans" it is
manufacturing data.

So every repair here is classified, and the classification is enforced:

  REVERSIBLE    The transformation is provably invertible. Applied on request.
  LOSSY         Recoverable in most cases but can guess wrong. Requires opt-in and
                reports every value it changed.
  IRRECOVERABLE Information is destroyed. Never repaired. Reported so you know to
                re-export rather than proceed.

Three rules
-----------
1. **Never guess silently.** A repair that might be wrong reports every value it
   touched, so the change is reviewable.
2. **Always keep the original.** Every `Repair` carries before and after. A repair
   you cannot undo is a second corruption.
3. **Refusing is a result.** `IRRECOVERABLE` is the most valuable verdict this module
   produces, because it is the one that stops you shipping a dataset you believe is
   clean.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Sequence

from .core import Finding, Severity


class Recoverability(str, Enum):
    REVERSIBLE = "reversible"
    LOSSY = "lossy"
    IRRECOVERABLE = "irrecoverable"


@dataclass
class Repair:
    """One applied change, with everything needed to review or undo it."""

    mode: str
    recoverability: Recoverability
    location: str
    changed: int
    total: int
    examples: tuple[tuple[str, str], ...] = ()   # (before, after)
    note: str = ""

    def __str__(self) -> str:
        head = (
            f"[{self.recoverability.value}] {self.mode} @ {self.location}: "
            f"{self.changed}/{self.total} values changed"
        )
        if self.note:
            head += f"\n    {self.note}"
        if self.examples:
            shown = "; ".join(f"{b!r} -> {a!r}" for b, a in self.examples[:3])
            head += f"\n    e.g. {shown}"
        return head


@dataclass
class Refusal:
    """A problem this module will not repair, and why."""

    mode: str
    location: str
    affected: int
    reason: str
    action: str
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        head = (
            f"[REFUSED] {self.mode} @ {self.location}: {self.affected} values\n"
            f"    why : {self.reason}\n"
            f"    do  : {self.action}"
        )
        if self.examples:
            head += f"\n    e.g. {', '.join(repr(s) for s in self.examples[:3])}"
        return head


@dataclass
class RepairResult:
    repairs: list[Repair] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)

    @property
    def is_clean_after(self) -> bool:
        """True only if nothing was refused.

        A file with repairs applied and no refusals is trustworthy. A file with even
        one refusal is not, no matter how much else was fixed — which is why this is
        an `and`, not a score.
        """
        return not self.refusals

    def __str__(self) -> str:
        lines: list[str] = []
        if self.repairs:
            lines.append(f"Repaired ({len(self.repairs)}):")
            lines.extend(f"  {r}" for r in self.repairs)
        if self.refusals:
            if lines:
                lines.append("")
            lines.append(f"NOT repairable ({len(self.refusals)}):")
            lines.extend(f"  {r}" for r in self.refusals)
            lines.append("")
            lines.append(
                "This data cannot be made correct by processing. Re-export from the "
                "source with the fixes above, or proceed knowing these values are wrong."
            )
        if not lines:
            lines.append("Nothing to repair.")
        return "\n".join(lines)


# --- individual repairs ----------------------------------------------------------

_MOJIBAKE_MARKERS = ("Ã", "Â", "â€", "ï»¿", "Ã©", "Ã¡")


def fix_mojibake(value: str) -> str | None:
    """Reverse UTF-8-decoded-as-Latin-1. Returns None if it does not apply.

    The check is strict on purpose. Round-tripping text that was never mojibake can
    corrupt legitimate Latin-1 content, so this only fires when the result both
    decodes cleanly AND removes the marker sequences that indicated the problem.
    """
    if not any(m in value for m in _MOJIBAKE_MARKERS):
        return None
    try:
        fixed = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if fixed == value:
        return None
    # Only accept if the repair actually removed the evidence.
    if any(m in fixed for m in _MOJIBAKE_MARKERS):
        return None
    return fixed


_NUMERIC = re.compile(r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?([eE][+-]?\d+)?$")


def fix_numeric_text(value: str) -> str | None:
    """Strip thousands separators so a numeric-looking string parses.

    Deliberately conservative: returns a normalised string rather than a float, so
    precision is never lost to a binary float on the way through. The caller decides
    the target type.
    """
    v = value.strip()
    if not _NUMERIC.match(v) or "," not in v:
        return None
    return v.replace(",", "")


_NULLISH = frozenset({
    "null", "NULL", "Null", "n/a", "N/A", "NA", "na", "none", "None", "NONE",
    "-", "--", "?", "nan", "NaN", "#N/A", "#NULL!", "\\N",
})


def fix_null_string(value: str) -> str | None:
    """Map null-like strings to an empty string (a real null on write)."""
    return "" if value.strip() in _NULLISH else None


def fix_duplicate_headers(header: Sequence[str]) -> list[str] | None:
    """Suffix duplicated column names so none is silently dropped on load."""
    counts = Counter(header)
    if all(c == 1 for c in counts.values()):
        return None
    seen: Counter[str] = Counter()
    out: list[str] = []
    for name in header:
        seen[name] += 1
        out.append(name if counts[name] == 1 else f"{name}__{seen[name]}")
    return out


def fix_leading_zeros(values: Sequence[str], width: int | None = None) -> dict[str, str] | None:
    """Zero-pad identifiers to a consistent width. LOSSY — it infers the width.

    If the true width is not supplied, the most common length is assumed. That is
    usually right and occasionally wrong, which is exactly why this is classified
    LOSSY and reports every value it changes.
    """
    digits = [v for v in values if v.isdigit()]
    if len(digits) < 3:
        return None
    if width is None:
        width = Counter(len(v) for v in digits).most_common(1)[0][0]
    mapping = {v: v.zfill(width) for v in digits if len(v) < width}
    return mapping or None


# --- refusals: named so the reason is explicit -----------------------------------

_IRRECOVERABLE: dict[str, tuple[str, str]] = {
    "excel_date_corruption": (
        "Excel replaced the original value with a date. The mapping is many-to-one "
        "and not invertible — '2-Sep' could have been SEPT2, or an actual date.",
        "Re-export from the source with the column formatted as Text before opening "
        "in Excel, or open the file with a tool that does not auto-convert.",
    ),
    "truncation": (
        "The characters beyond the length limit are not present in this file. There "
        "is nothing to restore them from.",
        "Re-export from the source with a wider column, or query the source directly.",
    ),
    "numeric_loss": (
        "Digits were written as unmapped bytes. The numeric values do not exist in "
        "this file in any form.",
        "Re-extract from the original PDF with a different engine, or obtain the "
        "data from its source rather than the document.",
    ),
    "missing_glyph": (
        "The font's character map omits these characters. Inferring them from "
        "context would be fabrication, not repair.",
        "Re-extract with OCR, or obtain a copy with complete embedded fonts.",
    ),
    "shattered_words": (
        "Word boundaries were lost during extraction. Rejoining requires guessing "
        "where words ended, and a wrong guess is indistinguishable from real text.",
        "Re-extract with a layout-aware engine, or OCR the page images.",
    ),
    "ragged_rows": (
        "Field counts do not match the header, so values sit under the wrong column "
        "names. Which value belongs where cannot be recovered from the damaged rows.",
        "Re-export with proper quoting, or re-parse the original with the correct "
        "delimiter and quote character.",
    ),
}


def repair_column(
    name: str,
    values: Sequence[str],
    findings: Iterable[Finding],
    allow_lossy: bool = False,
) -> tuple[list[str], RepairResult]:
    """Apply every safe repair to one column, refusing what cannot be recovered.

    Returns the repaired values and a full account of what was and was not done.
    The originals are never modified in place.
    """
    out = list(values)
    result = RepairResult()
    modes = {f.mode for f in findings}
    loc = f"column {name!r}"

    for f in findings:
        if f.mode in _IRRECOVERABLE:
            reason, action = _IRRECOVERABLE[f.mode]
            # `metric` is a ratio for most detectors but an absolute count for a
            # few (excel dates). Treating a count as a ratio produced an "affected"
            # figure larger than the table. Anything >1 is already a count.
            if f.metric is None:
                affected = len(f.samples)
            elif f.metric > 1:
                affected = int(f.metric)
            else:
                affected = int(round(f.metric * len(values)))
            result.refusals.append(Refusal(
                mode=f.mode, location=loc, affected=affected,
                reason=reason, action=action, examples=f.samples[:3],
            ))

    def _apply(mode: str, fn: Callable[[str], str | None],
               recov: Recoverability, note: str = "") -> None:
        examples: list[tuple[str, str]] = []
        changed = 0
        for i, v in enumerate(out):
            new = fn(v)
            if new is not None and new != v:
                if len(examples) < 5:
                    examples.append((v, new))
                out[i] = new
                changed += 1
        if changed:
            result.repairs.append(Repair(
                mode=mode, recoverability=recov, location=loc,
                changed=changed, total=len(out),
                examples=tuple(examples), note=note,
            ))

    if "mojibake" in modes:
        _apply("mojibake", fix_mojibake, Recoverability.REVERSIBLE,
               "decoded as latin-1 then re-decoded as utf-8; exactly invertible")
    if "null_as_string" in modes:
        _apply("null_as_string", fix_null_string, Recoverability.REVERSIBLE,
               "null-like strings replaced with empty values")
    if "numeric_as_text" in modes:
        _apply("numeric_as_text", fix_numeric_text, Recoverability.REVERSIBLE,
               "thousands separators removed; values kept as strings so no "
               "precision is lost to float conversion")

    if "leading_zero_loss" in modes:
        if allow_lossy:
            mapping = fix_leading_zeros(out)
            if mapping:
                examples = list(mapping.items())[:5]
                changed = 0
                for i, v in enumerate(out):
                    if v in mapping:
                        out[i] = mapping[v]
                        changed += 1
                result.repairs.append(Repair(
                    mode="leading_zero_loss", recoverability=Recoverability.LOSSY,
                    location=loc, changed=changed, total=len(out),
                    examples=tuple(examples),
                    note=("width inferred from the most common length — verify these "
                          "against the source before relying on joins"),
                ))
        else:
            result.refusals.append(Refusal(
                mode="leading_zero_loss", location=loc,
                affected=sum(1 for v in out if v.isdigit()),
                reason=("the original width is not recorded in the file, so padding "
                        "requires inferring it"),
                action=("pass allow_lossy=True to pad to the most common width, or "
                        "supply the correct width explicitly"),
            ))

    return out, result


def summarise_recoverability(results: Iterable[RepairResult]) -> str:
    """One-line-per-problem verdict across a whole dataset."""
    results = list(results)
    repairs = [r for res in results for r in res.repairs]
    refusals = [r for res in results for r in res.refusals]

    lines = [
        f"repaired      : {len(repairs)}",
        f"unrepairable  : {len(refusals)}",
    ]
    if refusals:
        by_mode = Counter(r.mode for r in refusals)
        lines.append("")
        lines.append("This dataset cannot be made correct by processing:")
        for mode, n in by_mode.most_common():
            lines.append(f"  {mode}: {n} location(s)")
        lines.append("")
        lines.append("Go back to the source. Processing further only hides it.")
    elif repairs:
        lines.append("")
        lines.append("All detected problems were reversible. The repaired data is sound.")
    return "\n".join(lines)
