"""
Core types shared by every detector.

Most data pipelines cannot tell you whether what they read is intact. They give you
back a dataframe, a string, a record count. Those look the same whether the source
was read correctly or mangled on the way in. Nothing throws an error. You get data,
it has the right shape, and some of it is wrong.

Every detector here answers one question about one artifact: is this what the source
actually said? It returns evidence rather than a bare boolean. If you are about to
quarantine ten thousand rows you should get to see why.

Three rules the detectors follow
--------------------------------
1. **Evidence, not verdicts.** A `Finding` carries the measured value, the threshold
   it crossed, and real examples. You can disagree with a threshold without losing
   the measurement.

2. **Refuse to judge too little.** Statistics on forty rows are noise. Detectors
   return nothing rather than guessing, and say so.

3. **Worst wins, never the average.** One corrupt page in fifty, one broken column
   in thirty. Averaging hides the thing you needed to know about.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


def plural(n: int, singular: str, many: str = "") -> str:
    """Count with a noun that agrees with it: plural(1, "row") -> "1 row".

    Report text gets read by people. "1 rows" tells them nobody read it back.
    """
    return f"{n:,} {singular if n == 1 else (many or singular + 's')}"


class Severity(str, Enum):
    """How bad it is, in three steps that map to three actions."""

    CLEAN = "clean"       # ingest it
    SUSPECT = "suspect"   # ingest it, but flag for review
    CORRUPT = "corrupt"   # quarantine it

    @property
    def rank(self) -> int:
        return {"clean": 0, "suspect": 1, "corrupt": 2}[self.value]


@dataclass(frozen=True)
class Finding:
    """One detected problem, carrying the evidence that produced it.

    `mode` is a free-form string rather than an enum so you can add a detector
    without editing this file. The cost is that there is no central registry of
    modes. The benefit is that a new detector is a new file and nothing else.
    """

    mode: str
    severity: Severity
    detail: str
    metric: float | None = None
    threshold: float | None = None
    samples: tuple[str, ...] = ()
    location: str = ""     # page 14, column "revenue", row 8821, wherever it is

    def __str__(self) -> str:
        where = f" @ {self.location}" if self.location else ""
        head = f"[{self.severity.value.upper()}] {self.mode}{where}: {self.detail}"
        if not self.samples:
            return head
        shown = ", ".join(repr(s) for s in self.samples[:5])
        return f"{head}\n    examples: {shown}"


@dataclass
class AuditResult:
    """Verdict for one artifact: a page, a column, a file, a table."""

    findings: list[Finding] = field(default_factory=list)
    units: int = 0             # rows, tokens, cells, whatever was examined
    judged: bool = True        # False when there was too little to say anything
    subject: str = ""

    @property
    def severity(self) -> Severity:
        if not self.findings:
            return Severity.CLEAN
        return max((f.severity for f in self.findings), key=lambda s: s.rank)

    @property
    def is_usable(self) -> bool:
        return self.severity is not Severity.CORRUPT

    @property
    def confidence(self) -> float:
        """1.0 clean, falling with each finding and how far past threshold it sits.

        Overshoot is capped. A metric ten times its threshold is not ten times worse
        than one at double, and letting it run makes the number meaningless.
        """
        score = 1.0
        for f in self.findings:
            overshoot = 1.0
            if f.metric is not None and f.threshold:
                try:
                    ratio = f.metric / f.threshold if f.threshold else 1.0
                    overshoot = min(3.0, max(1.0, abs(ratio)))
                except ZeroDivisionError:
                    overshoot = 1.0
            penalty = (0.45 if f.severity is Severity.CORRUPT else 0.15) * overshoot
            score -= penalty
        return max(0.0, min(1.0, score))

    def reasons(self) -> list[str]:
        return [str(f) for f in self.findings]

    def __str__(self) -> str:
        subj = f"{self.subject}: " if self.subject else ""
        if not self.judged:
            return f"{subj}not judged: too little data ({plural(self.units, 'unit')})"
        if not self.findings:
            return f"{subj}clean ({plural(self.units, 'unit')}, confidence {self.confidence:.2f})"
        lines = [
            f"{subj}{self.severity.value}, confidence {self.confidence:.2f}, "
            f"{plural(self.units, 'unit')}"
        ]
        lines.extend(f"  {f}" for f in self.findings)
        return "\n".join(lines)


def summarise(results: Iterable[AuditResult]) -> dict[str, object]:
    """Roll many artifact-level results into one verdict.

    Severity is the maximum, never the mean. A dataset is only as trustworthy as its
    worst column. Average one corrupt column across thirty clean ones and you get a
    reassuring number attached to an unusable dataset.

    Unjudged artifacts are left out of the confidence average, so a nearly-empty file
    cannot flatter the result.
    """
    results = list(results)
    if not results:
        return {
            "severity": Severity.CLEAN.value,
            "artifacts": 0,
            "confidence": 1.0,
            "modes": {},
            "flagged": [],
        }

    worst = max((r.severity for r in results), key=lambda s: s.rank)
    judged = [r for r in results if r.judged] or results

    modes: Counter[str] = Counter()
    for r in results:
        for f in r.findings:
            modes[f.mode] += 1

    flagged = [
        (r.subject or str(i)) for i, r in enumerate(results) if r.findings
    ]

    return {
        "severity": worst.value,
        "artifacts": len(results),
        "artifacts_judged": len(judged),
        "confidence": round(sum(r.confidence for r in judged) / len(judged), 3),
        "modes": dict(modes),
        "flagged": flagged,
    }


def report(results: Iterable[AuditResult], show_clean: bool = False) -> str:
    """Human-readable report, worst first.

    The ordering matters. List artifacts in file order and the one corrupt column
    ends up on page four of the output, which is about as useful as not finding it.
    """
    results = list(results)
    s = summarise(results)

    lines = [
        f"severity   : {s['severity']}",
        f"artifacts  : {s['artifacts']} ({s['artifacts_judged']} judged)",
        f"confidence : {s['confidence']}",
    ]
    if s["modes"]:
        modes = ", ".join(f"{k} x{v}" for k, v in sorted(s["modes"].items()))
        lines.append(f"modes      : {modes}")
    lines.append("")

    ordered = sorted(
        results, key=lambda r: (-r.severity.rank, r.confidence)
    )
    for r in ordered:
        if r.findings or show_clean:
            lines.append(str(r))
            lines.append("")

    if not s["modes"]:
        lines.append("No problems detected.")

    return "\n".join(lines).rstrip()
