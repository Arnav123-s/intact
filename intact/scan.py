"""
Scan — point it at anything, get one report.

`sources.resolve` turns a path, folder, pattern or URL into parsed sources.
`detectors/*` audit rows or text. This joins the two, so the common case is one call:

    print(scan("data/"))
    print(scan("https://example.org/export.csv", profile=JOINS))

What it adds beyond gluing
---------------------------
**Routing.** Tabular sources get the tabular and consistency detectors. Text sources
get the text detector. Running a CSV through a prose-shattering check produces
confident nonsense, so nothing runs where it does not apply.

**Cross-file findings.** Auditing thirty files one at a time misses what only shows
up across them: a column that is `customer_id` in one export and `CustomerID` in
another, or four files that agree on a date format and one that does not. Those get
reported at the end, because in a folder scan they are usually the finding.

**One verdict for the folder.** Worst wins, same as everywhere else here. A directory
is only as trustworthy as its worst file.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .core import AuditResult, Finding, Severity, plural, summarise
from .detectors import consistency, tabular, text as textdet
from .profiles import Profile, RAW, apply_profile
from .sources import Source, resolve


@dataclass
class ScanResult:
    """Everything found across everything scanned."""

    sources: list[Source] = field(default_factory=list)
    per_source: dict[str, list[AuditResult]] = field(default_factory=dict)
    cross_file: list[Finding] = field(default_factory=list)
    unreadable: list[Source] = field(default_factory=list)
    profile: Profile = RAW

    @property
    def severity(self) -> Severity:
        worst = Severity.CLEAN
        for results in self.per_source.values():
            for r in results:
                if r.severity.rank > worst.rank:
                    worst = r.severity
        for f in self.cross_file:
            if f.severity.rank > worst.rank:
                worst = f.severity
        return worst

    def __str__(self) -> str:
        return self.report()

    def report(self, show_clean: bool = False) -> str:
        lines: list[str] = []

        n_flagged = sum(
            1 for results in self.per_source.values()
            if any(r.findings for r in results)
        )
        lines += [
            f"scanned    : {plural(len(self.sources), 'source')}",
            f"flagged    : {n_flagged}",
            f"unreadable : {len(self.unreadable)}",
            f"verdict    : {self.severity.value}"
            + (f"  (for {self.profile.name})" if self.profile is not RAW else ""),
            "",
        ]

        if self.unreadable:
            lines.append("Could not be read:")
            for s in self.unreadable:
                why = next((n for n in s.notes if "could not read" in n), "unknown")
                lines.append(f"  {s.name}: {why}")
            lines.append("")

        # Worst sources first. A report in file order buries the one that matters.
        order = sorted(
            self.per_source.items(),
            key=lambda kv: -max(
                (r.severity.rank for r in kv[1]), default=0
            ),
        )
        for name, results in order:
            flagged = [r for r in results if r.findings]
            if not flagged and not show_clean:
                continue
            lines.append(f"--- {name} ---")
            if not flagged:
                lines.append("  clean")
            for r in flagged:
                lines.append(str(r))
            lines.append("")

        if self.cross_file:
            lines.append("Across files:")
            for f in self.cross_file:
                lines.append(f"  {f}")
            lines.append("")

        if not n_flagged and not self.cross_file and not self.unreadable:
            lines.append("Nothing found.")

        return "\n".join(lines).rstrip()


def _audit_source(s: Source) -> list[AuditResult]:
    """Run whichever detectors apply to this kind of source."""
    if s.rows:
        results = tabular.audit_rows(s.rows)
        # Consistency is additive. It finds convention breaks the fixed rules cannot
        # see, and stays quiet where a column has no convention.
        for c in consistency.audit_rows(s.rows):
            if c.findings:
                results.append(c)
        return results
    if s.text:
        return [textdet.audit_text(s.text)]
    return []


def _cross_file(sources: Sequence[Source]) -> list[Finding]:
    """Findings that only exist when you compare several sources.

    Two checks. Both are invisible file-by-file and expensive to discover later:

      inconsistent headers  the same field named differently across exports, which
                            breaks a union or an append and does it silently
      odd file out          one file whose columns disagree with the majority
    """
    tabular_sources = [s for s in sources if s.rows and len(s.rows) > 0]
    if len(tabular_sources) < 2:
        return []

    findings: list[Finding] = []

    # --- header names that differ only by case/punctuation across files
    normalised: dict[str, set[str]] = defaultdict(set)
    for s in tabular_sources:
        for col in s.rows[0]:
            key = "".join(ch for ch in str(col).lower() if ch.isalnum())
            if key:
                normalised[key].add(str(col))

    inconsistent = {k: v for k, v in normalised.items() if len(v) > 1}
    if inconsistent:
        examples = [
            " / ".join(sorted(v)) for v in list(inconsistent.values())[:5]
        ]
        findings.append(Finding(
            mode="inconsistent_headers",
            severity=Severity.SUSPECT,
            metric=float(len(inconsistent)),
            location="across files",
            detail=(
                f"{plural(len(inconsistent), 'field')} named differently across "
                f"files. A union or append will produce extra columns full of nulls "
                f"instead of an error"
            ),
            samples=tuple(examples),
        ))

    # --- one file whose shape disagrees with the rest
    shapes = Counter(
        tuple(sorted(str(c).lower() for c in s.rows[0])) for s in tabular_sources
    )
    if len(shapes) > 1:
        (common, n_common), = shapes.most_common(1)
        if n_common >= 2 and n_common / len(tabular_sources) >= 0.6:
            odd = [
                s.name for s in tabular_sources
                if tuple(sorted(str(c).lower() for c in s.rows[0])) != common
            ]
            findings.append(Finding(
                mode="odd_file_out",
                severity=Severity.SUSPECT,
                metric=float(len(odd)),
                location="across files",
                detail=(
                    f"{n_common} of {len(tabular_sources)} files share a column "
                    f"layout. {plural(len(odd), 'file')} did not. Check these are the "
                    f"same kind of export before you combine them"
                ),
                samples=tuple(odd[:5]),
            ))

    return findings


def scan(
    target: str | Path | Sequence[str | Path],
    profile: Profile = RAW,
    recursive: bool = True,
) -> ScanResult:
    """Audit anything: a file, a folder, a glob, a URL, or a list of those.

        scan("export.csv")
        scan("data/", profile=JOINS)
        scan("https://example.org/data.csv")
        scan(["a.csv", "reports/", "*.jsonl"])

    Pass a `profile` to score findings against what the data is for. Without one,
    everything gets reported as measured.
    """
    out = ScanResult(profile=profile)

    for s in resolve(target, recursive=recursive):
        if any("could not read" in n for n in s.notes):
            out.unreadable.append(s)
            continue

        out.sources.append(s)
        results = _audit_source(s)

        if profile is not RAW:
            results = [
                AuditResult(
                    findings=p.reported, units=r.units,
                    judged=r.judged, subject=r.subject,
                )
                for r, p in ((r, apply_profile(r, profile)) for r in results)
            ]

        out.per_source[s.name] = results

    out.cross_file = _cross_file(out.sources)
    if profile is not RAW:
        kept: list[Finding] = []
        for f in out.cross_file:
            sev, show = profile.apply(f)
            if show:
                kept.append(Finding(
                    mode=f.mode, severity=sev, detail=f.detail,
                    metric=f.metric, threshold=f.threshold,
                    samples=f.samples, location=f.location,
                ))
        out.cross_file = kept

    return out
