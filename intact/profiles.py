"""
Profiles — severity depends on what the data is for.

A truncated `notes` column does not matter if you are building a search index, and is
fatal if you are keeping a compliance archive. Lost leading zeros are cosmetic in a
report and catastrophic the moment you join on that column. Mojibake in a name field
breaks entity resolution and does nothing to a topic model.

So reporting one fixed severity is not just imprecise, it is unhelpful to everybody.
Too alarming for the person who does not care. Too quiet for the person whose
pipeline it is about to destroy.

Severity here is a function of two things:

    how damaged is it   x   what are you doing with it

You declare the second. The library already measures the first.

    apply_profile(result, JOINS)         # leading zeros escalate to CORRUPT
    apply_profile(result, SEARCH_INDEX)  # the same finding drops to CLEAN

Why a small fixed set instead of free configuration
----------------------------------------------------
Six profiles cover most of what people actually do with tabular and document data,
and a short list gets read. A configuration format gets skipped, everyone runs the
default, and the feature may as well not exist. `custom()` covers the cases the six
do not.

The weights are judgement calls, not measurements. They sit in one table where you
can see and argue with them, instead of being buried in the detectors. Disagree with
a number and you can change it without touching any detection logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .core import AuditResult, Finding, Severity

# Multipliers applied to a finding's base severity, per use case.
#
#   0.0  irrelevant. Do not report it as a problem, just log it
#   0.5  worth knowing, not worth stopping for
#   1.0  as detected
#   2.0  escalate. This breaks the thing you are doing
#
# Read a column as "what this use case cares about".

_WEIGHTS: dict[str, dict[str, float]] = {
    # Full-text search, RAG chunking, topic modelling.
    # Word boundaries and character fidelity are everything. Numbers and long-field
    # truncation mostly are not. A truncated note still indexes.
    "search_index": {
        "shattered_words": 2.0,
        "missing_glyph": 2.0,
        "mojibake": 1.0,
        "numeric_loss": 0.0,
        "numeric_as_text": 0.0,
        "truncation": 0.5,
        "excel_date_corruption": 0.0,
        "leading_zero_loss": 0.0,
        "null_as_string": 0.5,
        "ragged_rows": 1.0,
        "duplicate_columns": 1.0,
    },
    # Sums, averages, financial reporting, anything where a figure is the output.
    "analytics": {
        "numeric_loss": 2.0,
        "numeric_as_text": 2.0,
        "null_as_string": 2.0,       # "N/A" counted as a value wrecks every mean
        "truncation": 1.0,
        "excel_date_corruption": 1.0,
        "ragged_rows": 2.0,          # values under the wrong column names
        "mojibake": 0.5,
        "shattered_words": 0.0,
        "missing_glyph": 0.5,
        "leading_zero_loss": 1.0,
        "duplicate_columns": 2.0,
    },
    # Joining datasets, deduplicating, matching records to a master list.
    # Key fidelity is the whole job; prose damage is irrelevant.
    "joins": {
        "leading_zero_loss": 2.0,    # 01234 -> 1234 and the join silently misses
        "mojibake": 2.0,             # names will not match across sources
        "truncation": 2.0,           # a cut key matches the wrong row
        "null_as_string": 1.0,
        "ragged_rows": 2.0,
        "duplicate_columns": 2.0,
        "numeric_as_text": 1.0,
        "excel_date_corruption": 1.0,
        "numeric_loss": 0.5,
        "shattered_words": 0.0,
        "missing_glyph": 0.0,
    },
    # Gene symbols, sample identifiers, measurements. Identifier mangling here has
    # caused enough published errors that gene names were changed to stop it.
    "scientific": {
        "excel_date_corruption": 2.0,
        "numeric_loss": 2.0,
        "truncation": 2.0,
        "leading_zero_loss": 2.0,
        "numeric_as_text": 1.0,
        "missing_glyph": 1.0,
        "null_as_string": 1.0,
        "ragged_rows": 2.0,
        "mojibake": 0.5,
        "shattered_words": 0.5,
        "duplicate_columns": 2.0,
    },
    # Legal hold, regulatory retention, audit trail. The archive must equal the
    # original. Nothing is acceptable damage, because the point is fidelity itself.
    "archive": {
        "truncation": 2.0,
        "numeric_loss": 2.0,
        "missing_glyph": 2.0,
        "excel_date_corruption": 2.0,
        "shattered_words": 2.0,
        "mojibake": 2.0,
        "leading_zero_loss": 2.0,
        "ragged_rows": 2.0,
        "null_as_string": 1.0,
        "numeric_as_text": 1.0,
        "duplicate_columns": 2.0,
    },
    # Sorting documents into buckets, routing, coarse triage. You need enough signal
    # to tell one topic from another and nothing more. Log damage; do not act on it.
    "classification": {
        "shattered_words": 0.5,
        "missing_glyph": 0.5,
        "mojibake": 0.0,
        "numeric_loss": 0.0,
        "numeric_as_text": 0.0,
        "truncation": 0.0,
        "excel_date_corruption": 0.0,
        "leading_zero_loss": 0.0,
        "null_as_string": 0.0,
        "ragged_rows": 1.0,
        "duplicate_columns": 0.5,
    },
}

_DESCRIPTIONS = {
    "search_index": "full-text search, RAG, topic modelling",
    "analytics": "aggregates, reporting, anything numeric",
    "joins": "joining, deduplication, record matching",
    "scientific": "identifiers and measurements",
    "archive": "legal hold, retention, audit trail",
    "classification": "bucketing and routing documents",
}


@dataclass(frozen=True)
class Profile:
    """What you are doing with the data, and therefore what damage matters."""

    name: str
    weights: Mapping[str, float]
    description: str = ""
    default_weight: float = 1.0

    def weight_for(self, mode: str) -> float:
        """Unknown modes keep their detected severity.

        A new detector should not get silently ignored just because no profile has an
        opinion about it yet. Silence is the failure this whole library is about.
        """
        return self.weights.get(mode, self.default_weight)

    def apply(self, finding: Finding) -> tuple[Severity, bool]:
        """Return the profile-adjusted severity, and whether to report it at all."""
        w = self.weight_for(finding.mode)
        if w == 0.0:
            return Severity.CLEAN, False
        rank = finding.severity.rank
        if w >= 2.0:
            rank = min(2, rank + 1)
        elif w <= 0.5:
            rank = max(0, rank - 1)
        return [Severity.CLEAN, Severity.SUSPECT, Severity.CORRUPT][rank], True


SEARCH_INDEX = Profile("search_index", _WEIGHTS["search_index"], _DESCRIPTIONS["search_index"])
ANALYTICS = Profile("analytics", _WEIGHTS["analytics"], _DESCRIPTIONS["analytics"])
JOINS = Profile("joins", _WEIGHTS["joins"], _DESCRIPTIONS["joins"])
SCIENTIFIC = Profile("scientific", _WEIGHTS["scientific"], _DESCRIPTIONS["scientific"])
ARCHIVE = Profile("archive", _WEIGHTS["archive"], _DESCRIPTIONS["archive"])
CLASSIFICATION = Profile("classification", _WEIGHTS["classification"], _DESCRIPTIONS["classification"])

# No profile declared. Everything reported as measured. That is the right answer
# when you do not yet know what the data is for, which happens a lot.
RAW = Profile("raw", {}, "no downstream use declared; report everything as measured")

ALL = [SEARCH_INDEX, ANALYTICS, JOINS, SCIENTIFIC, ARCHIVE, CLASSIFICATION, RAW]


def custom(name: str, **weights: float) -> Profile:
    """Build a profile for a use case the built-ins do not cover.

        mine = custom("ml_features", numeric_loss=2.0, mojibake=0.0)
    """
    return Profile(name, weights, "custom")


@dataclass
class ProfiledResult:
    """An audit re-scored for one downstream use."""

    profile: Profile
    severity: Severity
    reported: list[Finding] = field(default_factory=list)
    suppressed: list[Finding] = field(default_factory=list)
    subject: str = ""

    @property
    def is_usable(self) -> bool:
        return self.severity is not Severity.CORRUPT

    def __str__(self) -> str:
        subj = f"{self.subject}: " if self.subject else ""
        head = f"{subj}{self.severity.value} for {self.profile.name}"
        lines = [head]
        for f in self.reported:
            lines.append(f"  {f}")
        if self.suppressed:
            modes = ", ".join(sorted({f.mode for f in self.suppressed}))
            lines.append(
                f"  (not relevant to {self.profile.name}, logged only: {modes})"
            )
        return "\n".join(lines)


def apply_profile(result: AuditResult, profile: Profile) -> ProfiledResult:
    """Re-score one audit for a declared downstream use.

    Suppressed findings are kept, not thrown away. The measurement stays in the
    record, so you can re-score the same audit for a different use later without
    re-reading the data. That matters, because one file usually feeds several
    pipelines with different tolerances.
    """
    reported: list[Finding] = []
    suppressed: list[Finding] = []

    for f in result.findings:
        sev, show = profile.apply(f)
        if show:
            reported.append(Finding(
                mode=f.mode, severity=sev, detail=f.detail, metric=f.metric,
                threshold=f.threshold, samples=f.samples, location=f.location,
            ))
        else:
            suppressed.append(f)

    severity = (
        max((f.severity for f in reported), key=lambda s: s.rank)
        if reported else Severity.CLEAN
    )
    return ProfiledResult(
        profile=profile, severity=severity, reported=reported,
        suppressed=suppressed, subject=result.subject,
    )


def compare_profiles(
    results: Iterable[AuditResult], profiles: Iterable[Profile] = ()
) -> str:
    """Show how the same data scores across every use case.

    Often the most useful thing this library prints. A dataset that is CLEAN for
    search and CORRUPT for joins tells you exactly which pipeline can consume it and
    which cannot. One overall verdict cannot tell you that.
    """
    results = list(results)
    profiles = list(profiles) or [p for p in ALL if p.name != "raw"]

    rows: list[tuple[str, str, int, str]] = []
    for p in profiles:
        scored = [apply_profile(r, p) for r in results]
        worst = max((s.severity for s in scored), key=lambda s: s.rank) \
            if scored else Severity.CLEAN
        n_reported = sum(len(s.reported) for s in scored)
        drivers = sorted({
            f.mode for s in scored for f in s.reported
            if f.severity is Severity.CORRUPT
        })
        rows.append((p.name, worst.value, n_reported, ", ".join(drivers) or "-"))

    w = max(len(r[0]) for r in rows)
    lines = [
        f"{'use case'.ljust(w)}  {'verdict'.ljust(8)}  {'issues':>6}  blocking",
        f"{'-' * w}  {'-' * 8}  {'-' * 6}  {'-' * 30}",
    ]
    for name, verdict, n, drivers in rows:
        lines.append(f"{name.ljust(w)}  {verdict.ljust(8)}  {n:>6}  {drivers}")

    lines.append("")
    lines.append(
        "Same data, same measurements. Which pipelines may consume it differs."
    )
    return "\n".join(lines)
