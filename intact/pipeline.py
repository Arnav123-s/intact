"""
The pipeline — detection, logging and learning as one object.

Keeping the feedback loop as a separate module people are supposed to remember to
call is how it ends up never being called. Here it is wired in:

    pipeline.audit(x)      runs detectors, applies current thresholds, logs features
    pipeline.quarantine()  what needs a human look, worst first
    pipeline.label(...)    your decision, which becomes training data
                           thresholds refit automatically once there are enough

There is no separate training step, and nothing degrades if you never label anything
— the shipped defaults apply unchanged. The system starts useful and gets better at
*your* data specifically, because your data is the only data it learns from.

Why thresholds and not a model
------------------------------
A one-dimensional threshold per feature, fitted by sweeping for best balanced
accuracy, is close to the simplest thing that can learn. That is deliberate:

  - It works on ~40 labels. A gradient-boosted model on 40 rows learns the noise.
  - You can read the result. "flag when mojibake_rate >= 0.004" is auditable;
    a tree ensemble's decision surface is not.
  - It cannot fail silently. A threshold that drifts somewhere absurd is visible
    the moment you print it.

For a data-quality gate — where a false negative silently poisons everything
downstream — legibility beats accuracy. If it ever needs to be a real model, the
feature log is already there, and nothing else has to change.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

from .core import AuditResult, Finding, Severity, report, summarise

Label = Literal["keep", "discard"]

# Below this many labels, or this many of either class, thresholds are not fitted.
# A boundary drawn through nine examples is confident and wrong.
MIN_LABELS_TO_FIT = 40
MIN_PER_CLASS = 10

# A learned threshold only displaces a default if it is meaningfully better, so
# ordinary noise cannot churn the configuration run to run.
MIN_IMPROVEMENT = 0.03


class Detector(Protocol):
    """Anything that can look at an artifact and report findings.

    Deliberately minimal. A detector needs a name, the features it measures (so the
    pipeline can log and learn them), and a way to turn an artifact into findings
    given the current thresholds.
    """

    name: str

    def features(self, artifact: Any) -> dict[str, float]:
        """Measured values, independent of any threshold."""
        ...

    def findings(
        self, artifact: Any, feats: dict[str, float], thresholds: dict[str, float]
    ) -> list[Finding]:
        """Apply thresholds to features and describe what crossed them."""
        ...

    def default_thresholds(self) -> dict[str, float]:
        ...

    def directions(self) -> dict[str, Literal["above_is_bad", "below_is_bad"]]:
        """Which side of each threshold indicates a problem."""
        ...


@dataclass
class AuditRecord:
    """One audit, with everything needed to re-derive its verdict later.

    Features are logged, not verdicts. A verdict is a threshold applied to a feature;
    log the feature and any past verdict can be recomputed, and every threshold
    change can be replayed against the whole history. Log the verdict and that
    information is gone for good.
    """

    record_id: str
    subject: str
    detector: str
    features: dict[str, float]
    severity: str
    ts: float
    label: Label | None = None
    note: str = ""


class FeedbackLog:
    """Append-only feature log. The pipeline's memory.

    Append-only is not fastidiousness. Thresholds are derived from this file, so if
    rows can be edited in place, no threshold in the system is reproducible. A
    changed mind is a new row; the latest row for a record_id wins.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, rec: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(rec)) + "\n")

    def all(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        latest: dict[str, AuditRecord] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = AuditRecord(**json.loads(line))
                    latest[r.record_id] = r
        return list(latest.values())

    def labelled(self, detector: str | None = None) -> list[AuditRecord]:
        return [
            r for r in self.all()
            if r.label is not None and (detector is None or r.detector == detector)
        ]

    def unlabelled(self, detector: str | None = None) -> list[AuditRecord]:
        return [
            r for r in self.all()
            if r.label is None and (detector is None or r.detector == detector)
        ]


def _balanced_accuracy(
    values: list[float], labels: list[Label], threshold: float, direction: str
) -> float:
    """Mean of sensitivity and specificity.

    Balanced, not raw, because corpora are lopsided. If 95% of files are clean, a
    detector that flags nothing scores 95% on raw accuracy and is worthless.
    """
    tp = fp = tn = fn = 0
    for v, lab in zip(values, labels):
        flagged = v >= threshold if direction == "above_is_bad" else v <= threshold
        bad = lab == "discard"
        if flagged and bad:
            tp += 1
        elif flagged:
            fp += 1
        elif bad:
            fn += 1
        else:
            tn += 1
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return (sens + spec) / 2


@dataclass
class LearnedThreshold:
    feature: str
    value: float
    default: float
    accuracy: float
    default_accuracy: float
    n_labels: int
    adopted: bool

    def __str__(self) -> str:
        tag = "ADOPTED" if self.adopted else "kept default"
        return (
            f"{self.feature}: {self.value:.5g} (default {self.default:.5g}) — "
            f"{self.accuracy:.1%} vs {self.default_accuracy:.1%} on "
            f"{self.n_labels} labels — {tag}"
        )


class Pipeline:
    """Detect, log, and learn — one object, one call site.

    Example
    -------
        pipe = Pipeline([TextDetector()], log_path="audit-log.jsonl")

        result = pipe.audit(extracted_text, subject="paper-1841.pdf")
        if not result.is_usable:
            quarantine(doc)

        # later, review what it flagged
        for rec in pipe.quarantine():
            print(rec.subject, rec.features)
            pipe.label(rec.record_id, "discard")   # or "keep"

        print(pipe.learning_report())
    """

    def __init__(
        self,
        detectors: Iterable[Detector],
        log_path: str | Path | None = None,
        on_quarantine: Callable[[AuditResult], None] | None = None,
    ):
        self.detectors = list(detectors)
        self.log = FeedbackLog(log_path) if log_path else None
        self.on_quarantine = on_quarantine
        self._threshold_cache: dict[str, dict[str, float]] = {}
        self._cache_stamp: int = -1

    # --- thresholds -------------------------------------------------------------

    def thresholds_for(self, detector: Detector) -> dict[str, float]:
        """Current thresholds: learned where the evidence supports it, else default.

        Cached against the label count so a long run does not refit on every call,
        but a newly labelled batch takes effect without restarting anything.
        """
        if self.log is None:
            return detector.default_thresholds()

        stamp = len(self.log.labelled())
        if stamp != self._cache_stamp:
            self._threshold_cache.clear()
            self._cache_stamp = stamp

        if detector.name in self._threshold_cache:
            return self._threshold_cache[detector.name]

        learned = self.fit(detector)
        thresholds = dict(detector.default_thresholds())
        if learned:
            for feat, lt in learned.items():
                if lt.adopted:
                    thresholds[feat] = lt.value

        self._threshold_cache[detector.name] = thresholds
        return thresholds

    def fit(self, detector: Detector) -> dict[str, LearnedThreshold] | None:
        """Refit one detector's thresholds from labelled history.

        Returns None — explicitly, rather than quietly handing back defaults — when
        there is not enough labelled data. The caller can then distinguish "learned,
        and it agreed with the defaults" from "has not learned anything yet", which
        are very different states to be in.
        """
        if self.log is None:
            return None

        rows = self.log.labelled(detector.name)
        keeps = sum(1 for r in rows if r.label == "keep")
        discards = len(rows) - keeps

        if len(rows) < MIN_LABELS_TO_FIT:
            return None
        if keeps < MIN_PER_CLASS or discards < MIN_PER_CLASS:
            return None

        defaults = detector.default_thresholds()
        directions = detector.directions()
        labels: list[Label] = [r.label for r in rows]  # type: ignore[misc]

        out: dict[str, LearnedThreshold] = {}
        for feat, default in defaults.items():
            values = [r.features.get(feat, 0.0) for r in rows]
            direction = directions.get(feat, "above_is_bad")

            baseline = _balanced_accuracy(values, labels, default, direction)

            candidates = sorted(set(values))
            if len(candidates) > 1:
                candidates += [
                    (a + b) / 2 for a, b in zip(candidates, candidates[1:])
                ]

            best_v, best_s = default, baseline
            for c in candidates:
                s = _balanced_accuracy(values, labels, c, direction)
                if s > best_s:
                    best_v, best_s = c, s

            adopted = best_s >= baseline + MIN_IMPROVEMENT
            out[feat] = LearnedThreshold(
                feature=feat,
                value=best_v if adopted else default,
                default=default,
                accuracy=best_s,
                default_accuracy=baseline,
                n_labels=len(rows),
                adopted=adopted,
            )
        return out

    # --- the main call ----------------------------------------------------------

    def audit(self, artifact: Any, subject: str = "") -> AuditResult:
        """Run every detector, apply current thresholds, log the features."""
        all_findings: list[Finding] = []
        units = 0
        judged = True

        for det in self.detectors:
            feats = det.features(artifact)
            thresholds = self.thresholds_for(det)
            found = det.findings(artifact, feats, thresholds)
            all_findings.extend(found)

            units = max(units, int(feats.get("units", 0)))
            if not feats.get("judged", 1):
                judged = False

            if self.log is not None:
                self.log.append(AuditRecord(
                    record_id=uuid.uuid4().hex[:12],
                    subject=subject,
                    detector=det.name,
                    features={k: v for k, v in feats.items() if k != "judged"},
                    severity=(
                        max((f.severity for f in found), key=lambda s: s.rank).value
                        if found else Severity.CLEAN.value
                    ),
                    ts=time.time(),
                ))

        result = AuditResult(
            findings=all_findings, units=units, judged=judged, subject=subject
        )

        if not result.is_usable and self.on_quarantine is not None:
            self.on_quarantine(result)

        return result

    def audit_many(
        self, artifacts: Iterable[tuple[str, Any]]
    ) -> list[AuditResult]:
        return [self.audit(a, subject=name) for name, a in artifacts]

    # --- the human loop ---------------------------------------------------------

    def quarantine(self, limit: int = 50) -> list[AuditRecord]:
        """Unlabelled records that were flagged, worst first.

        Ordered by severity because reviewer attention is the scarce resource — the
        first ten things a human looks at should be the ten most likely to matter.
        """
        if self.log is None:
            return []
        rank = {"corrupt": 0, "suspect": 1, "clean": 2}
        rows = [r for r in self.log.unlabelled() if r.severity != "clean"]
        rows.sort(key=lambda r: (rank.get(r.severity, 3), -r.ts))
        return rows[:limit]

    def label(self, record_id: str, label: Label, note: str = "") -> None:
        """Record a decision. This is the entire training interface."""
        if self.log is None:
            raise RuntimeError("pipeline has no log; pass log_path to enable learning")
        rows = {r.record_id: r for r in self.log.all()}
        rec = rows.get(record_id)
        if rec is None:
            raise KeyError(f"no audit recorded with id {record_id!r}")
        rec.label = label
        rec.note = note
        rec.ts = time.time()
        self.log.append(rec)
        self._cache_stamp = -1  # force refit on next audit

    # --- reporting --------------------------------------------------------------

    def learning_report(self) -> str:
        if self.log is None:
            return "No log configured — running on defaults, learning disabled."

        rows = self.log.all()
        labelled = [r for r in rows if r.label is not None]
        keeps = sum(1 for r in labelled if r.label == "keep")

        lines = [
            f"audited   : {len(rows)}",
            f"labelled  : {len(labelled)}  (keep {keeps}, discard {len(labelled) - keeps})",
            "",
        ]

        for det in self.detectors:
            learned = self.fit(det)
            if learned is None:
                need = max(0, MIN_LABELS_TO_FIT - len(self.log.labelled(det.name)))
                lines.append(
                    f"{det.name}: using defaults — needs {need} more labels "
                    f"(and >={MIN_PER_CLASS} of each class) before fitting"
                )
            else:
                adopted = sum(1 for t in learned.values() if t.adopted)
                lines.append(f"{det.name}: {adopted}/{len(learned)} thresholds adopted")
                lines.extend(f"    {t}" for t in learned.values())
        return "\n".join(lines)
