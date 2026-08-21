"""
Extraction audit — detect silently corrupted PDF text extraction.

Most PDF extraction stacks return a string and no indication of whether that string
is trustworthy. Docling, for example, currently exposes no confidence score
(github.com/docling-project/docling/discussions/2814). The failure is silent: you get
text back, it looks like text, and some fraction of it is wrong.

This module scores extracted text and names the specific failure mode, so a pipeline
can quarantine a document instead of ingesting corruption.

Three modes, all observed in the wild on a real corpus:

  MISSING_GLYPH    An embedded font lacks a character, or its ToUnicode CMap omits it.
                   "selection" extracts as "sele tion". The text looks almost right,
                   which is what makes it dangerous.

  SHATTERED_WORDS  Spaces injected mid-word from character-level positioning.
                   "extraction" becomes "ext rac tion".

  NUMERIC_LOSS     Digits stored as control bytes or unmapped CIDs. No number in the
                   document can be recovered. Fatal for any quantitative use, and
                   invisible if you only eyeball the prose.

Design notes
------------
Detectors are independent and each returns evidence, never a bare boolean. A caller
deciding to discard a 300-page document deserves to see why.

Thresholds are module-level constants rather than magic numbers so they can be tuned
per corpus and, more importantly, so they are visible.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class Severity(str, Enum):
    CLEAN = "clean"
    SUSPECT = "suspect"
    CORRUPT = "corrupt"


class Mode(str, Enum):
    MISSING_GLYPH = "missing_glyph"
    SHATTERED_WORDS = "shattered_words"
    NUMERIC_LOSS = "numeric_loss"


# --- tuning constants, deliberately visible -------------------------------------

# Fraction of words caught in fragment runs before we call it shattered.
# Clean prose measures ~0.00-0.02; light single-split damage ~0.05; genuine
# character-positioning failure 0.25+. SUSPECT is set just above the clean band
# rather than midway, because a missed corruption is far more costly than a
# second look at a clean document.
#
# These two numbers are the least defensible thing in this module — they come from
# one corpus. They are exactly what `learning.fit_thresholds` refits from your
# labels once you have enough of them.
SHATTERED_SUSPECT = 0.05
SHATTERED_CORRUPT = 0.25

# A document of substance that yields almost no digits is suspicious. Below this
# ratio of digit-bearing tokens we flag; papers cite years, sample sizes, p-values.
NUMERIC_SUSPECT = 0.004
NUMERIC_CORRUPT = 0.0005

# Replacement chars, unmapped CIDs and control bytes per 1000 chars.
GLYPH_SUSPECT = 0.5
GLYPH_CORRUPT = 3.0

# Below this many tokens the statistics are not meaningful.
MIN_TOKENS_FOR_STATS = 200

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TOKEN = re.compile(r"\S+")
_CID = re.compile(r"\(cid:\d+\)")
_DIGIT = re.compile(r"\d")

# Characters a well-formed extraction should not contain. U+FFFD is the explicit
# "I could not decode this" marker; the C0/C1 ranges are control bytes that mean a
# CMap lookup fell through.
_REPLACEMENT = "�"


@dataclass
class Finding:
    """One detected problem, with the evidence that produced it."""

    mode: Mode
    severity: Severity
    metric: float
    threshold: float
    detail: str
    samples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"[{self.severity.value.upper()}] {self.mode.value}: {self.detail}"
        if not self.samples:
            return head
        shown = ", ".join(repr(s) for s in self.samples[:5])
        return f"{head}\n    examples: {shown}"


@dataclass
class AuditResult:
    """Verdict for one extracted document."""

    severity: Severity
    findings: list[Finding]
    token_count: int
    confidence: float  # 0.0 unusable .. 1.0 clean

    @property
    def is_usable(self) -> bool:
        return self.severity is not Severity.CORRUPT

    def reasons(self) -> list[str]:
        return [str(f) for f in self.findings]

    def __str__(self) -> str:
        if not self.findings:
            return f"clean ({self.token_count} tokens, confidence {self.confidence:.2f})"
        lines = [
            f"{self.severity.value} — confidence {self.confidence:.2f}, "
            f"{self.token_count} tokens"
        ]
        lines.extend(f"  {f}" for f in self.findings)
        return "\n".join(lines)


# --- detectors ------------------------------------------------------------------


def _control_char_ratio(text: str) -> tuple[float, list[str]]:
    """Replacement chars, unmapped CIDs and stray control bytes per 1000 chars.

    Excludes tab/newline/carriage-return, which are legitimate in extracted text.
    """
    if not text:
        return 0.0, []

    samples: list[str] = []
    bad = 0

    for m in _CID.finditer(text):
        bad += 1
        if len(samples) < 8:
            samples.append(m.group())

    for ch in text:
        if ch == _REPLACEMENT:
            bad += 1
            if _REPLACEMENT not in samples and len(samples) < 8:
                samples.append("U+FFFD")
        elif ch in "\t\n\r":
            continue
        elif unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn"):
            bad += 1
            label = f"U+{ord(ch):04X}"
            if label not in samples and len(samples) < 8:
                samples.append(label)

    return (bad / len(text)) * 1000, samples


# Short words that legitimately appear in English prose at high frequency. Anything
# short and NOT in here is a fragment candidate. Kept as a stoplist rather than a
# dictionary so the module stays dependency-free.
_LEGIT_SHORT = frozenset("""
a i an as at be by do go he if in is it me my no of on or so to up us we am an
the and for was are but not you all can had her his its our out she who did has
this that with from they have been were will more when what your than them then
et al ie eg vs pp ed fig ref eq no cf ca vol
""".split())

# A run of this many consecutive short non-stopword tokens is the signature of
# shattering. Real English essentially never does this; "ext rac tion of the sam ple"
# does it constantly. Three is chosen because two occurs naturally (proper nouns,
# initials, units) and four would miss lightly-shattered text.
_RUN_LENGTH = 3
_SHORT = 4  # tokens of this length or less are fragment candidates


def _fragment_ratio(text: str) -> tuple[float, list[str]]:
    """Fraction of words caught in runs of consecutive short non-stopword tokens.

    A naive "count the short words" test does not work: shattering produces mostly
    3-4 character pieces, and those collide with real English words (the, for, with).
    Length alone cannot separate them.

    Consecutive runs can. English alternates short function words with longer content
    words, so three short non-stopwords in a row is rare in real prose and pervasive
    in shattered text. This measures the fraction of all words that sit inside such a
    run, which is both more sensitive and far less prone to false positives than a
    flat length threshold.
    """
    words = _WORD.findall(text)
    if len(words) < MIN_TOKENS_FOR_STATS:
        return 0.0, []

    is_frag = [
        len(w) <= _SHORT and w.lower() not in _LEGIT_SHORT for w in words
    ]

    in_run = [False] * len(words)
    run_start = 0
    for i in range(len(words) + 1):
        if i < len(words) and is_frag[i]:
            continue
        if i - run_start >= _RUN_LENGTH:
            for j in range(run_start, i):
                in_run[j] = True
        run_start = i + 1

    caught = sum(in_run)
    ratio = caught / len(words)

    # Report the actual runs, not individual tokens — the run is the evidence.
    samples: list[str] = []
    i = 0
    while i < len(words) and len(samples) < 6:
        if in_run[i]:
            j = i
            while j < len(words) and in_run[j]:
                j += 1
            samples.append(" ".join(words[i:j]))
            i = j
        else:
            i += 1

    return ratio, samples


def _digit_token_ratio(text: str) -> tuple[float, int]:
    """Fraction of whitespace tokens containing at least one digit.

    Academic prose is dense with years, counts, percentages and statistics. A long
    document with essentially none has almost certainly lost its digits to an
    unmapped font, and no figure in it can be trusted.
    """
    tokens = _TOKEN.findall(text)
    if len(tokens) < MIN_TOKENS_FOR_STATS:
        return -1.0, len(tokens)
    with_digits = sum(1 for t in tokens if _DIGIT.search(t))
    return with_digits / len(tokens), len(tokens)


# --- public API -----------------------------------------------------------------


def audit_text(text: str) -> AuditResult:
    """Score one extracted document and name any corruption modes present.

    Returns an AuditResult carrying every finding and its evidence. Callers should
    branch on `.is_usable` or inspect `.severity` directly; nothing is discarded
    here, because the decision belongs to the pipeline, not the detector.
    """
    findings: list[Finding] = []
    token_count = len(_TOKEN.findall(text))

    # --- missing glyphs / control bytes
    glyph_rate, glyph_samples = _control_char_ratio(text)
    if glyph_rate >= GLYPH_CORRUPT:
        findings.append(Finding(
            Mode.MISSING_GLYPH, Severity.CORRUPT, glyph_rate, GLYPH_CORRUPT,
            f"{glyph_rate:.2f} undecodable chars per 1000 — the font's character map "
            f"is incomplete and the text is unreliable throughout",
            glyph_samples,
        ))
    elif glyph_rate >= GLYPH_SUSPECT:
        findings.append(Finding(
            Mode.MISSING_GLYPH, Severity.SUSPECT, glyph_rate, GLYPH_SUSPECT,
            f"{glyph_rate:.2f} undecodable chars per 1000 — localised glyph loss",
            glyph_samples,
        ))

    # --- shattered words
    frag_rate, frag_samples = _fragment_ratio(text)
    if frag_rate >= SHATTERED_CORRUPT:
        findings.append(Finding(
            Mode.SHATTERED_WORDS, Severity.CORRUPT, frag_rate, SHATTERED_CORRUPT,
            f"{frag_rate:.1%} of words are fragments — spacing was reconstructed "
            f"from character positions and failed",
            frag_samples,
        ))
    elif frag_rate >= SHATTERED_SUSPECT:
        findings.append(Finding(
            Mode.SHATTERED_WORDS, Severity.SUSPECT, frag_rate, SHATTERED_SUSPECT,
            f"{frag_rate:.1%} of words are fragments — word boundaries are unreliable",
            frag_samples,
        ))

    # --- numeric loss
    digit_rate, _ = _digit_token_ratio(text)
    if digit_rate >= 0:  # -1 means too short to judge
        if digit_rate <= NUMERIC_CORRUPT:
            findings.append(Finding(
                Mode.NUMERIC_LOSS, Severity.CORRUPT, digit_rate, NUMERIC_CORRUPT,
                f"only {digit_rate:.3%} of tokens contain a digit — digits are stored "
                f"as unmapped bytes. NO NUMBER FROM THIS DOCUMENT CAN BE VERIFIED",
            ))
        elif digit_rate <= NUMERIC_SUSPECT:
            findings.append(Finding(
                Mode.NUMERIC_LOSS, Severity.SUSPECT, digit_rate, NUMERIC_SUSPECT,
                f"only {digit_rate:.3%} of tokens contain a digit — unusually few for "
                f"academic text; check figures before relying on them",
            ))

    if any(f.severity is Severity.CORRUPT for f in findings):
        severity = Severity.CORRUPT
    elif findings:
        severity = Severity.SUSPECT
    else:
        severity = Severity.CLEAN

    # Confidence falls with each finding, weighted by how far past threshold it is.
    confidence = 1.0
    for f in findings:
        overshoot = 1.0
        if f.threshold > 0:
            overshoot = min(3.0, max(1.0, f.metric / f.threshold))
        penalty = (0.45 if f.severity is Severity.CORRUPT else 0.15) * overshoot
        confidence -= penalty
    confidence = max(0.0, min(1.0, confidence))

    return AuditResult(severity, findings, token_count, confidence)


def audit_pages(pages: Sequence[str]) -> list[AuditResult]:
    """Audit each page separately.

    Corruption is frequently confined to particular pages — a figure-heavy section
    with a different embedded font, or scanned pages inside a digital document.
    Auditing the whole text at once averages that away and hides it.
    """
    return [audit_text(p) for p in pages]


def summarise(results: Iterable[AuditResult]) -> dict[str, object]:
    """Roll page-level results into one document verdict.

    A document is only as trustworthy as its worst substantive page, so severity is
    the maximum rather than the mean. Pages too short to judge are excluded from the
    confidence average so a title page cannot flatter the result.
    """
    results = list(results)
    if not results:
        return {"severity": Severity.CLEAN.value, "pages": 0, "confidence": 1.0}

    order = {Severity.CLEAN: 0, Severity.SUSPECT: 1, Severity.CORRUPT: 2}
    worst = max(results, key=lambda r: order[r.severity]).severity

    judged = [r for r in results if r.token_count >= MIN_TOKENS_FOR_STATS] or results
    mean_conf = sum(r.confidence for r in judged) / len(judged)

    mode_counts: Counter[str] = Counter()
    for r in results:
        for f in r.findings:
            mode_counts[f.mode.value] += 1

    flagged = [i for i, r in enumerate(results) if r.findings]

    return {
        "severity": worst.value,
        "pages": len(results),
        "pages_judged": len(judged),
        "confidence": round(mean_conf, 3),
        "modes": dict(mode_counts),
        "flagged_pages": flagged,
    }
