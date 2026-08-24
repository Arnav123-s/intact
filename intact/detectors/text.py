"""
Text detector — silently corrupted text extraction, usually from PDFs.

Most extraction stacks hand back a string and no way to tell whether it is
trustworthy. Docling, for one, exposes no confidence score for PDF results
(github.com/docling-project/docling/discussions/2814). The failure is silent. You get
text back, it looks like text, and some of it is wrong.

Three modes, all of which I hit on a real corpus:

  MISSING_GLYPH    An embedded font lacks a character, or its ToUnicode CMap omits
                   it. "selection" extracts as "sele tion". The text looks almost
                   right, which is what makes it dangerous.

  SHATTERED_WORDS  Spaces injected mid-word from character-level positioning.
                   "extraction" becomes "ext rac tion".

  NUMERIC_LOSS     Digits stored as control bytes or unmapped CIDs. No number in the
                   document can be recovered. Fatal for anything quantitative, and
                   invisible if you only read the prose.

Findings use the shared types in `core`, so a result from here can be compared and
summarised next to one from the tabular detector. This module used to define its own
Severity and Finding, which meant it could not. That bug only showed up once both
detectors ran over the same folder.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from enum import Enum
from typing import Iterable, Sequence

from ..core import AuditResult, Finding, Severity


class Mode(str, Enum):
    """This detector's own failure modes.

    `core.Finding.mode` is a plain string on purpose, so adding a detector means
    adding a file and nothing else. This enum is local vocabulary, not a registry.
    """

    MISSING_GLYPH = "missing_glyph"
    SHATTERED_WORDS = "shattered_words"
    NUMERIC_LOSS = "numeric_loss"


# --- tuning constants, deliberately visible -------------------------------------
#
# These are the least defensible numbers in the module: they come from one corpus.
# They are exactly what `pipeline.Pipeline.fit` refits from your own labels.

# Fraction of words caught in fragment runs before text is called shattered.
# Clean prose measures about 0.00-0.02. Light single-split damage is about 0.05.
# Real character-positioning failure is 0.25 and up. SUSPECT sits just above the
# clean band rather than midway, because missing a corruption costs more than taking
# a second look at a clean document.
SHATTERED_SUSPECT = 0.05
SHATTERED_CORRUPT = 0.25

# A document of substance yielding almost no digits is suspicious: papers cite years,
# sample sizes and p-values constantly.
NUMERIC_SUSPECT = 0.004
NUMERIC_CORRUPT = 0.0005

# Replacement characters, unmapped CIDs and stray control bytes per 1000 characters.
GLYPH_SUSPECT = 0.5
GLYPH_CORRUPT = 3.0

# Below this many tokens the statistics are noise, and nothing is judged.
MIN_TOKENS_FOR_STATS = 200

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TOKEN = re.compile(r"\S+")
_CID = re.compile(r"\(cid:\d+\)")
_DIGIT = re.compile(r"\d")

# U+FFFD is the explicit "could not decode this" marker.
_REPLACEMENT = "�"

# Short words that legitimately appear in English prose at high frequency. Anything
# short and NOT here is a fragment candidate. A stoplist rather than a dictionary, so
# the module stays dependency-free.
_LEGIT_SHORT = frozenset("""
a i an as at be by do go he if in is it me my no of on or so to up us we am
the and for was are but not you all can had her his its our out she who did has
this that with from they have been were will more when what your than them then
et al ie eg vs pp ed fig ref eq cf ca vol
""".split())

# A run of this many consecutive short non-stopword tokens is the signature of
# shattering. Real English essentially never does this; "ext rac tion of the sam ple"
# does it constantly. Three, because two occurs naturally (initials, units) and four
# would miss lightly-shattered text.
_RUN_LENGTH = 3
_SHORT = 4  # tokens this length or shorter are fragment candidates


# --- measurement ------------------------------------------------------------------


def _control_char_ratio(text: str) -> tuple[float, list[str]]:
    """Undecodable characters per 1000, with examples naming the codepoint.

    Tab, newline and carriage return are excluded. They are legitimate in extracted
    text and would otherwise dominate the count.
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
            if "U+FFFD" not in samples and len(samples) < 8:
                samples.append("U+FFFD")
        elif ch in "\t\n\r":
            continue
        elif unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn"):
            bad += 1
            label = f"U+{ord(ch):04X}"
            if label not in samples and len(samples) < 8:
                samples.append(label)

    return (bad / len(text)) * 1000, samples


def _fragment_ratio(text: str) -> tuple[float, list[str]]:
    """Fraction of words sitting inside runs of consecutive short non-stopwords.

    Counting short words alone does not work. Shattering produces mostly 3-4
    character pieces, which collide with real English words like the, for and with.
    Length cannot separate them.

    Runs can. English alternates short function words with longer content words, so
    three short non-stopwords in a row is rare in real prose and everywhere in
    shattered text. Evidence is reported as whole runs, because the run is the
    evidence. An isolated short word proves nothing.
    """
    words = _WORD.findall(text)
    if len(words) < MIN_TOKENS_FOR_STATS:
        return 0.0, []

    is_frag = [len(w) <= _SHORT and w.lower() not in _LEGIT_SHORT for w in words]

    in_run = [False] * len(words)
    run_start = 0
    for i in range(len(words) + 1):
        if i < len(words) and is_frag[i]:
            continue
        if i - run_start >= _RUN_LENGTH:
            for j in range(run_start, i):
                in_run[j] = True
        run_start = i + 1

    ratio = sum(in_run) / len(words)

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

    Returns -1.0 when the text is too short to judge, so the caller can tell "no
    digits" apart from "not enough text to say".
    """
    tokens = _TOKEN.findall(text)
    if len(tokens) < MIN_TOKENS_FOR_STATS:
        return -1.0, len(tokens)
    return sum(1 for t in tokens if _DIGIT.search(t)) / len(tokens), len(tokens)


# --- public API -------------------------------------------------------------------


def audit_text(text: str, subject: str = "") -> AuditResult:
    """Score one extracted document and name any corruption in it.

    Nothing is discarded here. The decision to quarantine belongs to the pipeline,
    not the detector. Every finding carries its measurement, its threshold and real
    examples, so you can disagree with a threshold without losing the number.
    """
    findings: list[Finding] = []
    tokens = len(_TOKEN.findall(text))
    judged = tokens >= MIN_TOKENS_FOR_STATS

    # --- missing glyphs / control bytes
    glyph_rate, glyph_samples = _control_char_ratio(text)
    if glyph_rate >= GLYPH_SUSPECT:
        corrupt = glyph_rate >= GLYPH_CORRUPT
        findings.append(Finding(
            mode=Mode.MISSING_GLYPH.value,
            severity=Severity.CORRUPT if corrupt else Severity.SUSPECT,
            metric=glyph_rate,
            threshold=GLYPH_CORRUPT if corrupt else GLYPH_SUSPECT,
            detail=(
                f"{glyph_rate:.2f} undecodable characters per 1000. The font's "
                f"character map is incomplete and the text is unreliable throughout"
                if corrupt else
                f"{glyph_rate:.2f} undecodable characters per 1000, so glyph loss is "
                f"localised"
            ),
            samples=tuple(glyph_samples),
            location=subject,
        ))

    # --- shattered words
    frag_rate, frag_samples = _fragment_ratio(text)
    if frag_rate >= SHATTERED_SUSPECT:
        corrupt = frag_rate >= SHATTERED_CORRUPT
        findings.append(Finding(
            mode=Mode.SHATTERED_WORDS.value,
            severity=Severity.CORRUPT if corrupt else Severity.SUSPECT,
            metric=frag_rate,
            threshold=SHATTERED_CORRUPT if corrupt else SHATTERED_SUSPECT,
            detail=(
                f"{frag_rate:.1%} of words are fragments. Spacing was reconstructed "
                f"from character positions and failed"
                if corrupt else
                f"{frag_rate:.1%} of words are fragments, so word boundaries are "
                f"unreliable"
            ),
            samples=tuple(frag_samples),
            location=subject,
        ))

    # --- numeric loss
    digit_rate, _ = _digit_token_ratio(text)
    if 0 <= digit_rate <= NUMERIC_SUSPECT:
        corrupt = digit_rate <= NUMERIC_CORRUPT
        findings.append(Finding(
            mode=Mode.NUMERIC_LOSS.value,
            severity=Severity.CORRUPT if corrupt else Severity.SUSPECT,
            metric=digit_rate,
            threshold=NUMERIC_CORRUPT if corrupt else NUMERIC_SUSPECT,
            detail=(
                f"only {digit_rate:.3%} of tokens contain a digit. Digits are "
                f"stored as unmapped bytes. NO NUMBER FROM THIS DOCUMENT CAN BE "
                f"VERIFIED"
                if corrupt else
                f"only {digit_rate:.3%} of tokens contain a digit, which is unusually "
                f"few for academic text. Check figures before relying on them"
            ),
            location=subject,
        ))

    return AuditResult(
        findings=findings, units=tokens, judged=judged, subject=subject
    )


def audit_pages(pages: Sequence[str]) -> list[AuditResult]:
    """Audit each page separately.

    Corruption often sits on particular pages: a figure-heavy section with a
    different embedded font, or scanned pages inside a digital document. Auditing the
    whole text at once averages that away and hides it.
    """
    return [audit_text(p, subject=f"page {i}") for i, p in enumerate(pages)]
