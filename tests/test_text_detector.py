"""
Tests for the extraction audit.

These are written to fail. During development the fragment detector returned "clean"
on deliberately shattered text — a detector that cannot detect is worse than none,
because it produces confident silence. Every case below exists because something
either did go wrong or plausibly could.

Run with:  python -m pytest tests/ -v
       or: python tests/test_extraction_audit.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intact.core import Severity, summarise  # noqa: E402
from intact.detectors.text import (  # noqa: E402
    MIN_TOKENS_FOR_STATS,
    Mode,
    audit_pages,
    audit_text,
)

# Academic-flavoured prose: prose, figures, years, decimals. Repeated to clear the
# minimum token count, since the detectors deliberately refuse to judge short text.
BASE = (
    "The selection of candidate materials was performed across 1247 samples "
    "collected between 1994 and 2003, yielding a mean response of 0.42 with a "
    "standard deviation of 0.07 across the full extraction procedure. "
)
CLEAN = BASE * 40


def _shatter(text: str, p: float, seed: int = 7) -> str:
    """Simulate character-positioning failure: words chopped into 2-4 char runs.

    Note the shape. An earlier version split each word exactly once, which leaves a
    long tail piece and does NOT reproduce the real failure — that bug hid a broken
    detector behind a passing test. Real shattering fragments a word throughout.
    """
    rng = random.Random(seed)
    out: list[str] = []
    for w in text.split():
        if len(w) > 4 and rng.random() < p:
            i = 0
            while i < len(w):
                n = rng.randint(2, 4)
                out.append(w[i:i + n])
                i += n
        else:
            out.append(w)
    return " ".join(out)


# --- clean text must not be flagged ---------------------------------------------


def test_clean_text_is_clean():
    r = audit_text(CLEAN)
    assert r.severity is Severity.CLEAN, r
    assert r.findings == []
    assert r.confidence == 1.0
    assert r.is_usable


def test_clean_text_survives_normal_punctuation_and_symbols():
    text = CLEAN + "See Fig. 3 (p < 0.001); cf. Smith et al., 2011 — n = 42. "
    assert audit_text(text).severity is Severity.CLEAN


# --- missing glyphs --------------------------------------------------------------


def test_missing_glyph_is_detected():
    """A font lacking 'c' turns every c into a replacement char."""
    r = audit_text(CLEAN.replace("c", "�"))
    assert r.severity is Severity.CORRUPT, r
    assert any(f.mode == Mode.MISSING_GLYPH.value for f in r.findings)
    assert not r.is_usable


def test_unmapped_cid_markers_are_detected():
    """pdfminer emits (cid:NNN) when a glyph has no ToUnicode entry."""
    text = CLEAN.replace("the", "(cid:87)")
    r = audit_text(text)
    assert any(f.mode == Mode.MISSING_GLYPH.value for f in r.findings), r


def test_glyph_evidence_names_the_character():
    r = audit_text(CLEAN.replace("e", "�"))
    finding = next(f for f in r.findings if f.mode == Mode.MISSING_GLYPH)
    assert finding.samples, "a finding with no evidence is not actionable"
    assert "U+FFFD" in finding.samples


# --- shattered words -------------------------------------------------------------


def test_heavy_shattering_is_corrupt():
    r = audit_text(_shatter(CLEAN, 0.6))
    assert r.severity is Severity.CORRUPT, r
    assert any(f.mode == Mode.SHATTERED_WORDS.value for f in r.findings)


def test_moderate_shattering_is_corrupt():
    r = audit_text(_shatter(CLEAN, 0.3))
    assert r.severity is Severity.CORRUPT, r


def test_light_shattering_is_at_least_suspect():
    """Regression: this returned CLEAN and it should not have.

    One split per long word is visibly damaged text — "The sel ection of ca ndidate"
    — and a reader would reject it. The threshold was lowered specifically for this.
    """
    rng = random.Random(7)
    out = []
    for w in CLEAN.split():
        if len(w) > 5 and rng.random() < 0.6:
            i = rng.randint(2, len(w) - 2)
            out.extend([w[:i], w[i:]])
        else:
            out.append(w)
    r = audit_text(" ".join(out))
    assert r.severity is not Severity.CLEAN, r


def test_shatter_evidence_shows_runs_not_single_tokens():
    """The run is the evidence; an isolated short word proves nothing."""
    r = audit_text(_shatter(CLEAN, 0.6))
    finding = next(f for f in r.findings if f.mode == Mode.SHATTERED_WORDS)
    assert finding.samples
    assert any(" " in s for s in finding.samples), finding.samples


# --- numeric loss ----------------------------------------------------------------


def test_digits_as_control_bytes_are_detected():
    """The dangerous one: prose reads fine, every number is gone."""
    text = "".join("\x01" if ch.isdigit() else ch for ch in CLEAN)
    r = audit_text(text)
    assert any(f.mode == Mode.NUMERIC_LOSS.value for f in r.findings), r
    assert r.severity is Severity.CORRUPT


def test_numeric_loss_flagged_even_when_prose_is_intact():
    """Digits stripped entirely — no control bytes, so only the digit test can catch it."""
    text = "".join(" " if ch.isdigit() else ch for ch in CLEAN)
    r = audit_text(text)
    modes = {f.mode for f in r.findings}
    assert Mode.NUMERIC_LOSS.value in modes, r
    assert Mode.MISSING_GLYPH.value not in modes, "no undecodable chars were introduced"


# --- refusing to judge -----------------------------------------------------------


def test_short_text_is_not_judged_on_statistics():
    """Below the minimum, ratios are noise. Silence beats a confident guess."""
    r = audit_text("Short fragment of text.")
    assert r.units < MIN_TOKENS_FOR_STATS
    assert not any(
        f.mode in (Mode.SHATTERED_WORDS.value, Mode.NUMERIC_LOSS.value) for f in r.findings
    )


def test_empty_text_does_not_crash():
    r = audit_text("")
    assert r.units == 0
    assert r.severity is Severity.CLEAN


# --- document-level rollup -------------------------------------------------------


def test_summary_takes_worst_page_not_average():
    """One corrupt page in fifty still makes the document unusable for that page.

    Averaging severity would hide it, which is exactly the silent failure this whole
    module exists to prevent.
    """
    broken = "".join("\x01" if ch.isdigit() else ch for ch in CLEAN)
    pages = [CLEAN] * 49 + [broken]
    s = summarise(audit_pages(pages))
    assert s["severity"] == Severity.CORRUPT.value
    assert s["flagged"] == ["page 49"]
    assert s["artifacts"] == 50


def test_summary_of_no_pages_is_safe():
    s = summarise([])
    assert s["artifacts"] == 0
    assert s["severity"] == Severity.CLEAN.value


def test_summary_counts_each_mode():
    glyph = CLEAN.replace("c", "�")
    shattered = _shatter(CLEAN, 0.6)
    s = summarise([audit_text(CLEAN), audit_text(glyph), audit_text(shattered)])
    assert s["modes"].get("missing_glyph", 0) >= 1
    assert s["modes"].get("shattered_words", 0) >= 1


# --- confidence ------------------------------------------------------------------


def test_confidence_falls_with_severity():
    clean = audit_text(CLEAN).confidence
    light = audit_text(_shatter(CLEAN, 0.15)).confidence
    heavy = audit_text(_shatter(CLEAN, 0.6)).confidence
    assert clean > light >= heavy
    assert 0.0 <= heavy <= 1.0


if __name__ == "__main__":
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
        except Exception:
            failed += 1
            print(f"  ERROR {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
