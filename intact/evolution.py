"""
Evolution — the pipeline grows detectors it was never shipped with.

Threshold fitting tunes rules that already exist. This does something different: it
looks at what you rejected, finds signals that separate rejects from keeps, and
proposes rules for signals nobody programmed.

That matters because corruption is corpus-specific in ways no library author can
anticipate. A 1970s journal scan fails differently from a born-digital preprint,
which fails differently from a bank's CSV export. Ship a fixed detector set and it
will be wrong somewhere. Let it mine your own rejections and it converges on what
actually breaks in your data.

What it mines
-------------
Three families, chosen because each is cheap to compute, cheap to explain, and has
caught real problems:

  CHARACTER CLASS   Over-representation of a Unicode category in rejects. Catches
                    encoding damage — a corpus where rejects are full of Cyrillic
                    lookalikes, or Symbol-category glyphs from a broken font map.

  TOKEN SHAPE       Structural token patterns: all-caps runs, repeated punctuation,
                    single characters between spaces, tokens with mixed scripts.

  MARKER STRING     Literal substrings over-represented in rejects. Catches the
                    specific artifacts of one broken toolchain — "(cid:", "\\x00",
                    a header that leaks into every failed export.

Why proposals, not silent adoption
-----------------------------------
A system that invents its own rules and applies them quietly cannot be debugged, and
will happily learn that every reject happens to contain the word "the". Every
proposal here carries its discriminative power, its support count, and examples,
and waits for a human. That is not timidity — an unreviewable data-quality gate is
worse than none, because it produces confident silence.

The guard against spurious patterns is support, precision and effect size together.
Any one alone finds noise.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

# A pattern must appear in at least this many rejects before it is worth a look.
# Two rejects sharing a quirk is a coincidence; ten is a signal.
MIN_SUPPORT = 8

# Of the artifacts containing the pattern, at least this fraction must be rejects.
MIN_PRECISION = 0.85

# And it must be at least this many times more common in rejects than in keeps,
# so a pattern present in everything cannot qualify on precision alone.
MIN_LIFT = 4.0

# Retire a detector that has fired this many times without a single reject agreeing.
USELESS_AFTER = 30

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Proposal:
    """A candidate detection rule, with the evidence that produced it."""

    kind: Literal["character_class", "token_shape", "marker_string"]
    signature: str
    description: str
    support: int          # rejects containing it
    precision: float      # of artifacts containing it, fraction rejected
    lift: float           # how many times more common in rejects than keeps
    samples: tuple[str, ...] = ()

    @property
    def strength(self) -> float:
        """One number for ranking. Precision matters most; lift is capped.

        Uncapped lift lets a pattern appearing in three rejects and zero keeps
        dominate one appearing in eighty rejects and two keeps, which is backwards.
        """
        return self.precision * min(self.lift, 20.0) * (self.support ** 0.5)

    def __str__(self) -> str:
        head = (
            f"[{self.kind}] {self.signature}\n"
            f"    {self.description}\n"
            f"    {self.support} rejects, {self.precision:.0%} precision, "
            f"{self.lift:.1f}x lift"
        )
        if self.samples:
            shown = ", ".join(repr(s) for s in self.samples[:3])
            return f"{head}\n    examples: {shown}"
        return head


@dataclass
class DetectorUtility:
    """How useful a detector has actually been on this corpus."""

    name: str
    fired: int = 0
    agreed: int = 0        # fired AND you discarded
    overruled: int = 0     # fired AND you kept it anyway
    missed: int = 0        # stayed silent AND you discarded

    @property
    def precision(self) -> float:
        return self.agreed / self.fired if self.fired else 0.0

    @property
    def verdict(self) -> str:
        if self.fired == 0:
            return "never fired — consider removing, it costs time and finds nothing"
        if self.fired >= USELESS_AFTER and self.agreed == 0:
            return "fired often, never agreed with you — retire it"
        if self.precision < 0.3 and self.fired >= USELESS_AFTER:
            return f"low precision ({self.precision:.0%}) — raise its threshold"
        if self.missed > self.agreed:
            return "misses more than it catches — its threshold may be too lax"
        return f"useful ({self.precision:.0%} precision on {self.fired} firings)"

    def __str__(self) -> str:
        return f"{self.name}: {self.verdict}"


# --- feature extraction over raw artifacts --------------------------------------


def _char_classes(text: str) -> Counter[str]:
    """Unicode general category counts, normalised to a rate per 1000 chars."""
    if not text:
        return Counter()
    counts: Counter[str] = Counter()
    for ch in text:
        counts[unicodedata.category(ch)] += 1
    return Counter({k: (v / len(text)) * 1000 for k, v in counts.items()})


def _token_shapes(text: str) -> set[str]:
    """Structural shapes present in the text.

    Shapes, not tokens. "ALLCAPS_RUN" generalises across corpora; the literal token
    "SECTION" does not.
    """
    shapes: set[str] = set()
    tokens = _TOKEN.findall(text)
    if not tokens:
        return shapes

    caps_run = 0
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            shapes.add("ISOLATED_LETTER")
        if t.isupper() and len(t) > 2:
            caps_run += 1
        else:
            if caps_run >= 4:
                shapes.add("ALLCAPS_RUN")
            caps_run = 0
        if re.search(r"(.)\1{3,}", t):
            shapes.add("REPEATED_CHAR")
        if re.search(r"[.,;:]{2,}", t):
            shapes.add("REPEATED_PUNCT")
        scripts = {
            "LATIN" if "LATIN" in unicodedata.name(c, "") else
            "CYRILLIC" if "CYRILLIC" in unicodedata.name(c, "") else
            "GREEK" if "GREEK" in unicodedata.name(c, "") else "OTHER"
            for c in t if c.isalpha()
        }
        if len({s for s in scripts if s != "OTHER"}) > 1:
            shapes.add("MIXED_SCRIPT")
    if caps_run >= 4:
        shapes.add("ALLCAPS_RUN")
    return shapes


def _markers(text: str, max_len: int = 6) -> set[str]:
    """Short literal substrings that look like toolchain artifacts.

    Restricted to strings containing a non-alphanumeric character. Natural language
    n-grams are excluded on purpose: they are where spurious patterns live, and a
    corpus where every reject contains "the" teaches you nothing.
    """
    out: set[str] = set()
    for m in re.finditer(r"[^\w\s]{2,%d}" % max_len, text):
        out.add(m.group())
    for m in re.finditer(r"\(cid:\d+\)", text):
        out.add("(cid:")
    for ch in text:
        if unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn") and ch not in "\t\n\r":
            out.add(f"U+{ord(ch):04X}")
    return out


# --- mining ---------------------------------------------------------------------


def _mine(
    rejects: Sequence[set[str]],
    keeps: Sequence[set[str]],
    kind: Literal["token_shape", "marker_string"],
    describe: Callable[[str], str],
) -> list[Proposal]:
    """Find set-membership features over-represented in rejects."""
    reject_counts: Counter[str] = Counter()
    for s in rejects:
        reject_counts.update(s)
    keep_counts: Counter[str] = Counter()
    for s in keeps:
        keep_counts.update(s)

    n_r, n_k = max(1, len(rejects)), max(1, len(keeps))
    out: list[Proposal] = []

    for sig, r_count in reject_counts.items():
        if r_count < MIN_SUPPORT:
            continue
        k_count = keep_counts.get(sig, 0)
        precision = r_count / (r_count + k_count)
        if precision < MIN_PRECISION:
            continue
        r_rate, k_rate = r_count / n_r, k_count / n_k
        lift = r_rate / k_rate if k_rate else float("inf")
        if lift < MIN_LIFT:
            continue
        out.append(Proposal(
            kind=kind,
            signature=sig,
            description=describe(sig),
            support=r_count,
            precision=precision,
            lift=min(lift, 999.0),
        ))
    return out


def propose_rules(
    rejected: Iterable[str], kept: Iterable[str]
) -> list[Proposal]:
    """Mine new detection rules from labelled artifacts, strongest first.

    Give it the raw text of things you discarded and things you kept. It returns
    candidate rules with measured discriminative power. Nothing is applied; these
    are for a human to accept or throw out.
    """
    rejected, kept = list(rejected), list(kept)
    if len(rejected) < MIN_SUPPORT:
        return []

    proposals: list[Proposal] = []

    # character classes — compare mean rate per class
    r_classes = [_char_classes(t) for t in rejected]
    k_classes = [_char_classes(t) for t in kept]
    all_cats = {c for d in r_classes + k_classes for c in d}
    for cat in all_cats:
        r_vals = [d.get(cat, 0.0) for d in r_classes]
        k_vals = [d.get(cat, 0.0) for d in k_classes] or [0.0]
        r_mean = sum(r_vals) / len(r_vals)
        k_mean = sum(k_vals) / len(k_vals)
        if r_mean < 0.1:
            continue
        lift = r_mean / k_mean if k_mean else float("inf")
        if lift < MIN_LIFT:
            continue
        present = sum(1 for v in r_vals if v > 0)
        if present < MIN_SUPPORT:
            continue
        k_present = sum(1 for v in k_vals if v > 0)
        precision = present / (present + k_present)
        if precision < MIN_PRECISION:
            continue
        proposals.append(Proposal(
            kind="character_class",
            signature=cat,
            description=(
                f"Unicode category {cat} ({_category_name(cat)}) appears at "
                f"{r_mean:.2f}/1000 chars in rejects vs {k_mean:.2f} in keeps"
            ),
            support=present,
            precision=precision,
            lift=min(lift, 999.0),
        ))

    proposals += _mine(
        [_token_shapes(t) for t in rejected],
        [_token_shapes(t) for t in kept],
        "token_shape",
        lambda s: f"Token shape {s} is characteristic of your rejects",
    )
    proposals += _mine(
        [_markers(t) for t in rejected],
        [_markers(t) for t in kept],
        "marker_string",
        lambda s: f"Marker {s!r} appears almost exclusively in rejects",
    )

    proposals.sort(key=lambda p: -p.strength)
    return proposals


def _category_name(cat: str) -> str:
    return {
        "Cc": "control", "Cf": "format", "Co": "private use", "Cn": "unassigned",
        "Zs": "space", "So": "symbol", "Sk": "modifier symbol",
        "Lo": "other letter", "Mn": "non-spacing mark", "Nd": "digit",
    }.get(cat, cat)


def review_detectors(
    utilities: Iterable[DetectorUtility],
) -> tuple[list[str], list[str]]:
    """Split detectors into keep and retire.

    Retiring is as much a part of adapting to a corpus as adding. A detector that
    fires constantly and is always overruled trains people to ignore the report,
    which quietly disables every other detector too.
    """
    keep, retire = [], []
    for u in utilities:
        if u.fired == 0:
            retire.append(u.name)
        elif u.fired >= USELESS_AFTER and u.agreed == 0:
            retire.append(u.name)
        else:
            keep.append(u.name)
    return keep, retire


def evolution_report(
    proposals: Sequence[Proposal], utilities: Sequence[DetectorUtility]
) -> str:
    lines: list[str] = []

    if utilities:
        lines.append("Detector utility on this corpus")
        lines.extend(f"  {u}" for u in utilities)
        keep, retire = review_detectors(utilities)
        if retire:
            lines.append(f"  -> suggest retiring: {', '.join(retire)}")
        lines.append("")

    if not proposals:
        lines.append(
            f"No new rules proposed. Either the shipped detectors already cover your "
            f"failure modes, or there are fewer than {MIN_SUPPORT} rejects to learn "
            f"from."
        )
        return "\n".join(lines)

    lines.append(f"{len(proposals)} candidate rules mined from your rejections:")
    lines.append("")
    for p in proposals[:10]:
        lines.append(str(p))
        lines.append("")
    lines.append(
        "None of these are active. Each is a pattern that separated your rejects "
        "from your keeps — review before adopting, because a pattern is not a cause."
    )
    return "\n".join(lines)
