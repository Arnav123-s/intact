"""
Sources — point it at anything and it works out the rest.

Everything else in this package takes rows or text. Getting rows or text out of "a
thing someone has" turns out to be a lot of tedium. Is it a file, a folder, a URL, a
pattern, a zip? Is it CSV or TSV or JSON-lines or an Excel export someone renamed to
.csv? Which of those thirty files in the folder are data and which are readmes?

This module answers all of that, so you can write:

    for source in resolve("data/"):        # a folder
    for source in resolve("*.csv"):        # a pattern
    for source in resolve("https://...")   # a URL
    for source in resolve(["a.csv", "b/"]) # several of the above

and get back one uniform stream of named, typed, already-parsed sources.

Format detection: content over extension
-----------------------------------------
Extensions lie constantly. Excel exports get renamed to `.csv`, JSON-lines files get
called `.json`, and `.txt` holds anything at all. So the extension is a hint and the
content is the evidence. The sniffer reads the first few kilobytes and decides from
what is actually there. When the two disagree it says so instead of quietly picking.

Zero dependencies, including Excel
-----------------------------------
An `.xlsx` is a zip archive of XML. `zipfile` and `xml.etree` are both standard
library, so xlsx gets read here without pandas or openpyxl. That is more code than
importing a library, and it keeps the promise that this installs anywhere with
nothing else.

Parquet is the deliberate exception. There is no reasonable stdlib path, so it gets
detected, named and skipped with a clear message instead of half-supported.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field

from .core import plural
from pathlib import Path
from typing import Any, Iterator, Sequence
from xml.etree import ElementTree

# Read this much to decide what a thing is. Enough to be sure, small enough to be
# free even when the source is a 40 GB file or a slow URL.
SNIFF_BYTES = 64 * 1024

# Files that are never data, whatever they are called. Stops a folder scan from
# reporting that your README is not valid CSV.
_SKIP_NAMES = {
    "readme", "license", "licence", "changelog", "contributing", "authors",
    "notice", "makefile", "dockerfile", ".gitignore", ".gitattributes",
}
_SKIP_SUFFIXES = {
    ".md", ".rst", ".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".go",
    ".rs", ".sh", ".bat", ".ps1", ".yml", ".yaml", ".toml", ".ini", ".cfg",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".zip", ".gz", ".tar",
    ".exe", ".dll", ".so", ".dylib", ".lock", ".log",
}

_URL = re.compile(r"^https?://", re.I)
_GLOB_CHARS = re.compile(r"[*?\[]")


class SourceError(Exception):
    """Raised when a source is named but cannot be read at all."""


@dataclass
class Source:
    """One resolved, parsed thing, ready for a detector.

    `rows` is set for tabular formats and `text` for document formats. Exactly one of
    them is populated, so you can branch on which is None without looking at `kind`.
    `kind` is there for when the distinction matters.
    """

    name: str
    kind: str                      # csv | tsv | json | jsonl | xlsx | text | unknown
    rows: list[list[str]] | None = None
    text: str | None = None
    encoding: str = ""
    origin: str = ""               # file path, URL, or "sheet 2 of book.xlsx"
    notes: list[str] = field(default_factory=list)

    @property
    def is_tabular(self) -> bool:
        return self.rows is not None

    def __str__(self) -> str:
        what = (
            f"{len(self.rows):,} rows" if self.rows is not None
            else f"{len(self.text or ''):,} chars"
        )
        note = f"  [{'; '.join(self.notes)}]" if self.notes else ""
        return f"{self.name} ({self.kind}, {what}){note}"


# --- getting bytes ---------------------------------------------------------------


def _fetch_url(url: str, timeout: int = 60) -> bytes:
    """Read a URL. No retries, no auth, no redirects beyond urllib's default.

    Deliberately thin. If you need headers, tokens or pagination, fetch the bytes
    yourself and hand them to `from_bytes`. This is a convenience, not an HTTP
    client, and pretending otherwise invites a pile of half-features.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "intact/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise SourceError(f"{url} returned HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise SourceError(f"{url} could not be reached: {e.reason}") from e


def _looks_like_data(path: Path) -> bool:
    """Filter obvious non-data before trying to parse it."""
    if path.name.lower() in _SKIP_NAMES:
        return False
    if path.stem.lower() in _SKIP_NAMES:
        return False
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return False
    return path.is_file() and path.stat().st_size > 0


# --- deciding what something is ---------------------------------------------------


def sniff_kind(head: bytes, hint: str = "") -> tuple[str, list[str]]:
    """Decide a format from content, using the extension only as a tiebreak.

    Returns the kind plus any notes worth surfacing, especially when the content
    disagrees with the extension. That happens often enough to be worth saying out
    loud instead of quietly overriding.
    """
    notes: list[str] = []
    ext = hint.lower().lstrip(".")

    # xlsx and every other OOXML file is a zip starting with "PK".
    if head[:2] == b"PK":
        if ext in ("xlsx", "xlsm"):
            return "xlsx", notes
        return "xlsx", ["file is a zip archive; treating as xlsx"]

    # Parquet is magic-numbered at both ends.
    if head[:4] == b"PAR1":
        return "parquet", notes

    if head[:5] == b"%PDF-":
        return "pdf", notes

    try:
        sample = head.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - decode with replace does not raise
        return "unknown", ["could not decode"]

    stripped = sample.lstrip()

    # JSON vs JSON-lines: both start with a brace, but JSON-lines has one complete
    # object per line, so line 2 also parses.
    if stripped[:1] in "{[":
        lines = [l for l in stripped.splitlines() if l.strip()][:3]
        if len(lines) >= 2:
            try:
                json.loads(lines[0])
                json.loads(lines[1])
                return "jsonl", notes
            except json.JSONDecodeError:
                pass
        return "json", notes

    # Delimited text: pick whichever candidate makes the rows rectangular.
    #
    # Counting delimiter characters per line does NOT work, and this module had that
    # bug. A column holding "1,234" contains a quoted comma, so raw comma counts vary
    # line to line, the sniffer decides there is no delimiter, and the file falls
    # through to "text" and never reaches the tabular detectors.
    #
    # So each candidate gets parsed with a real CSV reader, which honours quoting,
    # and scored by how many rows come out at the modal width. The right delimiter is
    # the one that makes the table rectangular.
    lines_ = [l for l in sample.splitlines() if l.strip()][:50]
    if len(lines_) >= 2:
        # Drop a trailing partial line. The sniff buffer usually cuts mid-row.
        keep = lines_[:-1] if len(lines_) > 2 else lines_
        body = "\n".join(keep)
        best, best_score = "", 0.0
        for delim in (",", "\t", ";", "|"):
            try:
                parsed = [r for r in csv.reader(io.StringIO(body), delimiter=delim) if r]
            except csv.Error:
                continue
            if not parsed:
                continue
            # Need enough rows for "rectangular" to mean anything. Two lines of
            # prose where one happens to contain a comma would otherwise score as a
            # two-column table. That is how this first misclassified plain text.
            if len(parsed) < 3:
                continue
            widths = Counter(len(r) for r in parsed)
            modal, n = widths.most_common(1)[0]
            if modal < 2:
                continue
            # And the modal width has to dominate. Prose produces a scatter of
            # widths. A table produces one width for nearly every row.
            agreement = n / len(parsed)
            if agreement < 0.8:
                continue
            score = agreement * min(modal, 40)
            if score > best_score:
                best, best_score = delim, score
        if best:
            if ext in ("txt", "dat", "") and best == ",":
                notes.append(
                    f"extension .{ext or '(none)'} but content is comma-delimited"
                )
            return ("tsv" if best == "\t" else "csv"), notes

    if ext in ("csv", "tsv"):
        notes.append(f"extension says .{ext} but no consistent delimiter was found")
        return "text", notes

    return "text", notes


# --- readers ----------------------------------------------------------------------


def _read_delimited(raw: bytes, delimiter: str, name: str) -> Source:
    from .solve import detect_encoding  # local import; avoids a cycle

    encoding, _ = detect_encoding(raw[:SNIFF_BYTES])
    text = raw.decode(encoding, errors="replace")
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    return Source(
        name=name, kind="tsv" if delimiter == "\t" else "csv",
        rows=rows, encoding=encoding,
    )


def _read_json(raw: bytes, name: str, lines: bool) -> Source:
    """Flatten JSON records to rows so the tabular detectors can see them.

    Only the top level gets flattened. Nested objects are serialised back to JSON in
    their own cell rather than exploded into columns. Exploding changes the shape of
    the data, and this module's job is to read it, not reinterpret it.
    """
    text = raw.decode("utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    notes: list[str] = []

    if lines:
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                notes.append(f"line {i} is not valid JSON ({e.msg}); skipped")
                continue
            if isinstance(obj, dict):
                records.append(obj)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SourceError(f"{name}: invalid JSON ({e.msg} at line {e.lineno})")
        if isinstance(data, dict):
            # A single object, or a wrapper around the real list.
            lists = [v for v in data.values() if isinstance(v, list) and v
                     and isinstance(v[0], dict)]
            if len(lists) == 1:
                records = lists[0]
                notes.append("records were found nested inside a wrapper object")
            else:
                records = [data]
        elif isinstance(data, list):
            records = [r for r in data if isinstance(r, dict)]

    if not records:
        return Source(name=name, kind="json", text=text, notes=notes)

    # Union of keys, in first-seen order, so columns are stable and predictable.
    header: list[str] = []
    seen = set()
    for rec in records:
        for k in rec:
            if k not in seen:
                seen.add(k)
                header.append(k)

    rows = [header]
    for rec in records:
        rows.append([
            "" if rec.get(k) is None
            else rec[k] if isinstance(rec.get(k), str)
            else json.dumps(rec[k], ensure_ascii=False)
            if isinstance(rec.get(k), (dict, list))
            else str(rec[k])
            for k in header
        ])

    return Source(
        name=name, kind="jsonl" if lines else "json",
        rows=rows, encoding="utf-8", notes=notes,
    )


_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx(raw: bytes, name: str) -> list[Source]:
    """Read xlsx with the standard library only.

    An xlsx is a zip holding one XML per sheet plus a shared string table. Cell
    values are either inline or an index into that table. This walks both.

    Worth knowing: it reads the STORED values, not what Excel displays. A cell showing
    42 might store 41.999999. That is deliberate. The stored value is what your
    pipeline will actually consume. The displayed one is a formatting illusion, and it
    has caught out a lot of people.
    """
    out: list[Source] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as e:
        raise SourceError(f"{name}: not a readable xlsx ({e})")

    shared: list[str] = []
    if "xl/sharedStrings.xml" in zf.namelist():
        root = ElementTree.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in root.findall(f"{_NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{_NS}t")))

    sheet_files = sorted(
        n for n in zf.namelist()
        if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
    )

    for idx, sheet in enumerate(sheet_files, 1):
        root = ElementTree.fromstring(zf.read(sheet))
        rows: list[list[str]] = []
        for row in root.iter(f"{_NS}row"):
            cells: list[str] = []
            for c in row.findall(f"{_NS}c"):
                v = c.find(f"{_NS}v")
                if v is None:
                    is_el = c.find(f"{_NS}is")
                    cells.append(
                        "".join(t.text or "" for t in is_el.iter(f"{_NS}t"))
                        if is_el is not None else ""
                    )
                    continue
                raw_val = v.text or ""
                if c.get("t") == "s":
                    try:
                        cells.append(shared[int(raw_val)])
                    except (ValueError, IndexError):
                        cells.append(raw_val)
                else:
                    cells.append(raw_val)
            if any(x.strip() for x in cells):
                rows.append(cells)

        if rows:
            out.append(Source(
                name=f"{name}#sheet{idx}", kind="xlsx", rows=rows,
                encoding="utf-8", origin=f"sheet {idx} of {name}",
                notes=["values are as stored, not as displayed by Excel"],
            ))

    if not out:
        raise SourceError(f"{name}: no sheets with data")
    return out


# --- the entry point ---------------------------------------------------------------


def from_bytes(raw: bytes, name: str = "<bytes>", hint: str = "") -> list[Source]:
    """Parse raw bytes into one or more sources. Everything else routes through here."""
    kind, notes = sniff_kind(raw[:SNIFF_BYTES], hint)

    if kind == "parquet":
        raise SourceError(
            f"{name}: Parquet needs pyarrow, which this package deliberately does "
            f"not depend on. Read it yourself and pass the rows in."
        )
    if kind == "xlsx":
        sources = _read_xlsx(raw, name)
        for s in sources:
            s.notes.extend(notes)
        return sources
    if kind in ("json", "jsonl"):
        s = _read_json(raw, name, lines=(kind == "jsonl"))
        s.notes.extend(notes)
        return [s]
    if kind in ("csv", "tsv"):
        s = _read_delimited(raw, "\t" if kind == "tsv" else ",", name)
        s.notes.extend(notes)
        return [s]

    from .solve import detect_encoding
    encoding, _ = detect_encoding(raw[:SNIFF_BYTES])
    return [Source(
        name=name, kind="pdf" if kind == "pdf" else "text",
        text=raw.decode(encoding, errors="replace"),
        encoding=encoding, notes=notes,
    )]


def resolve(
    target: str | Path | Sequence[str | Path],
    recursive: bool = True,
    on_error: str = "report",
) -> Iterator[Source]:
    """Turn anything into a stream of parsed sources.

    Takes a file path, a directory, a glob pattern, an http(s) URL, or any mix of
    those in a list.

    `on_error` controls what happens to unreadable sources. "report" yields a Source
    with the error in `notes`, so a batch run does not stop on one bad file. "raise"
    propagates. "skip" drops it silently. Reporting is the default because in a folder
    of two hundred files, the one that failed is usually the interesting one.
    """
    targets = (
        [target] if isinstance(target, (str, Path))
        else list(target)
    )

    for t in targets:
        t_str = str(t)

        try:
            if _URL.match(t_str):
                raw = _fetch_url(t_str)
                name = t_str.rsplit("/", 1)[-1].split("?")[0] or t_str
                hint = Path(name).suffix
                for s in from_bytes(raw, name=name, hint=hint):
                    s.origin = t_str
                    yield s
                continue

            path = Path(t_str)

            if _GLOB_CHARS.search(t_str) and not path.exists():
                base = Path(path.anchor or ".")
                pattern = t_str[len(str(base)):] if path.is_absolute() else t_str
                for p in sorted(base.glob(pattern)):
                    if _looks_like_data(p):
                        yield from _one_path(p, on_error)
                continue

            if path.is_dir():
                walker = path.rglob("*") if recursive else path.glob("*")
                for p in sorted(walker):
                    if _looks_like_data(p):
                        yield from _one_path(p, on_error)
                continue

            if path.is_file():
                yield from _one_path(path, on_error)
                continue

            raise SourceError(f"{t_str}: not a file, folder, pattern or URL")

        except SourceError as e:
            if on_error == "raise":
                raise
            if on_error == "report":
                yield Source(name=t_str, kind="unknown", origin=t_str,
                             notes=[f"could not read: {e}"])


def _one_path(path: Path, on_error: str) -> Iterator[Source]:
    try:
        raw = path.read_bytes()
        for s in from_bytes(raw, name=path.name, hint=path.suffix):
            s.origin = str(path)
            yield s
    except SourceError as e:
        if on_error == "raise":
            raise
        if on_error == "report":
            yield Source(name=path.name, kind="unknown", origin=str(path),
                         notes=[f"could not read: {e}"])
    except OSError as e:
        if on_error == "raise":
            raise SourceError(f"{path}: {e}") from e
        if on_error == "report":
            yield Source(name=path.name, kind="unknown", origin=str(path),
                         notes=[f"could not read: {e}"])


def inventory(target: str | Path | Sequence[str | Path]) -> str:
    """List what is there and how it was read, without auditing anything.

    Run this first on an unfamiliar folder. Finding out that eleven of your forty
    files are actually tab-delimited despite the .csv extension is usually the whole
    story, and it costs one pass.
    """
    lines: list[str] = []
    counts: dict[str, int] = {}
    problems = 0

    for s in resolve(target):
        counts[s.kind] = counts.get(s.kind, 0) + 1
        if any("could not read" in n for n in s.notes):
            problems += 1
        lines.append(f"  {s}")

    head = [
        f"{plural(sum(counts.values()), 'source')}: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())),
    ]
    if problems:
        head.append(f"{problems} could not be read")
    head.append("")
    return "\n".join(head + lines)
