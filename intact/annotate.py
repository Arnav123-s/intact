"""
Annotate — repaired data that still carries its own history.

A repaired dataset that threw away the original is not trustworthy. It is just
corruption you feel better about. Six months later nobody remembers which values got
touched, by what rule, or whether the rule was right, and there is no way to find
out.

So every output here carries three things together:

    the repaired value  +  the original  +  why it changed

Nothing gets destroyed. Every changed cell traces back to the rule that changed it,
every unrepairable finding gets annotated in place instead of quietly left, and
anything needing a person gets queued with enough context to decide.

Three outputs, for three audiences
-----------------------------------
`write_csv`      the repaired table plus a sidecar provenance file. The table loads
                 normally in anything. The sidecar answers "what did you change?"

`write_jsonl`    one record per row with per-cell provenance inline. For pipelines
                 that want to carry the history downstream instead of dropping it at
                 the first hop.

`review_sheet`   a markdown sheet of what needs a person, worst first, with enough
                 context to decide without opening the source file.

Why a sidecar instead of extra columns
---------------------------------------
Adding `name__original` next to `name` changes the schema, which breaks every
downstream consumer expecting the original columns. A separate provenance file keeps
the repaired table drop-in compatible and puts the history one join away. Use
`inline=True` when you would rather have it all in one file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from .core import Finding, Severity
from .repair import Recoverability, RepairResult


@dataclass(frozen=True)
class CellNote:
    """What happened to one cell, and why."""

    row: int
    column: str
    original: str
    value: str
    mode: str
    action: str                # "repaired" | "flagged" | "unrepairable"
    recoverability: str
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.original != self.value


@dataclass
class Annotated:
    """A table plus the full record of what was done to it."""

    header: list[str]
    rows: list[list[str]]
    notes: list[CellNote] = field(default_factory=list)
    column_findings: dict[str, list[Finding]] = field(default_factory=dict)

    @property
    def changed_cells(self) -> list[CellNote]:
        return [n for n in self.notes if n.changed]

    @property
    def needs_review(self) -> list[CellNote]:
        return [n for n in self.notes if n.action in ("flagged", "unrepairable")]

    def summary(self) -> str:
        changed = len(self.changed_cells)
        review = len(self.needs_review)
        cols = {n.column for n in self.notes}
        lines = [
            f"rows            : {len(self.rows)}",
            f"columns touched : {len(cols)}",
            f"cells changed   : {changed}",
            f"needs review    : {review}",
        ]
        if review:
            lines.append("")
            lines.append(
                "Values needing review were NOT changed. They are annotated in place, "
                "so nothing is lost and nothing is silently accepted."
            )
        return "\n".join(lines)


def annotate(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    repairs: dict[str, tuple[list[str], RepairResult]],
    findings: dict[str, list[Finding]] | None = None,
) -> Annotated:
    """Merge per-column repairs back into a table, recording every change.

    `repairs` maps column name -> (repaired values, RepairResult), which is exactly
    what `repair.repair_column` returns.
    """
    header = list(header)
    original = [list(r) for r in rows]
    out = [list(r) for r in rows]
    notes: list[CellNote] = []

    for col, (new_values, result) in repairs.items():
        if col not in header:
            continue
        idx = header.index(col)

        repaired_modes = {r.mode: r for r in result.repairs}
        refused_modes = {r.mode: r for r in result.refusals}

        for i, new in enumerate(new_values):
            if i >= len(out):
                break
            old = original[i][idx] if idx < len(original[i]) else ""
            if new != old:
                # Attribute the change to whichever repair rule ran on this column.
                rep = next(iter(repaired_modes.values()), None)
                out[i][idx] = new
                notes.append(CellNote(
                    row=i, column=col, original=old, value=new,
                    mode=rep.mode if rep else "unknown",
                    action="repaired",
                    recoverability=(
                        rep.recoverability.value if rep
                        else Recoverability.REVERSIBLE.value
                    ),
                    note=rep.note if rep else "",
                ))

        # Unrepairable findings annotate the cells they affect without touching them.
        for mode, refusal in refused_modes.items():
            for i, r in enumerate(original):
                val = r[idx] if idx < len(r) else ""
                if val and any(val == s or val in s for s in refusal.examples):
                    notes.append(CellNote(
                        row=i, column=col, original=val, value=val,
                        mode=mode, action="unrepairable",
                        recoverability=Recoverability.IRRECOVERABLE.value,
                        note=f"{refusal.reason} -> {refusal.action}",
                    ))

        if any(
            r.recoverability is Recoverability.LOSSY for r in result.repairs
        ):
            for n in notes:
                if n.column == col and n.action == "repaired" and \
                        n.recoverability == Recoverability.LOSSY.value:
                    object.__setattr__(n, "action", "flagged")

    return Annotated(
        header=header, rows=out, notes=notes,
        column_findings=findings or {},
    )


# --- writers ---------------------------------------------------------------------


def write_csv(
    ann: Annotated, path: str | Path, inline: bool = False
) -> tuple[Path, Path | None]:
    """Write the repaired table, plus provenance.

    Returns (table_path, provenance_path). With `inline=True` the provenance columns
    get appended to the table instead and no sidecar is written. Handy for a one-off
    review, disruptive for anything that consumes the schema.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if inline:
        touched = sorted({n.column for n in ann.changed_cells})
        header = ann.header + [f"{c}__original" for c in touched]
        by_cell = {(n.row, n.column): n.original for n in ann.changed_cells}

        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for i, row in enumerate(ann.rows):
                extras = [by_cell.get((i, c), "") for c in touched]
                w.writerow(list(row) + extras)
        return path, None

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ann.header)
        w.writerows(ann.rows)

    prov = path.with_suffix(".provenance.csv")
    with prov.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "row", "column", "original", "value", "mode", "action",
            "recoverability", "note",
        ])
        for n in ann.notes:
            w.writerow([
                n.row, n.column, n.original, n.value, n.mode, n.action,
                n.recoverability, n.note,
            ])
    return path, prov


def write_jsonl(ann: Annotated, path: str | Path) -> Path:
    """One JSON record per row, with per-cell provenance carried inline.

    For pipelines that should not lose the history at the first hop. Rows with no
    changes carry no `_provenance` key at all, so clean data stays clean on the wire.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    by_row: dict[int, list[CellNote]] = {}
    for n in ann.notes:
        by_row.setdefault(n.row, []).append(n)

    with path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ann.rows):
            rec: dict[str, object] = dict(zip(ann.header, row))
            if i in by_row:
                rec["_provenance"] = [
                    {
                        "column": n.column, "original": n.original,
                        "mode": n.mode, "action": n.action,
                        "recoverability": n.recoverability,
                    }
                    for n in by_row[i]
                ]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def review_sheet(ann: Annotated, path: str | Path | None = None, limit: int = 200) -> str:
    """A markdown sheet of everything needing a person, worst first.

    Ordered so the first thing a reviewer reads is the thing most likely to matter. A
    sheet in file order buries the one unrecoverable column under three hundred
    cosmetic fixes, which is about the same as not writing it.
    """
    order = {"unrepairable": 0, "flagged": 1, "repaired": 2}
    items = sorted(
        ann.needs_review, key=lambda n: (order.get(n.action, 3), n.column, n.row)
    )[:limit]

    lines = [
        "# Review sheet",
        "",
        ann.summary(),
        "",
    ]

    if not items:
        lines.append("Nothing requires review. Every change applied was reversible.")
    else:
        unrep = [n for n in items if n.action == "unrepairable"]
        flagged = [n for n in items if n.action == "flagged"]

        if unrep:
            lines += [
                "## Cannot be repaired",
                "",
                "These values are wrong and the original information is not in the "
                "file. They were left untouched.",
                "",
                "| row | column | value | why |",
                "|---|---|---|---|",
            ]
            for n in unrep:
                why = n.note.split("->")[0].strip()
                lines.append(
                    f"| {n.row} | `{n.column}` | `{n.original}` | {why} |"
                )
            lines.append("")
            actions = sorted({
                n.note.split("->")[-1].strip() for n in unrep if "->" in n.note
            })
            if actions:
                lines.append("**What to do instead:**")
                lines.extend(f"- {a}" for a in actions)
                lines.append("")

        if flagged:
            lines += [
                "## Repaired, but verify",
                "",
                "These changes needed an inference. They are probably right, but "
                "spot-check them before you rely on the data.",
                "",
                "| row | column | was | now | rule |",
                "|---|---|---|---|---|",
            ]
            for n in flagged:
                lines.append(
                    f"| {n.row} | `{n.column}` | `{n.original}` | `{n.value}` | {n.mode} |"
                )
            lines.append("")

    text = "\n".join(lines)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return text
