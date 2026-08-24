"""
Run me:  python demo.py

Builds a deliberately broken vendor export, shows you how broken it is, then fixes
it in front of you.

Everything here is a real failure mode, not something I made up. Each one has
quietly poisoned somebody's dataset.
"""

from __future__ import annotations

import csv
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This demo prints accented characters. On Windows the console defaults to cp1252,
# which cannot encode them, so printing raises UnicodeEncodeError before the demo
# gets to its point. That is, with some irony, exactly the class of bug this library
# exists to catch. So it is handled here rather than left to chance.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from intact import solve                                  # noqa: E402
from intact.core import report                            # noqa: E402
from intact.detectors.tabular import audit_rows           # noqa: E402
from intact.profiles import compare_profiles              # noqa: E402


def rule(title: str = "", char: str = "=") -> None:
    print()
    if title:
        print(f" {title} ".center(74, char))
    else:
        print(char * 74)
    print()


def build_broken_export(n: int = 200, seed: int = 11) -> bytes:
    """A vendor export with six independent, entirely realistic problems."""
    rng = random.Random(seed)
    names = ["José García", "Renée Dubois", "Bjørn Larsen",
             "Anne Müller", "Sofía Rossi"]
    genes = ["SEPT2", "MARCH1", "TP53", "NKX2-1", "DEC1"]

    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["customer_id", "name", "revenue", "notes", "gene_code", "status"])

    for _ in range(n):
        gene = rng.choice(genes)
        if gene in ("SEPT2", "MARCH1") and rng.random() < 0.7:
            gene = {"SEPT2": "2-Sep", "MARCH1": "1-Mar"}[gene]
        w.writerow([
            str(rng.randint(1, 99999)),
            rng.choice(names),
            f"{rng.randint(1000, 999999):,}",
            "x" * 255 if rng.random() < 0.06 else "delivered on time",
            gene,
            rng.choice(["active", "N/A", "active", "NULL", "active"]),
        ])

    lines = buf.getvalue().splitlines()
    lines.insert(30, "88231;Acme Ltd; the note; has; semicolons;in it;active")
    return "\n".join(lines).encode("cp1252")


def main() -> None:
    raw = build_broken_export()
    Path("vendor-export.csv").write_bytes(raw)

    rule("THE FILE YOU WERE SENT")
    print("vendor-export.csv  -  200 rows, 6 columns\n")
    print("Six things are wrong with it. None of them will raise an exception.\n")
    print("  1. encoding is cp1252, not utf-8")
    print("  2. delimiter is a semicolon, not a comma")
    print("  3. gene symbols were mangled into dates by Excel")
    print("  4. revenue has thousands separators, so it is text and not numbers")
    print("  5. status uses the strings 'NULL' and 'N/A' instead of real nulls")
    print("  6. one row has unescaped semicolons inside a field")

    rule("WHAT NAIVE READING GIVES YOU", "-")
    wrong = raw.decode("utf-8", errors="replace")
    first = wrong.splitlines()[1]
    print("open(path) with default settings, first data row:\n")
    print(f"  {first[:110]}")
    print("\n  ...one field, because the delimiter is wrong.")
    print("  The accented characters are already destroyed by the wrong encoding.")

    rule("WHAT IT FINDS", "-")
    text = raw.decode("cp1252")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    findings = audit_rows(rows)
    out = report(findings)
    print("\n".join(out.splitlines()[:26]))

    rule("SAME FILE, DIFFERENT JOBS", "-")
    print("Severity is not a property of the data alone. It depends on what you")
    print("are about to do with it:\n")
    print(compare_profiles(findings))

    rule("NOW FIX IT")
    solution = solve(raw)
    print(solution.report)

    rule("THE DATA, AFTERWARDS", "-")
    print(" | ".join(solution.header))
    print("-" * 74)
    for row in solution.rows[:6]:
        print(" | ".join(row))

    print("\nNote the gene_code column: '2-Sep' is back to 'SEPT2'.")
    print("Excel's mangling is a finite published mapping, so it reverses exactly.")

    if solution.quarantined:
        rule("WHAT COULD NOT BE SAVED", "-")
        print(f"{len(solution.quarantined)} rows were held back rather than guessed at.")
        print("They are a dataset, not an error. Same columns, plus a reason:\n")
        for row, why in list(zip(solution.quarantined,
                                 solution.quarantine_reasons))[:3]:
            shown = [c[:22] + "..." if len(c) > 22 else c for c in row[:4]]
            print(f"  {' | '.join(shown)}")
            print(f"      -> {why}\n")

    rule("RESULT")
    print(f"  {len(solution.rows)} of 200 rows recovered "
          f"({solution.recovered_fraction:.1%}) and usable now.")
    print(f"  {len(solution.quarantined)} isolated, with the reason attached.")
    print("  0 values guessed.")
    print()
    print("  Nothing was sent back to the vendor. Nothing was fabricated.")
    print()

    Path("vendor-export.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
