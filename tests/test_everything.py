"""
Whole-library audit: does every claim the README makes actually hold?

Written after building each piece in isolation and finding a bug every time one was
wired to another. Piecemeal testing found piecemeal bugs; this exercises every
detector, every format, and every module together, with both positive cases (it must
fire) and negative cases (it must stay quiet).

The negative cases matter more. A detector that fires on everything is not a
detector, and a false positive on a customer's clean data costs more credibility than
a miss.

Run:  python tests/test_everything.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from intact.core import Severity, summarise                       # noqa: E402
from intact.detectors import consistency, tabular                 # noqa: E402
from intact.detectors import text as textdet                      # noqa: E402
# Same shadowing as `solve`: the package exports `annotate` the function, which
# hides `annotate` the module. Import what is actually needed from the module.
from intact.annotate import (                                     # noqa: E402
    annotate as make_annotated, review_sheet, write_csv, write_jsonl,
)
# NOTE: `intact.solve` is both a module and the function it exports, and the
# function shadows the module in the package namespace. Import the function
# from its module explicitly rather than relying on attribute access.
from intact.solve import solve as run_solve                       # noqa: E402
from intact import evolution, profiles, repair, scan              # noqa: E402
from intact import sources, stream                                # noqa: E402
from intact.pipeline import Pipeline                              # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append((name, detail))
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark}  {name}" + (f"   [{detail}]" if detail and not condition else ""))


def modes_of(results) -> set[str]:
    return {f.mode for r in results for f in r.findings}


CLEAN_PROSE = (
    "The selection of candidate materials was performed across 1247 samples "
    "collected between 1994 and 2003, yielding a mean response of 0.42 with a "
    "standard deviation of 0.07 across the full extraction procedure. "
) * 40


def clean_table(n: int = 200) -> list[list[str]]:
    rows = [["customer_id", "name", "amount", "joined", "status"]]
    for i in range(n):
        rows.append([
            f"{100000 + i}", f"Person {i}", f"{1000 + i}",
            f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "active" if i % 3 else "closed",
        ])
    return rows


# =====================================================================
print("\nTEXT DETECTOR - must fire")
# =====================================================================
r = textdet.audit_text(CLEAN_PROSE.replace("c", "�"))
check("missing glyph detected", "missing_glyph" in {f.mode for f in r.findings})
check("missing glyph is CORRUPT", r.severity is Severity.CORRUPT)

shattered = " ".join(
    w[i:i + 3] for w in CLEAN_PROSE.split() for i in range(0, len(w), 3)
)
r = textdet.audit_text(shattered)
check("shattered words detected", "shattered_words" in {f.mode for f in r.findings})

nodigits = "".join("\x01" if c.isdigit() else c for c in CLEAN_PROSE)
r = textdet.audit_text(nodigits)
check("numeric loss detected", "numeric_loss" in {f.mode for f in r.findings})

r = textdet.audit_text(CLEAN_PROSE.replace("the", "(cid:87)"))
check("unmapped CID detected", "missing_glyph" in {f.mode for f in r.findings})

print("\nTEXT DETECTOR - must stay quiet")
r = textdet.audit_text(CLEAN_PROSE)
check("clean prose is clean", r.severity is Severity.CLEAN, str(r.findings))
r = textdet.audit_text(CLEAN_PROSE + "See Fig. 3 (p < 0.001); cf. Smith et al., 2011.")
check("citations and symbols do not trip it", r.severity is Severity.CLEAN)
r = textdet.audit_text("Too short to judge.")
check("short text is not judged", not r.judged)
check("empty text does not crash", textdet.audit_text("").units == 0)


# =====================================================================
print("\nTABULAR DETECTOR - must fire")
# =====================================================================
rows = clean_table()
for i in range(1, 80):
    rows[i][1] = "José".encode("utf-8").decode("latin-1")
check("mojibake detected", "mojibake" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
for i in range(1, 201):
    rows[i][2] = f"{1000 + i:,}"
check("numbers-as-text detected", "numeric_as_text" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
for i in range(1, 60):
    rows[i][4] = "N/A"
check("null strings detected", "null_as_string" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
for i in range(1, 40):
    rows[i][1] = f"{'x' * 250}{i:05d}"[:255]
check("truncation detected", "truncation" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
for i in range(1, 60):
    rows[i][3] = "2-Sep" if i % 2 else "1-Mar"
check("excel dates detected", "excel_date_corruption" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
rows[0] = ["id", "name", "amount", "joined", "id"]
check("duplicate headers detected", "duplicate_columns" in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
rows[5] = rows[5] + ["extra"]
rows[9] = rows[9] + ["extra"]
check("ragged rows detected", "ragged_rows" in modes_of(tabular.audit_rows(rows)))

print("\nTABULAR DETECTOR - must stay quiet")
found = modes_of(tabular.audit_rows(clean_table()))
check("clean table is clean", not found, f"got {found}")

# The NYC 311 regressions: both false positives, both must stay silent.
rows = clean_table()
for i in range(1, 201):
    rows[i][1] = "Department of Housing Preservation and Development"  # exactly 50
check("repeated 50-char value is NOT truncation",
      "truncation" not in modes_of(tabular.audit_rows(rows)))

rows = clean_table()
for i in range(1, 30):
    rows[i][1] = f"{i:03d} EAST 9 STREET, MANHATTAN (NEW YORK), NY, 100"[:50]
for i in range(30, 201):
    rows[i][1] = "x" * 60      # something longer exists, so nothing is cut at 50
check("distinct 50-char values with a longer max are NOT truncation",
      "truncation" not in modes_of(tabular.audit_rows(rows)))

rows = clean_table(20)
check("tiny table is not judged on statistics",
      not modes_of(tabular.audit_rows(rows)))


# =====================================================================
print("\nCONSISTENCY DETECTOR")
# =====================================================================
rows = clean_table()
rows[40][3] = "2-Sep"
check("convention break detected", "convention_break" in modes_of(consistency.audit_rows(rows)))

found = modes_of(consistency.audit_rows(clean_table()))
check("consistent column is quiet", not found, f"got {found}")

rows = clean_table()
for i in range(1, 201):
    rows[i][1] = ["short", "a much longer free text value", "mid length"][i % 3]
check("free text is not treated as a convention",
      "convention_break" not in modes_of(consistency.audit_rows(rows)))

rows = clean_table()
for i in range(1, 101):
    rows[i][3] = f"{(i % 28) + 1:02d}/{(i % 12) + 1:02d}/2026"   # a second format
check("two legitimate formats are not corruption",
      "convention_break" not in modes_of(consistency.audit_rows(rows)))


# =====================================================================
print("\nREPAIR - reversible only")
# =====================================================================
original = "José García — café"
check("mojibake round-trips exactly",
      repair.fix_mojibake(original.encode("utf-8").decode("latin-1")) == original)
check("clean text left alone", repair.fix_mojibake("plain ascii") is None)
check("thousands separators removed", repair.fix_numeric_text("1,234,567") == "1234567")
check("plain number untouched", repair.fix_numeric_text("1234") is None)
check("null string mapped", repair.fix_null_string("N/A") == "")
check("real value untouched", repair.fix_null_string("active") is None)
check("duplicate headers suffixed",
      repair.fix_duplicate_headers(["a", "b", "a"]) == ["a__1", "b", "a__2"])

rows = clean_table()
for i in range(1, 40):
    rows[i][1] = f"{'x' * 250}{i:05d}"[:255]
findings = [f for r in tabular.audit_rows(rows) for f in r.findings
            if f.location == "column 'name'"]
_, res = repair.repair_column("name", [r[1] for r in rows[1:]], findings)
check("truncation is refused, not repaired",
      any(x.mode == "truncation" for x in res.refusals))
check("refusal explains what to do",
      all(x.action for x in res.refusals))


# =====================================================================
print("\nPROFILES")
# =====================================================================
rows = clean_table()
for i in range(1, 40):
    rows[i][1] = f"{'x' * 250}{i:05d}"[:255]
col = next(r for r in tabular.audit_rows(rows) if r.subject == "column name")
arch = profiles.apply_profile(col, profiles.ARCHIVE)
srch = profiles.apply_profile(col, profiles.SEARCH_INDEX)
check("truncation is worse for archive than search",
      arch.severity.rank > srch.severity.rank,
      f"archive={arch.severity.value} search={srch.severity.value}")
check("suppressed findings are kept, not lost",
      len(srch.reported) + len(srch.suppressed) == len(col.findings))
check("compare_profiles renders", "use case" in profiles.compare_profiles([col]))


# =====================================================================
print("\nSOURCES - format detection")
# =====================================================================
tmp = Path(tempfile.mkdtemp(prefix="intact_audit_"))

(tmp / "plain.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")
(tmp / "quoted.csv").write_text(
    'id,amount\n1,"1,234"\n2,"99,999"\n3,"7,000"\n', encoding="utf-8")
(tmp / "tabs.csv").write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9\n", encoding="utf-8")
(tmp / "data.json").write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
(tmp / "lines.jsonl").write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
(tmp / "prose.txt").write_text(
    "This is prose, with a comma.\nMore prose here.\nAnd a third line.\nFourth.\n",
    encoding="utf-8")
(tmp / "README.md").write_text("# skip me\n", encoding="utf-8")
(tmp / "bad.json").write_text('{"unterminated', encoding="utf-8")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("xl/sharedStrings.xml",
               '<?xml version="1.0"?><sst xmlns="http://schemas.openxmlformats.org'
               '/spreadsheetml/2006/main"><si><t>gene</t></si><si><t>SEPT2</t></si>'
               "</sst>")
    z.writestr("xl/worksheets/sheet1.xml",
               '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats'
               '.org/spreadsheetml/2006/main"><sheetData>'
               '<row><c t="s"><v>0</v></c></row><row><c t="s"><v>1</v></c></row>'
               "</sheetData></worksheet>")
(tmp / "book.xlsx").write_bytes(buf.getvalue())

by_name = {s.name: s for s in sources.resolve(tmp)}
check("plain csv", by_name["plain.csv"].kind == "csv")
check("csv with quoted delimiters", by_name["quoted.csv"].kind == "csv",
      by_name["quoted.csv"].kind)
check("tsv misnamed .csv detected by content", by_name["tabs.csv"].kind == "tsv")
check("json array", by_name["data.json"].kind == "json")
check("json lines", by_name["lines.jsonl"].kind == "jsonl")
check("prose is not mistaken for csv", by_name["prose.txt"].kind == "text",
      by_name["prose.txt"].kind)
check("xlsx read with stdlib only", "book.xlsx#sheet1" in by_name)
check("README skipped", "README.md" not in by_name)
check("broken json reported not raised",
      any("could not read" in n for n in by_name["bad.json"].notes))
check("inventory renders", "source(s)" in sources.inventory(tmp))


# =====================================================================
print("\nSCAN - folders and cross-file")
# =====================================================================
folder = Path(tempfile.mkdtemp(prefix="intact_scan_"))
for name, header in (("jan.csv", "customer_id"), ("feb.csv", "customer_id"),
                     ("mar.csv", "CustomerID")):
    with (folder / name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([header, "joined", "amount"])
        for i in range(80):
            w.writerow([f"{1000+i}", f"2026-{(i%12)+1:02d}-{(i%28)+1:02d}",
                        f"{1000+i:,}"])

sr = scan.scan(folder)
check("scan finds all files", len(sr.sources) == 3, str(len(sr.sources)))
check("inconsistent headers across files",
      any(f.mode == "inconsistent_headers" for f in sr.cross_file))
check("odd file out flagged",
      any(f.mode == "odd_file_out" for f in sr.cross_file))
check("scan report renders", "scanned" in sr.report())
check("scan accepts a profile",
      scan.scan(folder, profile=profiles.JOINS).profile is profiles.JOINS)


# =====================================================================
print("\nSOLVE - end to end repair")
# =====================================================================
out = io.StringIO()
w = csv.writer(out, delimiter=";", lineterminator="\n")
w.writerow(["id", "name", "amount", "gene", "status"])
for i in range(200):
    w.writerow([str(i), "José", f"{1000+i:,}",
                "2-Sep" if i % 2 else "TP53",
                "N/A" if i % 3 else "active"])
raw = out.getvalue().encode("cp1252")

sol = run_solve(raw)
check("encoding detected", sol.encoding in ("cp1252", "latin-1"), sol.encoding)
check("delimiter detected", sol.delimiter == ";", sol.delimiter)
check("rows recovered", len(sol.rows) == 200, str(len(sol.rows)))
acted = {a.what for a in sol.actions}
# No mojibake repair should appear here: reading cp1252 correctly means the text was
# never mangled in the first place. Detecting the encoding solves it a layer earlier,
# so the check is on the OUTPUT, not on which repair ran.
check("name is correct after decode", sol.rows[0][1] == "José", sol.rows[0][1])
check("no needless mojibake repair", "mojibake" not in acted, str(acted))
check("numbers repaired", "numeric_as_text" in acted)
check("nulls repaired", "null_as_string" in acted)
check("gene symbols restored", "excel_date_corruption" in acted)
gene_col = [r[3] for r in sol.rows]
check("2-Sep became SEPT2", "SEPT2" in gene_col and "2-Sep" not in gene_col)
check("report renders", "recovered" in sol.report)


# =====================================================================
print("\nSTREAM - constant memory")
# =====================================================================
big = folder / "big.csv"
with big.open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "name", "amount"])
    for i in range(30000):
        w.writerow([str(i), "José".encode("utf-8").decode("latin-1"),
                    f"{1000+i:,}"])

sres = stream.solve_stream(big, folder / "big.clean.csv")
check("streamed all rows", sres.rows_in == 30000, str(sres.rows_in))
check("stream repaired mojibake", sres.repairs.get("mojibake", 0) > 0)
check("stream produced audits", len(sres.audits) == 3, str(len(sres.audits)))
check("stream report renders", "rows in" in sres.report)


# =====================================================================
print("\nPIPELINE - learning")
# =====================================================================
log = folder / "audit-log.jsonl"
pipe = Pipeline([tabular.TabularDetector()], log_path=log)
res = pipe.audit(clean_table(), subject="clean.csv")
check("pipeline audits", res is not None)
check("pipeline logs", log.exists())
check("refuses to fit with no labels", pipe.fit(tabular.TabularDetector()) is None)
check("learning report explains why", "defaults" in pipe.learning_report())
q = pipe.quarantine()
check("quarantine returns a list", isinstance(q, list))


# =====================================================================
print("\nEVOLUTION - rule mining")
# =====================================================================
keeps = [CLEAN_PROSE for _ in range(30)]
rejects = [CLEAN_PROSE.replace("o", "", 40) for _ in range(20)]
props = evolution.propose_rules(rejects, keeps)
check("mines a rule from rejections", len(props) > 0, str(len(props)))
check("proposals carry evidence", all(p.support >= 1 for p in props))
check("no rules from too few rejects", evolution.propose_rules(keeps[:2], keeps) == [])
u = [evolution.DetectorUtility("never", fired=0),
     evolution.DetectorUtility("useless", fired=50, agreed=0, overruled=50)]
keep_l, retire = evolution.review_detectors(u)
check("retires useless detectors", set(retire) == {"never", "useless"})


# =====================================================================
print("\nANNOTATE - provenance")
# =====================================================================
vals = ["José".encode("utf-8").decode("latin-1")] * 100
rows2 = [["name"]] + [[v] for v in vals]
f2 = [f for r in tabular.audit_rows(rows2) for f in r.findings]
fixed, rr = repair.repair_column("name", vals, f2)
a = make_annotated(["name"], [[v] for v in vals], {"name": (fixed, rr)})
check("changes recorded", len(a.changed_cells) > 0, str(len(a.changed_cells)))
check("original preserved", all(n.original != n.value for n in a.changed_cells))
p1, p2 = write_csv(a, folder / "annotated.csv")
check("csv + provenance written", p1.exists() and p2 and p2.exists())
check("jsonl written", write_jsonl(a, folder / "annotated.jsonl").exists())
check("review sheet renders", "Review sheet" in review_sheet(a))


# =====================================================================
print("\nCORE - reporting")
# =====================================================================
rows = clean_table()
rows[40][3] = "2-Sep"
allr = tabular.audit_rows(rows) + consistency.audit_rows(rows)
s = summarise(allr)
check("summary takes worst not average", s["severity"] != "clean")
check("summary counts artifacts", s["artifacts"] == len(allr))
from intact.core import report as core_report                      # noqa: E402
check("report renders", "severity" in core_report(allr))


# =====================================================================
print("\n" + "=" * 62)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFAILURES:")
    for name, detail in FAIL:
        print(f"  - {name}" + (f"   [{detail}]" if detail else ""))
sys.exit(1 if FAIL else 0)
