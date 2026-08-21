# intact

**Hand it a broken data file. Get back usable data.**

Not a report telling you to ask your vendor for a better export. The data, fixed,
with whatever genuinely could not be saved held to one side so the rest can move.

```python
from intact import solve

result = solve("vendor-export.csv")
result.rows          # usable data
result.quarantined   # what could not be saved, same columns
print(result.report) # what was done, if you want to know
```

Real output, from a real mess — cp1252 encoding, semicolon delimiters,
Excel-mangled gene symbols, thousands separators, null strings, truncated fields
and a row with an unescaped delimiter:

```
read as     : cp1252, delimiter ';'
usable rows : 192
quarantined : 9
recovered   : 95.5%

Fixed:
  encoding: detected cp1252 and read with it (utf-8 would have produced mojibake)
  delimiter: detected ';', not a comma - parsed with it
  numeric_as_text (200): removed thousands separators so values parse as numbers
  null_as_string (97): replaced null-like strings with real empty values
  excel_date_corruption (87): reversed Excel's gene-symbol mangling using the
    published mapping (2-Sep -> SEPT2)
  truncation (8): isolated rows with cut-off values so the remaining 192 rows
    are usable now
```

That gene line is the one worth pausing on. Excel converts gene symbols to dates so
reliably that in 2020 the HGNC gave up and **renamed the genes** - `SEPT2` is now
`SEPTIN2` - because fixing the spreadsheet was not working. The mangling is finite
and published, so it reverses exactly.

## The problem it exists for

Your pipeline reads a file. It returns a dataframe with the right columns and the
right row count. Nothing errors. Some fraction of the values are wrong.

Not the crash - the silence.

---

## Fix what can be fixed. Isolate what cannot. Never guess into the output.

Some corruption is reversible and some destroys the original information. Those need
opposite responses, so this sorts them before touching anything:

| Repairable | Unrecoverable |
|---|---|
| **Mojibake** — `café` → `cafÃ©`. The bytes are still there, decoded wrongly. One decode reverses it exactly | **Excel dates** — `SEPT2` → `2-Sep`. You cannot know whether that was a gene or a real September date |
| **Numbers as text** — `"1,234"`. Parse it | **Truncation** — the characters past the `VARCHAR(255)` limit are not in the file |
| **Null strings** — `"N/A"`, `"NULL"`, `"-"` sitting where nulls belong | **Digits as control bytes** — the numbers do not exist in any form |
| **Duplicate column names** — silently dropped by most loaders | **Missing glyphs** — inferring them from context is fabrication, not repair |

Where a domain mapping exists, the right-hand column shrinks - gene symbols are the
worked example. Where it does not, those rows are **quarantined rather than guessed
at**, and the remaining data flows through immediately rather than the whole load
blocking on them.

Quarantine is a dataset, not an error: same columns, same order, plus a reason.

```
Held back (9 rows):
  8 rows - value in 'notes' truncated at a length limit - the rest of the
           text is not in this file
  1 rows - field count 7, expected 6 - unescaped delimiter
```

---

## Severity depends on what the data is for

A truncated `notes` column is irrelevant to a search index and fatal to a compliance
archive. Lost leading zeros are cosmetic in a report and catastrophic the moment you
join on that column. Reporting one fixed severity is unhelpful to everybody — too
alarming for the person who does not care, too quiet for the person whose pipeline it
will destroy.

Declare what you are doing with the data, and the same measurements are scored
against it:

```
use case        verdict   issues  blocking
--------------  --------  ------  ------------------------------
classification  clean          0  -
search_index    corrupt        3  mojibake
analytics       corrupt        7  excel_date_corruption, null_as_string, numeric_as_text
joins           corrupt        7  excel_date_corruption, leading_zero_loss, mojibake, truncation
archive         corrupt        7  excel_date_corruption, leading_zero_loss, mojibake, truncation

Same data, same measurements. Which pipelines may consume it differs.
```

```python
from intact.profiles import apply_profile, JOINS, SEARCH_INDEX

apply_profile(column_result, JOINS)          # corrupt — the keys will not match
apply_profile(column_result, SEARCH_INDEX)   # clean — irrelevant to indexing
```

Suppressed findings are kept, not discarded, so one audit can be re-scored for a
different pipeline later without re-reading the data.

**Where that matters in practice:** a vendor export you have to accept or reject; a
migration where you need to know what breaks; a RAG corpus before you index it; a key
column before you join on it; an archive you have to certify. A single overall
verdict cannot answer any of those. Which use case it is safe for can.

---

## Quick start

```python
from intact.detectors.tabular import audit_rows
from intact.core import report

import csv
with open("export.csv", encoding="utf-8") as fh:
    rows = list(csv.reader(fh))

print(report(audit_rows(rows)))
```

Repair what is safely repairable:

```python
from intact.repair import repair_column

fixed, result = repair_column("name", values, findings)
print(result)                 # every change, with before -> after
print(result.is_clean_after)  # False if anything was refused
```

No dependencies. Standard library only.

---

## What it detects

**Six built-in profiles** — `search_index`, `analytics`, `joins`, `scientific`,
`archive`, `classification`, plus `RAW` for when you do not yet know, and `custom()`
for anything else.

**Tabular** — mojibake, numbers stored as text, ragged rows (unescaped delimiters),
`VARCHAR` truncation, null-like strings, Excel date corruption, lost leading zeros,
duplicate column names.

**Text and PDF extraction** — missing glyphs from incomplete font maps, shattered
words from character-positioning failures, and digits written as unmapped control
bytes.

That last one deserves its own note. Prose reads perfectly, every number is gone, and
nothing in a normal pipeline notices. If you are extracting figures from documents,
it is the one to fear.

---

## Point it at anything

A file, a folder, a pattern, or a URL — one call:

```python
from intact import scan

print(scan("export.csv"))
print(scan("data/"))                       # every data file, recursively
print(scan("*.jsonl"))
print(scan("https://example.org/data.csv"))
print(scan(["a.csv", "reports/", "*.json"]))
```

**Formats:** CSV, TSV, JSON, JSON-Lines, xlsx, plain text.

**And databases** — corruption is usually already in the warehouse long before it
reaches a file:

```python
import sqlite3            # or psycopg, pyodbc, snowflake, duckdb...
from intact.database import audit_table, audit_database

conn = sqlite3.connect("warehouse.db")
print(audit_table(conn, "customers"))
print(audit_database(conn))          # discovers the tables itself
```

Any driver implementing PEP 249 works, which is all of them, so this needs no new
dependency. You make the connection; this borrows it — a data-quality library has no
business holding database passwords. Table names are validated against an allowlist
and quoted rather than interpolated, and there is a row limit by default because this
is usually running against a database somebody else depends on.

`.xlsx` is read with `zipfile` and `xml.etree` — an xlsx is a zip of XML, and both
are standard library, so the zero-dependency promise survives Excel. It reads the
**stored** values, not the displayed ones: a cell showing `42` may store
`41.999999`, and the stored value is what your pipeline will actually consume.

Parquet is the deliberate exception. There is no reasonable stdlib path, so it is
detected, named, and skipped with a clear message rather than half-supported.

### Format is decided by content, not extension

Extensions lie constantly, so the extension is a hint and the content is the
evidence:

```
plain.csv      -> csv
quoted.csv     -> csv     (a column holding "1,234" no longer defeats the sniffer)
tabs.csv       -> tsv     content is tab-delimited despite the name
prose.txt      -> text    a comma in a sentence is not a delimiter
book.xlsx      -> xlsx    read with the standard library
README.md      -> skipped
bad.json       -> reported, and the scan continues
```

### Findings that only exist across files

Auditing thirty exports one at a time misses what is only visible between them:

```
Across files:
  [SUSPECT] inconsistent_headers: 1 field(s) are named differently in different
  files. A union or append across these will produce extra columns with
  mostly-null values rather than an error
    examples: 'CustomerID / customer_id'

  [SUSPECT] odd_file_out: 2 of 3 files share a column layout; 1 do not.
    examples: 'mar.csv'
```

In a folder scan, that is usually the finding.

---

## It learns your data

Shipped thresholds encode what corruption looked like on one corpus. Yours differs —
a 1970s journal scan, a born-digital preprint and a bank's CSV export all fail
differently.

```python
pipe = Pipeline([TabularDetector()], log_path="audit-log.jsonl")

result = pipe.audit(rows, subject="export.csv")   # works day one, on defaults

for rec in pipe.quarantine():                     # flagged, worst first
    pipe.label(rec.record_id, "discard")          # or "keep"
```

At 40 labels it refits every threshold to your corpus, keeping a learned value only
when it beats the default by 3%+ on balanced accuracy. **Below 40 it refuses and says
so** — a boundary drawn through nine examples is confident and wrong.

**Features are logged, never verdicts.** A verdict is a threshold applied to a
feature; log the feature and any past verdict can be recomputed and any threshold
change replayed against the whole history. Log the verdict and that information is
gone.

### It also grows rules it wasn't shipped with

`evolution.py` compares what you rejected against what you kept and mines patterns
that are not among the shipped detectors — character classes, token shapes, marker
strings. On a test corpus carrying a corruption signature none of the built-in
detectors look for, it found it:

```
[character_class] Co
    Unicode category Co (private use) appears at 25.05/1000 chars in rejects
    vs 0.00 in keeps
    20 rejects, 100% precision, 999.0x lift

[token_shape] MIXED_SCRIPT
    20 rejects, 100% precision, 999.0x lift
```

It also tracks whether each detector agrees with you, and says which to retire:

```
text.shattered     : never fired — consider removing
text.numeric_loss  : fired often, never agreed with you — retire it
```

**Proposals are surfaced, never auto-applied.** A tool that quietly invents its own
rules cannot be debugged, and would happily learn that every reject contains the
word "the".

---

## Design decisions

**Evidence, not verdicts.** Every finding carries the measured value, the threshold
it crossed, and real examples. You can disagree with a threshold without losing the
measurement.

**Worst wins, never the average.** One corrupt column in thirty makes the dataset
corrupt. Averaging produces a reassuring number and an unusable dataset — which is
the exact failure this library exists to prevent.

**Refusing to judge is a valid answer.** Below the minimum row count, detectors
return nothing and say why. Statistics on forty rows are noise.

**Column-wise, not file-wise.** Corruption concentrates. A file-level score hides it.

---

## Honest limitations

- **Thresholds come from one corpus.** They are a starting point, which is exactly
  why the learning loop exists. Treat the shipped numbers as a hypothesis.
- **English-centric.** The shattered-word detector uses English word-length
  statistics and will misfire on languages with different morphology.
- **Mojibake detection covers UTF-8-as-Latin-1/CP1252**, the common case. Other
  encoding confusions are not yet handled.
- **No OCR.** If a PDF is a scan, this tells you the text layer is empty. It does not
  read the image.
- **False positives are possible** on unusual but legitimate data — a column of
  genuine September dates will look like Excel corruption. The output is designed to
  be reviewed, not obeyed.

---

## A note on the tests

While building the shattered-word detector I wrote a version that returned **"clean"
on deliberately shattered text.** The test passed, because my test data was wrong —
it split each word once, leaving a long tail piece that broke the pattern the
detector looked for. Real corruption fragments a word throughout.

I only caught it by printing the raw measurements instead of trusting the green tick.

Both the fix and a regression test for that exact case are in
`tests/test_extraction_audit.py`. It is the most instructive thing in the repo: a
detector that cannot detect is worse than no detector, because it produces confident
silence — which is the same failure mode the whole library is about.

```
17/17 passed
```

---

## Two kinds of detector, and they find different things

Everything above applies a rule written in advance: 255 characters is suspicious,
these strings mean null, this regex is a date. Rules written in advance are wrong in
ways their author cannot anticipate — which is how the truncation rule produced two
false positives on real NYC data before the guards were added.

`detectors/consistency.py` carries no rules about corruption. It learns what each
column looks like, then reports what does not fit:

```
Learned conventions (nothing was told to this detector in advance):

  order_id    100% follow 'AAA-dddd'
  order_date  100% follow 'dddd-dd-dd'      (1 other shape)
  gene_code   no dominant pattern (5 shapes, top 38%) - free text, not checked
  phone       100% follow '(ddd) ddd-dddd'  (1 other shape)

  [SUSPECT] convention_break @ column 'order_date'
      examples: '2-Sep' (shape 'd-Mmm')
  [SUSPECT] convention_break @ column 'phone'
      examples: '5550109' (shape 'dddd')
```

It found a mangled date and an unformatted phone number **without being told what a
date or a phone number is**. It needs no labels, no training and no prior knowledge,
because the reference it compares against is the file itself.

It also says nothing about `gene_code`, which is correct: gene symbols have no shared
shape, so there is no convention for an anomaly to break.

### The two approaches have no overlap

Same file, three planted anomalies, both detectors run:

| | Found |
|---|---|
| Consistency | `order_date`, `phone` |
| Fixed rules | `gene_code`, `free_notes` |

Neither found anything the other did. Consistency catches anything that breaks a
column's own convention, including failure modes that have no name — but it is blind
where a column has no single shape. Fixed rules require knowing the failure mode in
advance, but work regardless of what the rest of the column looks like.

**Each covers the other's blind spot**, which is why both ship.

*(Credit where due: the idea for this detector came from someone pointing out that
a person tends to write dates the same way throughout a document, so a deviation is
detectable without knowing anything about dates.)*

---

## Related work

This sits alongside existing tools rather than replacing them, and several of its
detectors have better-established equivalents. Worth knowing before you choose it:

| Tool | What it does that overlaps |
|---|---|
| [**ftfy**](https://github.com/rspeer/python-ftfy) | Mojibake detection and repair, including multi-layer cases, with `fix_and_explain()`. More thorough than the detector here. If mojibake is your only problem, use ftfy |
| [**Frictionless**](https://github.com/frictionlessdata/frictionless-py) | Ragged rows, blank rows, duplicate headers as first-class structural errors |
| [**Great Expectations**](https://greatexpectations.io/expectations/) · [**Soda**](https://docs.soda.io/) · [**Pandera**](https://pandera.readthedocs.io/) | Declarative validation. You assert what should be true; they check it |
| [**Deequ / PyDeequ**](https://github.com/awslabs/python-deequ) | Spark-native constraint verification, metrics over time, constraint suggestion |
| [**DataProfiler**](https://capitalone.github.io/DataProfiler/overview.html) | Profiling including distinct null-like string representations |
| [**Databricks DLT expectations**](https://docs.databricks.com/aws/en/delta-live-tables/expectation-patterns) | A documented quarantine pattern: flag invalid records, split valid/invalid |
| [**Raha / Baran**](https://github.com/BigDaMa/raha) · [**HoloClean**](http://www.vldb.org/pvldb/vol10/p1190-rekatsinas.pdf) | Research systems that learn error detection and repair from a small number of user labels |

**The difference in approach**, stated as neutrally as I can manage: validation
frameworks check assertions you wrote. This looks for specific pathologies you would
have to know about in advance to assert — a spike of values at exactly 255
characters, a gene symbol that became a date, digits written as unmapped bytes.

**What is not claimed:** none of these detectors is the first of its kind, and the
mojibake one is not benchmarked against ftfy. The fitness-for-use idea behind
profiles is decades old in the data-quality literature; what is implemented here is a
mechanism for it, not the idea. The threshold refit is a one-dimensional sweep over
labels — not machine learning, and it should not be described as such.

**What has not been measured:** detection precision and recall against a labelled
corpus. The detectors were validated by pointing them at real data (NYC 311 open
data) and investigating every finding by hand, which caught two false positives — but
that is anecdote, not a benchmark, and it is described as anecdote throughout.

---

## Licence

MIT.
