# intact

**Give it a broken data file. Get back usable data.**

Not a report telling you to go ask your vendor for a better export. The actual data,
fixed, with anything that couldn't be saved kept to one side so the rest can move.

```python
from intact import solve

result = solve("vendor-export.csv")
result.rows          # usable data
result.quarantined   # what couldn't be saved, same columns
print(result.report) # what was done, if you want to know
```

Real output from `python demo.py`, on a file with cp1252 encoding, semicolon
delimiters, Excel-mangled gene symbols, thousands separators, null strings, and one
row with an unescaped delimiter:

```
read as     : cp1252, delimiter ';'
usable rows : 200
quarantined : 1
recovered   : 99.5%

Fixed:
  encoding: detected cp1252 and read with it (utf-8 would have produced mojibake)
  delimiter: detected ';', not a comma, and parsed with it
  ragged rows (1): 1 row had the wrong field count after parsing. Held back rather
    than shifted into the wrong columns
  numeric_as_text (200): removed thousands separators so values parse as numbers
  null_as_string (83): replaced null-like strings with real empty values
  excel_date_corruption (66): reversed Excel's gene-symbol mangling using the
    published mapping (2-Sep -> SEPT2). Only unambiguous symbols were restored

Held back (1 row):
  1 row: field count 7, expected 6 (unescaped delimiter)
```

That gene line is the one worth stopping on. Excel converts gene symbols to dates so
reliably that in 2020 the HGNC gave up and **renamed the genes**. `SEPT2` is now
`SEPTIN2`, because fixing the spreadsheet wasn't working. The mangling is finite and
published, so it reverses exactly.

## Why it exists

Your pipeline reads a file. You get back a dataframe with the right columns and the
right row count. Nothing throws an error. Some of the values are still wrong.

Nothing crashed, so nobody looked.

---

## Fix what can be fixed. Isolate what cannot. Never guess into the output.

Some corruption is reversible. Some of it destroys the original information. Those
need opposite responses, so this sorts them before it touches anything.

| Repairable | Unrecoverable |
|---|---|
| **Mojibake**: `café` → `cafÃ©`. The bytes are still there, decoded wrongly. One decode reverses it exactly | **Excel dates**: `SEPT2` → `2-Sep`. You can't know whether that was a gene or a real September date |
| **Numbers as text**: `"1,234"`. Parse it | **Truncation**: the characters past the `VARCHAR(255)` limit aren't in the file |
| **Null strings**: `"N/A"`, `"NULL"`, `"-"` sitting where nulls belong | **Digits as control bytes**: the numbers don't exist in any form |
| **Duplicate column names**: silently dropped by most loaders | **Missing glyphs**: inferring them from context is fabrication, not repair |

Where a domain mapping exists, the right-hand column shrinks. Gene symbols are the
worked example. Where it doesn't, those rows get **quarantined instead of guessed
at**, and the rest of the data flows through immediately rather than the whole load
blocking on them.

Quarantine is a dataset, not an error. Same columns, same order, plus a reason.

---

## Severity depends on what the data is for

A truncated `notes` column doesn't matter to a search index and is fatal to a
compliance archive. Lost leading zeros are cosmetic in a report and catastrophic the
moment you join on that column. Reporting one fixed severity helps nobody. It's too
alarming for the person who doesn't care and too quiet for the person whose pipeline
it's about to destroy.

Declare what you're doing with the data and the same measurements get scored against
it:

```
use case        verdict   issues  blocking
--------------  --------  ------  ------------------------------
search_index    suspect        2  -
analytics       corrupt        5  excel_date_corruption, null_as_string, numeric_as_text, ragged_rows
joins           corrupt        5  excel_date_corruption, leading_zero_loss, ragged_rows
scientific      corrupt        5  excel_date_corruption, leading_zero_loss, ragged_rows
archive         corrupt        5  excel_date_corruption, leading_zero_loss, ragged_rows
classification  suspect        1  -

Same data, same measurements. Which pipelines may consume it differs.
```

```python
from intact.profiles import apply_profile, JOINS, SEARCH_INDEX

apply_profile(column_result, JOINS)          # corrupt, the keys won't match
apply_profile(column_result, SEARCH_INDEX)   # clean, irrelevant to indexing
```

Suppressed findings are kept, not thrown away, so you can re-score one audit for a
different pipeline later without re-reading the data.

**Where that matters:** a vendor export you have to accept or reject. A migration
where you need to know what breaks. A RAG corpus before you index it. A key column
before you join on it. An archive you have to certify. One overall verdict can't
answer any of those. "Which use case is it safe for" can.

---

## What you need to know before you use it

Not much, and I have tried to keep it that way.

**To run it: Python 3.10 or newer. Nothing else.** No pip install, no dependencies, no
API key, no network. If you can run `python`, you can run this. Excel files and
databases work out of the box too, because I read xlsx with `zipfile` and talk to
databases through whatever driver you already have.

**You do not need to know anything about encodings, or what mojibake is, or what a
VARCHAR limit does.** That is the whole point. You hand it a file, it tells you what
is wrong in plain words, and `solve()` hands the data back fixed.

**One thing you do need to decide: what the data is for.** A truncated notes column
does not matter for a search index and is fatal for an archive. Pass a profile and the
same measurements get scored against your actual use. If you do not know yet, do not
pass one, and everything gets reported as measured.

**Read the output, do not obey it.** It is built to be reviewed. A column of genuine
September dates will look like Excel corruption to it, and it will say so rather than
quietly deciding for you.

**It never guesses into your data.** Anything that cannot be recovered goes to
`quarantined` with a reason attached. If you only remember one thing, remember that:
the good rows move, the bad rows are somewhere specific rather than somewhere unknown.

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

Repair what's safely repairable:

```python
from intact.repair import repair_column

fixed, result = repair_column("name", values, findings)
print(result)                 # every change, with before -> after
print(result.is_clean_after)  # False if anything was refused
```

No dependencies. Standard library only.

---

## What it detects

**Six built-in profiles**: `search_index`, `analytics`, `joins`, `scientific`,
`archive`, `classification`, plus `RAW` for when you don't know yet, and `custom()`
for anything else.

**Tabular**: mojibake, numbers carrying display formatting, ragged rows (unescaped
delimiters), `VARCHAR` truncation, null-like strings, Excel date corruption, lost
leading zeros, duplicate column names.

**Text and PDF extraction**: missing glyphs from incomplete font maps, shattered
words from character-positioning failures, and digits written as unmapped control
bytes.

That last one deserves its own note. The prose reads perfectly, every number is gone,
and nothing in a normal pipeline notices. If you're extracting figures from
documents, it's the one to worry about.

---

## Point it at anything

A file, a folder, a pattern, or a URL. One call:

```python
from intact.scan import scan

print(scan("export.csv"))
print(scan("data/"))                       # every data file, recursively
print(scan("*.jsonl"))
print(scan("https://example.org/data.csv"))
print(scan(["a.csv", "reports/", "*.json"]))
```

**Formats:** CSV, TSV, JSON, JSON-Lines, xlsx, plain text.

**And databases**, because corruption is usually already in the warehouse long before
it reaches a file:

```python
import sqlite3            # or psycopg, pyodbc, snowflake, duckdb...
from intact.database import audit_table, audit_database

conn = sqlite3.connect("warehouse.db")
print(audit_table(conn, "customers"))
print(audit_database(conn))          # discovers the tables itself
```

Any driver implementing PEP 249 works, which is all of them, so this needs no new
dependency. You make the connection and this borrows it. A data-quality library has
no business holding database passwords. Table names get checked against an allowlist
and quoted rather than interpolated, and there's a row limit by default, because this
is usually running against a database somebody else depends on.

`.xlsx` is read with `zipfile` and `xml.etree`. An xlsx is a zip of XML and both are
standard library, so the zero-dependency promise survives Excel. It reads the
**stored** values, not the displayed ones. A cell showing `42` may store `41.999999`,
and the stored value is what your pipeline will actually consume.

Parquet is the deliberate exception. There's no reasonable stdlib path, so it gets
detected, named and skipped with a clear message instead of half-supported.

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

Auditing thirty exports one at a time misses what's only visible between them:

```
Across files:
  [SUSPECT] inconsistent_headers @ across files: 1 field named differently across
  files. A union or append will produce extra columns full of nulls instead of an
  error
    examples: 'CustomerID / customer_id'

  [SUSPECT] odd_file_out @ across files: 2 of 3 files share a column layout. 1 file
  did not. Check these are the same kind of export before you combine them
    examples: 'mar.csv'
```

In a folder scan, that's usually the finding.

---

## It learns your data

The shipped thresholds encode what corruption looked like on one corpus. Yours is
different. A 1970s journal scan, a born-digital preprint and a bank's CSV export all
fail differently.

```python
from intact import Pipeline
from intact.detectors.tabular import TabularDetector

pipe = Pipeline([TabularDetector()], log_path="audit-log.jsonl")

result = pipe.audit(rows, subject="export.csv")   # works day one, on defaults

for rec in pipe.quarantine():                     # flagged, worst first
    pipe.label(rec.record_id, "discard")          # or "keep"
```

At 40 labels it refits every threshold to your corpus, keeping a learned value only
when it beats the default by 3% or more on balanced accuracy. **Below 40 it refuses
and says so.** A boundary drawn through nine examples is confident and wrong.

**Features get logged, never verdicts.** A verdict is just a threshold applied to a
feature. Log the feature and you can recompute any past verdict and replay any
threshold change against the whole history. Log the verdict and that information is
gone.

### It also grows rules it wasn't shipped with

`evolution.py` compares what you rejected against what you kept and mines patterns
that aren't among the shipped detectors: character classes, token shapes, marker
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

It also tracks whether each detector agrees with you, and says which ones to retire:

```
text.shattered     : never fired. Consider removing it, since it costs time and
                     finds nothing
text.numeric_loss  : fired often, never agreed with you. Retire it
```

**Proposals get surfaced, never auto-applied.** A tool that quietly invents its own
rules can't be debugged, and it would happily learn that every reject contains the
word "the".

---

## Design decisions

**Evidence, not verdicts.** Every finding carries the measured value, the threshold it
crossed, and real examples. You can disagree with a threshold without losing the
measurement.

**Worst wins, never the average.** One corrupt column in thirty makes the dataset
corrupt. Averaging gives you a reassuring number attached to an unusable dataset.

**Refusing to judge is a valid answer.** Below the minimum row count, detectors return
nothing and say why. Statistics on forty rows are noise.

**Column-wise, not file-wise.** Corruption concentrates. A file-level score hides it.

---

## Honest limitations

- **Thresholds come from one corpus.** They're a starting point, which is exactly why
  the learning loop exists. Treat the shipped numbers as a hypothesis.
- **English-centric.** The shattered-word detector uses English word-length statistics
  and will misfire on languages with different morphology.
- **Mojibake detection covers UTF-8-as-Latin-1/CP1252**, the common case. Other
  encoding confusions aren't handled yet.
- **No OCR.** If a PDF is a scan, this tells you the text layer is empty. It doesn't
  read the image.
- **False positives are possible** on unusual but legitimate data. A column of genuine
  September dates will look like Excel corruption. The output is meant to be reviewed,
  not obeyed.

---

## A note on the tests

While building the shattered-word detector I wrote a version that returned **"clean"
on deliberately shattered text.** The test passed, because my test data was wrong. It
split each word once, which left a long tail piece that broke the pattern the detector
was looking for. Real corruption fragments a word throughout.

I only caught it by printing the raw measurements instead of trusting the green tick.

Both the fix and a regression test for that exact case are in
`tests/test_text_detector.py`. It's the most instructive thing in the repo. A detector
that can't detect is worse than no detector, because it produces confident silence,
which is the same failure mode the whole library is about.

```
96 passed, 0 failed
```

Run them with:

```
python -m unittest discover -s tests
```

---

## Two kinds of detector, and they find different things

Everything above applies a rule written in advance. 255 characters is suspicious,
these strings mean null, this regex is a date. Rules written in advance are wrong in
ways their author can't see coming, which is how my truncation rule produced two false
positives on real NYC data before I added the guards.

`detectors/consistency.py` carries no rules about corruption. It learns what each
column looks like, then reports what doesn't fit:

```
Learned conventions (nothing was told to this detector in advance):

  order_id    100% follow 'AAA-dddd'
  order_date   99% follow 'dddd-dd-dd'  (1 other shape)
  gene_code   no dominant pattern (4 shapes, top 39%): free text, not checked
  phone        99% follow '(ddd) ddd-dddd'  (1 other shape)

  [SUSPECT] convention_break @ column 'order_date': 99% of values follow the
  pattern 'dddd-dd-dd'. 1 value did not. Nothing here knows what this column is
  meant to hold. These values just do not match what the rest of the column does
    examples: "'2-Sep' (shape 'd-Mmm')"

  [SUSPECT] convention_break @ column 'phone': 99% of values follow the pattern
  '(ddd) ddd-dddd'. 1 value did not. Nothing here knows what this column is meant
  to hold. These values just do not match what the rest of the column does
    examples: "'5550109' (shape 'dddd')"
```

It found a mangled date and an unformatted phone number **without being told what a
date or a phone number is**. It needs no labels, no training and no prior knowledge,
because the thing it compares against is the file itself.

It also says nothing about `gene_code`, which is correct. Gene symbols have no shared
shape, so there's no convention for an anomaly to break.

### The two approaches don't overlap

Same file, three planted anomalies, both detectors run:

| | Found |
|---|---|
| Consistency | `order_date`, `phone` |
| Fixed rules | `gene_code`, `free_notes` |

Neither found anything the other did. Consistency catches anything that breaks a
column's own convention, including failure modes that have no name, but it's blind
where a column has no single shape. Fixed rules need you to know the failure mode in
advance, but they work regardless of what the rest of the column looks like.

**Each covers the other's blind spot**, which is why both ship.

---

## Related work

This sits alongside existing tools rather than replacing them, and several of its
detectors have better-established equivalents. Worth knowing before you pick it:

| Tool | What it does that overlaps |
|---|---|
| [**ftfy**](https://github.com/rspeer/python-ftfy) | Mojibake detection and repair, including multi-layer cases, with `fix_and_explain()`. More thorough than the detector here. If mojibake is your only problem, use ftfy |
| [**Frictionless**](https://github.com/frictionlessdata/frictionless-py) | Ragged rows, blank rows, duplicate headers as first-class structural errors |
| [**Great Expectations**](https://greatexpectations.io/expectations/) · [**Soda**](https://docs.soda.io/) · [**Pandera**](https://pandera.readthedocs.io/) | Declarative validation. You assert what should be true, they check it |
| [**Deequ / PyDeequ**](https://github.com/awslabs/python-deequ) | Spark-native constraint verification, metrics over time, constraint suggestion |
| [**DataProfiler**](https://capitalone.github.io/DataProfiler/overview.html) | Profiling, including distinct null-like string representations |
| [**Databricks DLT expectations**](https://docs.databricks.com/aws/en/delta-live-tables/expectation-patterns) | A documented quarantine pattern: flag invalid records, split valid from invalid |
| [**Raha / Baran**](https://github.com/BigDaMa/raha) · [**HoloClean**](http://www.vldb.org/pvldb/vol10/p1190-rekatsinas.pdf) | Research systems that learn error detection and repair from a small number of user labels |

**The difference in approach:** validation frameworks check assertions you wrote. This
looks for specific pathologies you'd have to already know about in order to assert
them. A spike of values at exactly 255 characters. A gene symbol that became a date.
Digits written as unmapped bytes.

**What I'm not claiming:** none of these detectors is the first of its kind, and the
mojibake one isn't benchmarked against ftfy. The fitness-for-use idea behind profiles
is decades old in the data-quality literature. What's implemented here is a mechanism
for it, not the idea. The threshold refit is a one-dimensional sweep over labels. It
isn't machine learning and shouldn't be described as such.

**What I haven't measured:** detection precision and recall against a labelled corpus.
I validated the detectors by pointing them at real data (NYC 311 open data) and
investigating every finding by hand, which caught two false positives. That's
anecdote, not a benchmark, and I've called it anecdote throughout.

---

## Licence

MIT.
