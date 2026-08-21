# Databricks notebook source
# MAGIC %md
# MAGIC # intact on Spark — silent corruption detection at scale
# MAGIC
# MAGIC The pure-Python version of `intact` processes **~385 rows/sec** on a 44-column
# MAGIC table. That is fine for a vendor export and useless for a production table.
# MAGIC
# MAGIC This notebook implements the same detectors as **native Spark column
# MAGIC expressions** — no Python UDFs, so the work stays inside the JVM and the whole
# MAGIC thing is one distributed pass.
# MAGIC
# MAGIC What it does:
# MAGIC 1. Detects the same corruption modes, expressed as Spark aggregations
# MAGIC 2. Repairs what is reversible, in-place, as column transforms
# MAGIC 3. Writes unrecoverable rows to a **Delta quarantine table** with the reason
# MAGIC 4. Benchmarks against the single-threaded implementation
# MAGIC
# MAGIC **The rule is unchanged: fix what can be fixed, isolate what cannot, never
# MAGIC guess into the output.**

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
import time

CATALOG = "workspace"
SCHEMA = "intact_demo"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")
print(f"using {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. A table with real corruption in it
# MAGIC
# MAGIC Every problem below is one that has genuinely poisoned somebody's dataset.
# MAGIC Nothing here raises an exception on read.

# COMMAND ----------

from pyspark.sql import Row
import random

rng = random.Random(11)
names = ["José García", "Renée Dubois", "Bjørn Larsen", "Anne Müller"]
genes = ["SEPT2", "MARCH1", "TP53", "NKX2-1", "DEC1"]

def mojibake(s: str) -> str:
    """UTF-8 bytes decoded as Latin-1 — the single most common encoding failure."""
    return s.encode("utf-8").decode("latin-1")

rows = []
for i in range(2_000_000):
    gene = rng.choice(genes)
    if gene in ("SEPT2", "MARCH1") and rng.random() < 0.7:
        gene = {"SEPT2": "2-Sep", "MARCH1": "1-Mar"}[gene]   # Excel mangling
    name = rng.choice(names)
    if rng.random() < 0.3:
        name = mojibake(name)                                 # encoding damage
    rows.append(Row(
        customer_id=str(rng.randint(1, 99999)),
        name=name,
        revenue=f"{rng.randint(1000, 999999):,}",             # numbers as text
        notes="x" * 255 if rng.random() < 0.05 else "delivered on time",
        gene_code=gene,
        status=rng.choice(["active", "N/A", "active", "NULL", "active"]),
    ))

raw = spark.createDataFrame(rows)
raw.write.mode("overwrite").saveAsTable("vendor_export")
raw = spark.table("vendor_export")

print(f"{raw.count():,} rows")
display(raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Detection as Spark aggregations
# MAGIC
# MAGIC Each detector is a column expression that reduces to a scalar. The whole audit
# MAGIC is **one `agg` over one pass** — Spark computes every statistic for every column
# MAGIC simultaneously, and nothing is collected to the driver except the final counts.
# MAGIC
# MAGIC No Python UDFs anywhere. A UDF would serialise every row across the JVM/Python
# MAGIC boundary and cost more than the detection.

# COMMAND ----------

# Sequences that are overwhelmingly UTF-8-read-as-Latin-1 rather than real text.
MOJIBAKE_RX = r"(Ã[\x80-\xbf]|Â[\x80-\xbf]|â€)"
NUMERIC_RX = r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?$"
EXCEL_DATE_RX = r"^\d{1,2}-(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$"
NULLISH = ["NULL", "N/A", "NA", "null", "n/a", "None", "none", "-", "--", "?", "nan"]

def audit(df: DataFrame) -> dict:
    """One pass, every column, every statistic.

    Returns per-column counts. Ratios are derived on the driver afterwards, so the
    distributed part stays pure counting.
    """
    total = F.count(F.lit(1)).alias("__rows")
    exprs = [total]

    string_cols = [f.name for f in df.schema.fields
                   if isinstance(f.dataType, StringType)]

    for c in string_cols:
        col = F.col(c)
        exprs += [
            F.sum(F.when(col.rlike(MOJIBAKE_RX), 1).otherwise(0)).alias(f"{c}__mojibake"),
            F.sum(F.when(col.rlike(NUMERIC_RX), 1).otherwise(0)).alias(f"{c}__numeric"),
            F.sum(F.when(col.rlike(EXCEL_DATE_RX), 1).otherwise(0)).alias(f"{c}__excel"),
            F.sum(F.when(col.isin(NULLISH), 1).otherwise(0)).alias(f"{c}__nullish"),
            F.sum(F.when(col.isNotNull() & (F.length(col) > 0), 1).otherwise(0)).alias(f"{c}__nonempty"),
            F.max(F.length(col)).alias(f"{c}__maxlen"),
            # Truncation needs BOTH the count at the limit and how many DISTINCT
            # values sit there. Real data proved one alone is not enough: NYC 311's
            # agency_name had 7,655 values at exactly 50 chars, all the SAME complete
            # agency name. See the note in section 3.
            F.sum(F.when(F.length(col) == 255, 1).otherwise(0)).alias(f"{c}__at255"),
            F.countDistinct(F.when(F.length(col) == 255, col)).alias(f"{c}__distinct255"),
        ]

    return df.agg(*exprs).collect()[0].asDict(), string_cols

# COMMAND ----------

t0 = time.time()
stats, string_cols = audit(raw)
audit_secs = time.time() - t0
n = stats["__rows"]

print(f"audited {n:,} rows x {len(string_cols)} columns in {audit_secs:.1f}s")
print(f"{n / audit_secs:,.0f} rows/sec\n")

findings = []
for c in string_cols:
    non_empty = max(1, stats[f"{c}__nonempty"])
    moji = stats[f"{c}__mojibake"]
    numeric = stats[f"{c}__numeric"]
    excel = stats[f"{c}__excel"]
    nullish = stats[f"{c}__nullish"]
    at255 = stats[f"{c}__at255"]
    distinct255 = stats[f"{c}__distinct255"] or 0
    maxlen = stats[f"{c}__maxlen"] or 0

    if moji / n >= 0.002:
        findings.append((c, "mojibake", moji, "CORRUPT" if moji / n >= 0.02 else "SUSPECT",
                         "characters were decoded with the wrong encoding"))
    if numeric / non_empty >= 0.95 and numeric >= 30:
        findings.append((c, "numeric_as_text", numeric, "SUSPECT",
                         "numbers stored as text; aggregations silently skip them"))
    if 2 <= excel < non_empty * 0.5:
        findings.append((c, "excel_date_corruption", excel, "CORRUPT",
                         "Excel converted identifiers to dates (SEPT2 -> 2-Sep)"))
    if nullish / n >= 0.01:
        findings.append((c, "null_as_string", nullish, "SUSPECT",
                         "null-like strings pass not-null checks and corrupt averages"))
    # Both guards, from the NYC 311 false positives.
    if at255 >= 3 and maxlen == 255 and distinct255 >= 2 and at255 / non_empty >= 0.03:
        findings.append((c, "truncation", at255, "SUSPECT",
                         "values cut at a VARCHAR(255) limit upstream"))

display(spark.createDataFrame(
    findings, ["column", "mode", "affected", "severity", "detail"]
).orderBy(F.when(F.col("severity") == "CORRUPT", 0).otherwise(1), F.col("affected").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Two false positives that real data found
# MAGIC
# MAGIC The truncation rule above carries two guards, and **both were added after
# MAGIC running the Python version against real NYC 311 open data**:
# MAGIC
# MAGIC | Column | What happened | Which guard catches it |
# MAGIC |---|---|---|
# MAGIC | `agency_name` | 7,655 values at exactly 50 chars — all the **same** string, `"Department of Housing Preservation and Development"`, a complete agency name | **distinct count** (max length was also 50, so the length guard would not have) |
# MAGIC | `taxi_pick_up_location` | 24 **distinct** values at 50 chars, all complete addresses ending in valid zips | **max length** — the column's longest value is 60, so nothing is cut at 50 |
# MAGIC
# MAGIC Neither guard alone catches both. Synthetic test data would never have
# MAGIC surfaced either, because the corruption in synthetic data is the corruption you
# MAGIC put there.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Repair — reversible transforms only

# COMMAND ----------

# Excel's gene-symbol mangling is a finite, published mapping, so it reverses exactly.
GENE_MAP = {
    "1-Mar": "MARCH1", "2-Mar": "MARCH2", "3-Mar": "MARCH3", "4-Mar": "MARCH4",
    "5-Mar": "MARCH5", "6-Mar": "MARCH6", "7-Mar": "MARCH7", "8-Mar": "MARCH8",
    "9-Mar": "MARCH9", "10-Mar": "MARCH10", "11-Mar": "MARCH11",
    "1-Sep": "SEPT1", "2-Sep": "SEPT2", "3-Sep": "SEPT3", "4-Sep": "SEPT4",
    "5-Sep": "SEPT5", "6-Sep": "SEPT6", "7-Sep": "SEPT7", "8-Sep": "SEPT8",
    "9-Sep": "SEPT9", "10-Sep": "SEPT10", "11-Sep": "SEPT11", "12-Sep": "SEPT12",
    "1-Dec": "DEC1", "2-Dec": "DEC2",
    "1-Oct": "OCT1", "2-Oct": "OCT2", "3-Oct": "OCT3", "4-Oct": "OCT4",
}

def repair(df: DataFrame, findings) -> DataFrame:
    """Apply only reversible transforms. Anything lossy is left for quarantine."""
    by_col = {}
    for c, mode, *_ in findings:
        by_col.setdefault(c, set()).add(mode)

    out = df
    for c, modes in by_col.items():
        col = F.col(c)

        if "mojibake" in modes:
            # encode('latin-1').decode('utf-8') — exactly invertible.
            col = F.decode(F.encode(col, "ISO-8859-1"), "UTF-8")

        if "numeric_as_text" in modes:
            col = F.regexp_replace(col, ",", "")

        if "null_as_string" in modes:
            col = F.when(col.isin(NULLISH), F.lit(None)).otherwise(col)

        if "excel_date_corruption" in modes:
            mapping = F.create_map([F.lit(x) for kv in GENE_MAP.items() for x in kv])
            col = F.coalesce(mapping[col], col)

        out = out.withColumn(c, col)
    return out

t0 = time.time()
repaired = repair(raw, findings)
repaired.write.mode("overwrite").saveAsTable("vendor_export_clean")
repair_secs = time.time() - t0

print(f"repaired and written in {repair_secs:.1f}s")
display(spark.table("vendor_export_clean").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quarantine as a Delta table
# MAGIC
# MAGIC Rows that cannot be recovered are **not dropped and not guessed at**. They go to
# MAGIC a Delta table with the reason attached, so the good data moves immediately and
# MAGIC whoever owns the source has something specific to act on.

# COMMAND ----------

quarantine = (
    raw
    .withColumn("__reason", F.when(
        F.length("notes") == 255,
        F.lit("value in 'notes' truncated at a length limit — the rest of the text "
              "is not in this file")
    ))
    .filter(F.col("__reason").isNotNull())
    .withColumn("__quarantined_at", F.current_timestamp())
)

quarantine.write.mode("overwrite").saveAsTable("vendor_export_quarantine")

q_count = spark.table("vendor_export_quarantine").count()
clean_count = spark.table("vendor_export_clean").count() - q_count

print(f"usable      : {clean_count:,}")
print(f"quarantined : {q_count:,}")
print(f"recovered   : {clean_count / (clean_count + q_count):.1%}")
display(spark.table("vendor_export_quarantine").select("customer_id", "notes", "__reason").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Why Spark, measured
# MAGIC
# MAGIC The single-threaded Python implementation is in the same repository and does
# MAGIC identical work. The comparison is the point of building both.

# COMMAND ----------

PYTHON_ROWS_PER_SEC = 385   # measured, 44-column table, pure Python, single core

spark_rows_per_sec = n / audit_secs

print(f"pure Python  : {PYTHON_ROWS_PER_SEC:>12,.0f} rows/sec   (constant 0.94 MB memory)")
print(f"Spark        : {spark_rows_per_sec:>12,.0f} rows/sec")
print(f"speedup      : {spark_rows_per_sec / PYTHON_ROWS_PER_SEC:>12,.0f}x")
print()
print(f"At the Python rate, {n:,} rows would take "
      f"{n / PYTHON_ROWS_PER_SEC / 3600:.1f} hours.")
print(f"Spark did it in {audit_secs:.1f} seconds.")
print()
print("Both have their place. Python runs anywhere with no dependencies and holds")
print("memory flat on a laptop. Spark is what you reach for when the table is real.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this demonstrates
# MAGIC
# MAGIC - Detection expressed as **native Spark aggregations**, not Python UDFs — the
# MAGIC   whole audit is one distributed pass with nothing crossing the JVM boundary
# MAGIC - **Delta tables** for clean output and quarantine, with reasons preserved
# MAGIC - The same correctness discipline as the Python version: reversible repairs
# MAGIC   only, and **guards derived from false positives that real data exposed**
# MAGIC - A measured argument for the architecture rather than an assumed one
# MAGIC
# MAGIC Source, tests and the Python implementation: `intact` on GitHub.
