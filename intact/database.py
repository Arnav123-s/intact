"""
Database — audit what is already in a table, not just what arrives in a file.

Corruption does not start when data reaches a CSV. It is usually already in the
warehouse. An import ran with the wrong encoding two years ago. A column got widened
after it had already been truncating. A spreadsheet round-trip turned identifiers
into dates before anyone loaded it. By the time it exports, the damage is old.

This reads from any database Python can already talk to, with no new dependencies.

Zero dependencies, because DB-API is a standard
------------------------------------------------
Every Python database driver implements PEP 249: sqlite3, psycopg, mysqlclient,
pyodbc, snowflake, duckdb. That means a connection always has `.cursor()`, and a
cursor always has `.execute()`, `.fetchmany()` and `.description`. So this takes a
connection you already made and never needs to know which database it is:

    import psycopg
    conn = psycopg.connect(...)
    print(audit_table(conn, "public.customers"))

`sqlite3` is standard library, so the sqlite path works out of the box, and that is
the path the tests exercise.

Everything is read as text, on purpose
---------------------------------------
Values become strings before the detectors see them. That sounds lossy and it is the
point. The detectors look for damage that survives *as text*: a name mangled on
import, an identifier that became a date, a number carrying a thousands separator. A
driver handing back a clean Python `int` has already hidden whether the column is
text or numeric, and that is exactly the question. Reading as text keeps the evidence
visible.

What this deliberately does not do
-----------------------------------
No connection strings, no credential handling, no driver installation. You make the
connection, this borrows it. Anything else would mean this module holds database
passwords, and a data-quality library has no business doing that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Protocol, Sequence

from .core import AuditResult, Finding, Severity, summarise
from .detectors import consistency, tabular

# Rows pulled per round trip. Big enough that the network cost amortises, small
# enough that a wide table does not arrive all at once.
FETCH_SIZE = 10_000

# Default cap on rows examined per table. Corruption is a property of a column, and a
# hundred thousand rows pins down a rate just as well as ten million, at a fraction of
# the load on a database someone else is using.
DEFAULT_LIMIT = 100_000

# Identifiers are quoted, never interpolated blind. This is the allowlist for what a
# table name may contain before it is quoted.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Connection(Protocol):
    """Any PEP 249 connection. Nothing beyond the standard is used."""

    def cursor(self) -> Any: ...


@dataclass
class TableAudit:
    """Findings for one table."""

    table: str
    rows_read: int
    columns: list[str] = field(default_factory=list)
    results: list[AuditResult] = field(default_factory=list)
    truncated: bool = False        # did the row limit stop us early

    @property
    def severity(self) -> Severity:
        worst = Severity.CLEAN
        for r in self.results:
            if r.severity.rank > worst.rank:
                worst = r.severity
        return worst

    def __str__(self) -> str:
        head = [
            f"{self.table}: {self.rows_read:,} rows, {len(self.columns)} columns"
            + ("  (row limit reached)" if self.truncated else ""),
            f"verdict: {self.severity.value}",
            "",
        ]
        flagged = [r for r in self.results if r.findings]
        if not flagged:
            head.append("  nothing found")
        for r in flagged:
            head.append(str(r))
        return "\n".join(head)


def _quote_ident(name: str) -> str:
    """Quote a possibly-qualified identifier, rejecting anything unusual.

    Table names come from callers and end up in SQL, so they get checked against an
    allowlist and quoted part by part rather than interpolated. A name that does not
    match is refused, not escaped and hoped for.

    Double quotes are the SQL standard and work on Postgres, SQLite, DuckDB, Oracle
    and Snowflake. MySQL in its default mode wants backticks, so pass a pre-quoted
    name or set ANSI_QUOTES.
    """
    parts = name.split(".")
    for p in parts:
        if not _SAFE_IDENT.match(p):
            raise ValueError(
                f"refusing to build SQL from {name!r}: identifier parts must be "
                f"letters, digits and underscores. Pass a query instead if you need "
                f"something more complex."
            )
    return ".".join(f'"{p}"' for p in parts)


def read_rows(
    conn: Connection,
    query: str,
    params: Sequence[Any] | None = None,
    limit: int | None = DEFAULT_LIMIT,
) -> tuple[list[str], list[list[str]]]:
    """Run a query and return (column names, rows as text).

    Fetched in batches rather than all at once, so a large result does not have to
    fit in memory twice. Values become strings because that is what the detectors
    need to see. The module docstring explains why.
    """
    cur = conn.cursor()
    cur.execute(query, params or ())

    header = [d[0] for d in (cur.description or [])]
    rows: list[list[str]] = []

    while True:
        batch = cur.fetchmany(FETCH_SIZE)
        if not batch:
            break
        for r in batch:
            rows.append(["" if v is None else str(v) for v in r])
        if limit is not None and len(rows) >= limit:
            rows = rows[:limit]
            break

    try:
        cur.close()
    except Exception:  # pragma: no cover - some drivers close implicitly
        pass

    return header, rows


def audit_query(
    conn: Connection,
    query: str,
    params: Sequence[Any] | None = None,
    name: str = "<query>",
    limit: int | None = DEFAULT_LIMIT,
) -> TableAudit:
    """Audit the result of any SQL you write.

    Use it for joins, filters, or one suspect column. Anything the database can
    return, this can check.
    """
    header, rows = read_rows(conn, query, params, limit)
    if not header:
        return TableAudit(table=name, rows_read=0)

    table_rows = [header] + rows
    results = tabular.audit_rows(table_rows)
    for c in consistency.audit_rows(table_rows):
        if c.findings:
            results.append(c)

    return TableAudit(
        table=name,
        rows_read=len(rows),
        columns=header,
        results=results,
        truncated=(limit is not None and len(rows) >= limit),
    )


def audit_table(
    conn: Connection,
    table: str,
    limit: int | None = DEFAULT_LIMIT,
    where: str | None = None,
) -> TableAudit:
    """Audit one table by name.

        audit_table(conn, "customers")
        audit_table(conn, "sales.orders", limit=50_000)

    `where` gets appended verbatim, so it must not contain untrusted input. When in
    doubt use `audit_query`, which makes the SQL yours and the responsibility too.
    """
    ident = _quote_ident(table)
    sql = f"SELECT * FROM {ident}"
    if where:
        sql += f" WHERE {where}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return audit_query(conn, sql, name=table, limit=limit)


def list_tables(conn: Connection) -> list[str]:
    """Best-effort table listing across the common databases.

    There is no standard way to do this, because every database exposes its catalogue
    differently. So this tries the portable options in turn and returns an empty list
    rather than raising if none of them work. If you know your database, pass table
    names directly instead.
    """
    attempts = (
        # ANSI information_schema: Postgres, MySQL, SQL Server, Snowflake, DuckDB
        ("SELECT table_schema || '.' || table_name FROM information_schema.tables "
         "WHERE table_type = 'BASE TABLE' AND table_schema NOT IN "
         "('information_schema', 'pg_catalog', 'sys')", ()),
        # SQLite
        ("SELECT name FROM sqlite_master WHERE type = 'table' "
         "AND name NOT LIKE 'sqlite_%'", ()),
    )
    for sql, params in attempts:
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            names = [str(r[0]) for r in cur.fetchall()]
            cur.close()
            if names:
                return sorted(names)
        except Exception:
            continue
    return []


def audit_database(
    conn: Connection,
    tables: Iterable[str] | None = None,
    limit: int | None = DEFAULT_LIMIT,
) -> list[TableAudit]:
    """Audit every table, or a named subset.

    Tables get read one at a time, not concurrently. This is usually running against
    a database somebody else depends on, and being slow beats being the reason their
    queries got slow.
    """
    names = list(tables) if tables is not None else list_tables(conn)
    return [audit_table(conn, t, limit=limit) for t in names]


def report(audits: Sequence[TableAudit], show_clean: bool = False) -> str:
    """Worst tables first, because that is the order attention should go in."""
    if not audits:
        return "no tables audited"

    flagged = [a for a in audits if any(r.findings for r in a.results)]
    lines = [
        f"tables   : {len(audits)}",
        f"flagged  : {len(flagged)}",
        f"verdict  : {max((a.severity for a in audits), key=lambda s: s.rank).value}",
        "",
    ]

    for a in sorted(audits, key=lambda x: -x.severity.rank):
        if not any(r.findings for r in a.results) and not show_clean:
            continue
        lines.append(str(a))
        lines.append("")

    if not flagged:
        lines.append("Nothing found in any table.")
    return "\n".join(lines).rstrip()
