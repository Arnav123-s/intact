"""intact — hand it a broken data file, get back usable data."""

from .solve import solve, solve_to_files, Solution
from .core import AuditResult, Finding, Severity, plural, report, summarise
from .pipeline import Pipeline
from .repair import Recoverability, RepairResult, repair_column
from .evolution import propose_rules, evolution_report
from .profiles import (
    apply_profile, compare_profiles, custom,
    ANALYTICS, ARCHIVE, CLASSIFICATION, JOINS, RAW, SCIENTIFIC, SEARCH_INDEX,
)
from .annotate import annotate, review_sheet, write_csv, write_jsonl

__version__ = "0.1.0"

__all__ = [
    # the main entry point
    "solve", "solve_to_files", "Solution",
    # inspection
    "AuditResult", "Finding", "Severity", "report", "summarise", "plural",
    # scoring for a downstream use
    "apply_profile", "compare_profiles", "custom",
    "ANALYTICS", "ARCHIVE", "CLASSIFICATION", "JOINS", "RAW",
    "SCIENTIFIC", "SEARCH_INDEX",
    # repair and provenance
    "Recoverability", "RepairResult", "repair_column",
    "annotate", "review_sheet", "write_csv", "write_jsonl",
    # learning
    "Pipeline", "propose_rules", "evolution_report",
]
