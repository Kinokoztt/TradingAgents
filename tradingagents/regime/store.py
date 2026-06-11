"""Local JSON persistence for regime-gate reports.

Mirrors the concept_graph store layout: ``{out_dir}/{as_of_date}/regime_report.json``.
Written with ensure_ascii=False so Chinese/Unicode rationales stay readable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import Scorecard
from .schemas import RegimeReport

DEFAULT_OUT_DIR = "regime_gate_output"
REPORT_FILE = "regime_report.json"
SCORECARD_FILE = "scorecard.json"


def save_report(as_of_date: str, report: RegimeReport, out_dir: str = DEFAULT_OUT_DIR) -> str:
    """Write the report to ``{out_dir}/{as_of_date}/regime_report.json``. Returns the path."""
    day_dir = Path(out_dir) / as_of_date
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / REPORT_FILE
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_report(as_of_date: str, out_dir: str = DEFAULT_OUT_DIR) -> RegimeReport:
    path = Path(out_dir) / as_of_date / REPORT_FILE
    return RegimeReport.model_validate_json(path.read_text(encoding="utf-8"))


def save_scorecard(session: str, scorecard: Scorecard, out_dir: str = DEFAULT_OUT_DIR) -> str:
    """Write the scorecard alongside its report at ``{out_dir}/{session}/scorecard.json``."""
    day_dir = Path(out_dir) / session
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / SCORECARD_FILE
    path.write_text(
        json.dumps(scorecard.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_scorecard(session: str, out_dir: str = DEFAULT_OUT_DIR) -> Scorecard:
    path = Path(out_dir) / session / SCORECARD_FILE
    return Scorecard.model_validate_json(path.read_text(encoding="utf-8"))
