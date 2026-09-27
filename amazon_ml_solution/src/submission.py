"""
submission.py — Versioned submission management.

Creates submission_XX/ snapshots with all artifacts, config, and local metrics.
Maintains a leaderboard log for tracking experiment history.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import EXPERIMENTS_DIR, OUTPUT_DIR, SUBMISSIONS_DIR

log = logging.getLogger(__name__)


def save_submission_snapshot(
    submission_name: str,
    local_metrics: dict,
    threshold: float,
    notes: str = "",
) -> Path:
    """
    Save a versioned snapshot of the current submission.
    
    Copies:
      - matching_results.tsv
      - candidate_pairs.tsv
      - local_metrics.json
      - notes.md
    """
    snap_dir = SUBMISSIONS_DIR / submission_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Copy output files
    for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
        src = OUTPUT_DIR / fname
        if src.exists():
            shutil.copy2(src, snap_dir / fname)

    # Save metrics
    metrics_payload = {
        "submission_name": submission_name,
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "local_metrics": local_metrics,
    }
    with open(snap_dir / "local_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save notes
    with open(snap_dir / "notes.md", "w") as f:
        f.write(f"# {submission_name}\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Threshold:** {threshold:.4f}\n\n")
        f.write(f"**Local F_0.5:** {local_metrics.get('macro_f05', 'N/A')}\n\n")
        f.write(f"**Notes:** {notes}\n")

    log.info("Submission snapshot saved → %s", snap_dir)

    # Update leaderboard log
    _update_leaderboard_log(submission_name, threshold, local_metrics, notes)

    return snap_dir


def _update_leaderboard_log(
    submission_name: str,
    threshold: float,
    local_metrics: dict,
    notes: str,
) -> None:
    """Append a row to the leaderboard CSV log."""
    log_path = SUBMISSIONS_DIR / "leaderboard_log.csv"
    
    new_row = {
        "submission_id": submission_name,
        "timestamp": datetime.now().isoformat(),
        "local_f05": local_metrics.get("macro_f05", ""),
        "local_precision": local_metrics.get("mean_precision", ""),
        "local_recall": local_metrics.get("mean_recall", ""),
        "threshold": threshold,
        "public_score": "",   # Fill in manually after uploading to portal
        "notes": notes,
    }

    if log_path.exists():
        df = pd.read_csv(log_path)
    else:
        df = pd.DataFrame(columns=list(new_row.keys()))

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(log_path, index=False)
    log.info("Leaderboard log updated → %s", log_path)


def list_submissions() -> pd.DataFrame:
    """List all submission snapshots with their local metrics."""
    log_path = SUBMISSIONS_DIR / "leaderboard_log.csv"
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame()
