"""
validate_and_submit.py — One-click validation + submission packaging.

Usage:
    python validate_and_submit.py --submission-name submission_01

Steps:
  1. Run the official validator script
  2. Print PASS/FAIL with details
  3. Create a versioned snapshot in submissions/
"""

import argparse
import subprocess
import sys
from pathlib import Path

STUDENT_RES = Path(r"D:\Deksto-1\Desktop\Hackthons\Projects\Amazon_ML\6ab10eb3b23ba_student_resource\student_resource")
OUTPUT_DIR  = Path(__file__).parent / "output"
SUBS_DIR    = Path(__file__).parent / "submissions"


def run_official_validator() -> bool:
    """Run the official validate_submission.py and return True if PASS."""
    validator = STUDENT_RES / "utils" / "validate_submission.py"
    matching  = OUTPUT_DIR / "matching_results.tsv"
    candidate = OUTPUT_DIR / "candidate_pairs.tsv"
    test_dir  = STUDENT_RES / "dataset" / "test"

    if not matching.exists():
        print(f"ERROR: {matching} not found. Run pipeline first.")
        return False

    cmd = [
        sys.executable, str(validator),
        "--matching",   str(matching),
        "--candidate",  str(candidate),
        "--test-dir",   str(test_dir),
    ]
    print("Running official validator ...")
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-name", default="submission_01")
    args = parser.parse_args()

    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026 — SUBMISSION VALIDATOR")
    print("=" * 60)

    passed = run_official_validator()

    if passed:
        print("\n✅ PASS — Files are safe to submit.")
        print(f"\nNext step: Upload {OUTPUT_DIR / 'matching_results.tsv'} to the portal.")
    else:
        print("\n❌ FAIL — Fix the errors above before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
