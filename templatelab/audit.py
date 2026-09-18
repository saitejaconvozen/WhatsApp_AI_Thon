"""Export held-out errors without using the refitted serving model."""

import argparse
import json
from pathlib import Path

from .data import Store
from .model import Baseline


def error_report(baseline):
    status = baseline.status()
    if not status["trained"] or status.get("stale"):
        raise ValueError("Train a baseline on the current dataset before exporting errors.")
    bundle = baseline.bundle
    if "holdout" not in bundle:
        raise ValueError("Retrain once to capture holdout predictions for this older model.")
    errors = [row for row in bundle["holdout"] if row["recorded_category"] != row["prediction"]]
    return {
        "trained_at": bundle["report"]["trained_at"],
        "dataset_revision": bundle["revision"],
        "test_families": len(bundle["holdout"]),
        "false_utility": [row for row in errors if row["prediction"] == "UTILITY"],
        "false_marketing": [row for row in errors if row["prediction"] == "MARKETING"],
        "limitations": [
            "Errors are disagreements with recorded Meta categories; their provenance was confirmed by the platform owner on 2026-09-18, not independently checked through Meta's API.",
            "Predictions were made on the holdout before refitting the serving model.",
            "This export contains template content and may contain personal information. Keep it local.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    parser.add_argument("--retrain", action="store_true", help="Retrain and capture holdout predictions first.")
    args = parser.parse_args()
    baseline = Baseline(Store(args.data_dir))
    try:
        if args.retrain:
            baseline.train()
        report = error_report(baseline)
    except ValueError as exc:
        parser.error(str(exc))
    path = args.data_dir / "holdout-errors.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "test_families": report["test_families"],
                      "false_utility": len(report["false_utility"]),
                      "false_marketing": len(report["false_marketing"])}, indent=2))


if __name__ == "__main__":
    main()
