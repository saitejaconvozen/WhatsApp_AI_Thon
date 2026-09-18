"""One-shot assessment of the validation-selected candidate on the fixed holdout."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .data import Store, record_content
from .improve import fixed_split, metrics
from .model import Baseline
from .requested import conditional_predictions


def run(store):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(6)
    baseline = Baseline(store)
    if not baseline.status().get("trained") or baseline.status().get("stale"):
        raise ValueError("A current baseline and persisted holdout are required.")
    directory = store.directory / "improvement"
    path = directory / "locked-assessment.json"
    if path.exists():
        raise ValueError("The fixed historical holdout has already been assessed. Do not rescore it for selection.")
    _, validation, test, manifest = fixed_split(baseline.bundle, directory)
    protocol = json.loads((directory / "requested-blend.json").read_text())
    if not protocol["complete"] or protocol["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("A complete validation-selected protocol is required.")
    selected = protocol["selected"]
    text_artifact = joblib.load(directory / "requested-calibrated.joblib")
    if text_artifact["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Calibrated text model was trained on another dataset.")
    text = text_artifact["model"]
    text_probability = text.predict_proba(test)[:, list(text.classes_).index("UTILITY")]
    model = SentenceTransformer(str(directory / "encoder-selected"), local_files_only=True, device="cpu")
    model.max_seq_length = 256
    vectors = model.encode([record_content(r) for r in test], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False)
    head = torch.nn.Linear(vectors.shape[1], 1)
    head.load_state_dict(torch.load(directory / "neural-head.pt", map_location="cpu", weights_only=True))
    head.eval()
    with torch.no_grad():
        neural_probability = torch.sigmoid(head(torch.tensor(vectors)).flatten()).numpy()
    blended = (1 - selected["neural_weight"]) * text_probability + selected["neural_weight"] * neural_probability
    candidate_prediction = conditional_predictions(test, np.where(blended >= selected["threshold"],
                                                                    "UTILITY", "MARKETING"))
    text_prediction = conditional_predictions(test, np.where(text_probability >= .5,
                                                               "UTILITY", "MARKETING"))
    baseline_rows = {r["id"]: r for r in baseline.bundle["holdout"]}
    if set(baseline_rows) != {r["id"] for r in test}:
        raise ValueError("Test records do not match the persisted baseline holdout.")
    baseline_prediction = [baseline_rows[r["id"]]["prediction"] for r in test]
    report = {"fingerprint": manifest["fingerprint"], "validation_selected": selected,
              "train": protocol["train"], "validation": len(validation), "test": len(test),
              "test_evaluated": True,
              "candidate": metrics(test, candidate_prediction),
              "conditional_text_only": metrics(test, text_prediction),
              "baseline_same_test": metrics(test, baseline_prediction),
              "limitations": [
                  "This historical test was inspected in earlier project work; it is not an untouched external test.",
                  "The candidate was trained on the fixed 2,106 training families; the baseline's original evaluation model used 2,808.",
                  "Requested category must be known at prediction time for this conditional model.",
                  "No serving model was replaced by this assessment.",
              ]}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    run(Store(args.data_dir))


if __name__ == "__main__":
    main()
