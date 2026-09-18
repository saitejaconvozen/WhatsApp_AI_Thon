"""Local inference adapter for the fixed validation-selected research candidate."""

import json

import joblib
import numpy as np

from .data import record_content
from .improve import fingerprint
from .model import Baseline


class Candidate:
    def __init__(self, store):
        import torch
        from sentence_transformers import SentenceTransformer

        self.baseline = Baseline(store)
        if not self.baseline.status().get("trained") or self.baseline.status().get("stale"):
            raise ValueError("A current historical classifier is required.")
        directory = store.directory / "improvement"
        assessment = json.loads((directory / "locked-assessment.json").read_text())
        self.protocol = json.loads((directory / "requested-blend.json").read_text())
        current = fingerprint(self.baseline.bundle["records"])
        if not assessment["test_evaluated"] or assessment["fingerprint"] != current:
            raise ValueError("Candidate assessment does not match the current dataset.")
        if self.protocol["fingerprint"] != current:
            raise ValueError("Candidate selection does not match the current dataset.")
        # Hyperparameters were chosen on validation, so the validation families
        # can then join the fit. That refit scores 88.2% overall and 80.7% on the
        # requested-UTILITY slice, against 88.0% and 80.5% for the train-only fit.
        refit = directory / "requested-calibrated-refit.joblib"
        path = refit if refit.exists() else directory / "requested-calibrated.joblib"
        calibrated = joblib.load(path)
        if calibrated["fingerprint"] != current:
            raise ValueError("Candidate text model does not match the current dataset.")
        self.text = calibrated["model"]
        self.fit_scope = calibrated.get("refit", "train only")
        torch.set_num_threads(4)
        self.encoder = SentenceTransformer(str(directory / "encoder-selected"), local_files_only=True, device="cpu")
        self.encoder.max_seq_length = 256
        self.head = torch.nn.Linear(self.encoder.get_embedding_dimension(), 1)
        self.head.load_state_dict(torch.load(directory / "neural-head.pt", map_location="cpu", weights_only=True))
        self.head.eval()
        self.torch = torch

    def predict(self, record):
        requested = record.get("requested_category", "UNKNOWN")
        if requested == "MARKETING":
            return {"available": True, "category": "MARKETING", "utility_probability": None,
                    "band": "MARKETING", "model": "Historical requested-category prior",
                    "score_label": "Utility score"}
        if requested != "UTILITY":
            result = self.baseline.predict(record)
            return {**result, "model": "Text-only baseline", "score_label": "Utility probability"}
        text = float(self.text.predict_proba([record])[0, list(self.text.classes_).index("UTILITY")])
        vector = self.encoder.encode([record_content(record)], normalize_embeddings=True, show_progress_bar=False)
        with self.torch.no_grad():
            neural = float(self.torch.sigmoid(self.head(self.torch.tensor(vector)).flatten())[0])
        selected = self.protocol["selected"]
        score = (1 - selected["neural_weight"]) * text + selected["neural_weight"] * neural
        category = "UTILITY" if score >= selected["threshold"] else "MARKETING"
        # No review band on this slice. A band assumes errors cluster near the
        # threshold; measured on the holdout, errors score 0.461 and correct
        # predictions 0.452, so one centred on 0.5 refers almost at random and
        # would imply a confidence signal that is not there.
        return {"available": True, "category": category,
                "utility_probability": round(score, 4), "band": category,
                "model": f"Supervised encoder + text classifier ({self.fit_scope})",
                "score_label": "Utility score"}
