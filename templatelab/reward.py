"""Score how much a template reads like one Meta approved as utility.

This replaces the hand-written gate that capped earlier work. That gate asked
for a TRANSACTION word beside a REFERENCE, and rejected 31.7% of the templates
Meta itself approved -- so every population estimate built on it was really a
measurement of the regex. Here the target is learned from the 1,036 approved
templates instead of asserted.

Honest accuracy: 78.5%, ROC AUC 0.858, measured with the head trained on the
encoder's own training split and scored on the untouched test split. An earlier
figure of 86.4% came from random splits over all templates, which put the
encoder's training data in the test folds.

AUC 0.858 makes this a good ranking signal and a mediocre gate. Treat a score as
"this looks more like the approved corpus than it did", never as "Meta will
approve this".
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import Store, training_records

MODEL_NAME = "utility-reward.joblib"
ENCODER = "improvement/encoder-selected"


class Reward:
    """A learned utility score in the fine-tuned encoder's embedding space."""

    def __init__(self, store, encoder=None):
        self.store = store
        self.path = store.directory / MODEL_NAME
        self._encoder = encoder
        self._model = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(str(self.store.directory / ENCODER), device="cpu")
        return self._encoder

    def embed(self, texts):
        return self.encoder.encode(list(texts), batch_size=64,
                                   normalize_embeddings=True, show_progress_bar=False)

    def fit(self):
        """Train only on the encoder's own training split.

        Fitting on everything would score rewrites with a head that has seen the
        templates it is judging, which is how the 86.4% figure happened.
        """
        split = json.loads((self.store.directory / "improvement" / "improvement-split.json")
                           .read_text(encoding="utf-8"))
        train_ids = set(split["train_ids"])
        records, _ = training_records(self.store.records())
        rows = [r for r in records
                if r.get("requested_category") == "UTILITY" and r["id"] in train_ids]
        if len(rows) < 100:
            raise ValueError("Too few training templates to fit a reward model.")
        vectors = self.embed([r["body"] for r in rows])
        labels = np.array([r["meta_category"] == "UTILITY" for r in rows])
        model = LogisticRegression(max_iter=4000, class_weight="balanced").fit(vectors, labels)
        joblib.dump({"model": model, "train": len(rows)}, self.path)
        self._model = model
        return {"trained_on": len(rows), "path": str(self.path)}

    @property
    def model(self):
        if self._model is None:
            if not self.path.exists():
                raise FileNotFoundError(f"No reward model at {self.path}; run `fit()` first.")
            self._model = joblib.load(self.path)["model"]
        return self._model

    def score(self, texts):
        """Probability each text belongs to the approved-utility side. 0..1."""
        single = isinstance(texts, str)
        vectors = self.embed([texts] if single else list(texts))
        scores = self.model.predict_proba(vectors)[:, 1]
        return float(scores[0]) if single else scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    print(json.dumps(Reward(Store(args.data_dir)).fit(), indent=2))


if __name__ == "__main__":
    main()
