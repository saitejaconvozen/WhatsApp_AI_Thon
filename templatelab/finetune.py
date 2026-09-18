"""CPU supervised encoder adaptation using the locked improvement partitions.

The encoder and logistic head learn only from training labels. Epoch, threshold
and lexical blend selection use validation; no test predictions are made here.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import Store, record_content
from .embedding import ENCODER_NAME
from .improve import fixed_split, metrics, ranking
from .model import Baseline


def select_blend(records, neural, lexical):
    best = None
    for weight in (0., .25, .5, .75, 1.):
        probability = weight * neural + (1 - weight) * lexical
        for threshold in (.35, .4, .45, .5, .55, .6, .65):
            scores = metrics(records, np.where(probability >= threshold, "UTILITY", "MARKETING"))
            row = {"neural_weight": weight, "threshold": threshold, "validation": scores}
            if best is None or ranking(scores) > ranking(best["validation"]):
                best = row
    return best


def run(store, epochs=5, threads=6):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    torch.manual_seed(20260918)
    np.random.seed(20260918)
    random.seed(20260918)
    directory = store.directory / "improvement"
    directory.mkdir(exist_ok=True)
    path = directory / "neural-validation.json"
    if path.exists():
        raise ValueError("A neural experiment already exists; inspect it before starting another run.")
    baseline = Baseline(store)
    if not baseline.status().get("trained") or baseline.status().get("stale"):
        raise ValueError("A current baseline and holdout are required.")
    train, validation, test, manifest = fixed_split(baseline.bundle, directory)
    lexical = joblib.load(directory / "lexical-validation.joblib")
    if lexical["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Lexical model and neural experiment datasets differ.")
    lexical_model = lexical["model"]
    if not hasattr(lexical_model, "predict_proba"):
        raise ValueError("The selected lexical model must expose probabilities for blending.")
    lexical_probability = lexical_model.predict_proba(validation)[:, list(lexical_model.classes_).index("UTILITY")]
    model = SentenceTransformer(ENCODER_NAME, local_files_only=True, device="cpu")
    model.max_seq_length = 256
    train_text = [record_content(r) for r in train]
    validation_text = [record_content(r) for r in validation]
    labels = np.array([r["meta_category"] == "UTILITY" for r in train], dtype=np.float32)
    vectors = model.encode(train_text, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    initial = LogisticRegression(C=3, max_iter=2000, random_state=42).fit(vectors, labels)
    head = torch.nn.Linear(vectors.shape[1], 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(initial.coef_, dtype=torch.float32))
        head.bias.copy_(torch.tensor(initial.intercept_, dtype=torch.float32))
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": 2e-5},
                                  {"params": head.parameters(), "lr": 1e-3}], weight_decay=.01)
    loss_function = torch.nn.BCEWithLogitsLoss()
    report = {"encoder": ENCODER_NAME, "fingerprint": manifest["fingerprint"],
              "train": len(train), "validation": len(validation), "test": len(test),
              "seed": 20260918, "epochs_requested": epochs, "max_seq_length": 256,
              "test_evaluated": False, "complete": False, "epochs": [],
              "limitations": ["Validation-selected results are not test accuracy.",
                              "Encoder inputs are truncated at 256 tokens.",
                              "No requested-category or recorded-category fields are model inputs."]}
    best = None

    def validate(epoch, loss=None):
        nonlocal best
        model.eval()
        head.eval()
        embedded = model.encode(validation_text, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        with torch.no_grad():
            probabilities = torch.sigmoid(head(torch.tensor(embedded)).flatten()).numpy()
        selected = select_blend(validation, probabilities, lexical_probability)
        row = {"epoch": epoch, "train_loss": loss, **selected,
               "neural_only": metrics(validation, np.where(probabilities >= .5, "UTILITY", "MARKETING"))}
        report["epochs"].append(row)
        if best is None or ranking(row["validation"]) > ranking(best["validation"]):
            best = row
            model.save(str(directory / "encoder-selected"))
            torch.save(head.state_dict(), directory / "neural-head.pt")
            np.savez(directory / "validation-probabilities.npz", neural=probabilities, lexical=lexical_probability,
                     ids=np.array([r["id"] for r in validation]))
        report["selected"] = best
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)
        print(json.dumps(row), flush=True)

    validate(0)
    batch_size = 16
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        total_loss = 0.
        order = np.random.permutation(len(train))
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            features = model.tokenize([train_text[i] for i in indices])
            embedded = torch.nn.functional.normalize(model(features)["sentence_embedding"], dim=1)
            logits = head(embedded).flatten()
            loss = loss_function(logits, torch.tensor(labels[indices]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            if offset % (batch_size * 25) == 0:
                print(f"Epoch {epoch}/{epochs}: {min(offset + batch_size, len(train))}/{len(train)} training examples", flush=True)
        validate(epoch, total_loss / len(train))
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10 or not 1 <= args.threads <= 8:
        parser.error("Use 1-10 epochs and 1-8 CPU threads.")
    run(Store(args.data_dir), args.epochs, args.threads)


if __name__ == "__main__":
    main()
