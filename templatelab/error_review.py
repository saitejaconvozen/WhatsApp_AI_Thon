"""Human annotations are separate from immutable dataset labels and predictions."""

import hashlib
import json
from datetime import datetime, timezone

from .audit import error_report


class ErrorReviews:
    def __init__(self, store, baseline):
        self.store, self.baseline = store, baseline
        with store.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS error_reviews (
                snapshot TEXT NOT NULL, record_id TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY(snapshot, record_id))""")

    def current(self):
        report = error_report(self.baseline)
        snapshot = hashlib.sha256(
            (report["dataset_revision"] + report["trained_at"]).encode()).hexdigest()[:24]
        with self.store.connection() as db:
            saved = {row[0]: json.loads(row[1]) for row in db.execute(
                "SELECT record_id, payload FROM error_reviews WHERE snapshot = ?", (snapshot,))}
        items = []
        for direction in ("false_utility", "false_marketing"):
            for row in report[direction]:
                items.append({**row, "direction": direction, "annotation": saved.get(row["id"], {
                    "status": "pending", "reason": "unknown", "notes": "", "context": "",
                    "purpose": "unknown", "relationship_confirmed": False,
                    "draft": None, "meta_outcome": "not_submitted", "outcome_reference": "",
                    "version": 0,
                })})
        return {"snapshot": snapshot, "trained_at": report["trained_at"],
                "dataset_revision": report["dataset_revision"], "items": items,
                "counts": {"total": len(items),
                           "false_utility": len(report["false_utility"]),
                           "false_marketing": len(report["false_marketing"]),
                           "reviewed": sum(r["annotation"]["status"] == "reviewed" for r in items)},
                "limitations": report["limitations"]}

    def save(self, record_id, payload):
        current = self.current()
        if payload["snapshot"] != current["snapshot"]:
            raise ValueError("The evaluation snapshot changed. Reload before saving.")
        item = next((row for row in current["items"] if row["id"] == record_id), None)
        if item is None:
            raise KeyError("This record is not an error in the current holdout.")
        if payload["meta_outcome"] != "not_submitted" and (
                not payload["draft"] or not payload["draft"]["body"].strip()
                or not payload["outcome_reference"].strip()):
            raise ValueError("An observed Meta outcome requires a draft and a decision reference.")
        if payload["status"] == "reviewed" and (payload["reason"] == "unknown" or not payload["notes"].strip()):
            raise ValueError("A completed review requires a reason and review notes.")
        annotation = {key: value for key, value in payload.items() if key != "snapshot"}
        annotation["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM error_reviews WHERE snapshot = ? AND record_id = ?",
                             (current["snapshot"], record_id)).fetchone()
            version = json.loads(row[0])["version"] if row else 0
            if payload["version"] != version:
                raise ValueError("Another review was saved. Reload before saving your changes.")
            previous = json.loads(row[0]) if row else None
            if (previous and previous.get("meta_outcome") != "not_submitted"
                    and payload["meta_outcome"] != "not_submitted"
                    and previous.get("draft") != payload["draft"]
                    and previous.get("outcome_reference") == payload["outcome_reference"]):
                raise ValueError("An edited draft needs a new Meta decision reference or an unrecorded outcome.")
            annotation["version"] = version + 1
            db.execute("INSERT OR REPLACE INTO error_reviews VALUES (?, ?, ?)",
                       (current["snapshot"], record_id, json.dumps(annotation)))
        return annotation
