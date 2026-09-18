"""Find the templates most like this one that Meta has already ruled on.

Training deduplicates near-identical templates, which is right: a model that
sees the same wording forty times learns that wording rather than the rule. For
retrieval the same collapse is wrong. Showing a rewriter a real approved
template that resembles the one in front of it is more useful the more approved
templates there are to draw on, duplicates included -- the pool goes from 1,036
to 3,116, and the share of downgraded templates with a close approved example
rises at every similarity level.

Both sides are returned. An approved example alone says what to aim at; an
approved and a downgraded example worded similarly say where the line falls
between them, which is the thing a rewrite has to act on.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Store, candidate, normalized_content


class Examples:
    """Nearest labelled precedents, drawn from the undeduplicated corpus."""

    def __init__(self, store, requested="UTILITY"):
        rows = [r for r in store.records()
                if candidate(r) and r.get("requested_category") == requested
                and (r.get("body") or "").strip()]
        self.approved = [r for r in rows if r.get("meta_category") == "UTILITY"]
        self.downgraded = [r for r in rows if r.get("meta_category") == "MARKETING"]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          max_features=40000)
        corpus = self.approved + self.downgraded
        matrix = self.vectorizer.fit_transform([normalized_content(r["body"]) for r in corpus])
        self.approved_matrix = matrix[:len(self.approved)]
        self.downgraded_matrix = matrix[len(self.approved):]

    def _near(self, body, pool, matrix, count, exclude):
        if not pool:
            return []
        scores = (self.vectorizer.transform([normalized_content(body)]) @ matrix.T).toarray()[0]
        out = []
        for index in np.argsort(scores)[::-1]:
            record = pool[int(index)]
            # A template must never be shown its own label as a precedent.
            if exclude and (record.get("id") == exclude or record.get("body") == body):
                continue
            out.append((record, float(scores[index])))
            if len(out) >= count:
                break
        return out

    def for_template(self, body, approved=2, downgraded=1, exclude=None, floor=0.0):
        """Nearest approved examples, plus the nearest downgraded one for contrast."""
        good = [(r, s) for r, s in self._near(body, self.approved, self.approved_matrix,
                                              approved, exclude) if s >= floor]
        bad = [(r, s) for r, s in self._near(body, self.downgraded, self.downgraded_matrix,
                                             downgraded, exclude) if s >= floor]
        return {"approved": good, "downgraded": bad}

    @staticmethod
    def render(found):
        """The examples as prompt text, or an empty string when nothing is close."""
        blocks = []
        for record, score in found.get("approved", []):
            blocks.append(f"--- Meta approved this as UTILITY (wording similarity {score:.2f}) ---\n"
                          + (record.get("body") or "").strip())
        for record, score in found.get("downgraded", []):
            blocks.append(f"--- Meta downgraded this to MARKETING (wording similarity {score:.2f}) ---\n"
                          + (record.get("body") or "").strip())
        if not blocks:
            return ""
        contrast = ("\nWhere an approved and a downgraded example are worded similarly, the "
                    "difference between them is what decides the category.\n"
                    if found.get("approved") and found.get("downgraded") else "\n")
        return ("Templates Meta has already ruled on, closest in wording to this one:\n\n"
                + "\n\n".join(blocks) + "\n" + contrast)
