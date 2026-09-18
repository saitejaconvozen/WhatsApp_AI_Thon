"""Draft a utility template from a description of what it needs to say.

Conversion fights a template's existing promotional purpose. Drafting has no
such handicap: nothing has to be removed, and the form is ours to choose. That
matters because approval is strongly form-dependent -- one form in this corpus
holds 86 approvals and no failures, another 27 approvals against 131 failures
(templatelab/forms.py). A drafter that targets the first is working with the
grain of Meta's own decisions.

So a draft here is not just text. It comes with the form it follows and how
often Meta approved that form, which is the closest thing to evidence available
without submitting.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .data import Store
from .llm import BACKENDS, Cache, Unavailable, configuration
from .policy import REFERENCE, TRANSACTION, promotion_findings
from .recast import PLACEHOLDER, parse_rewrite
from .reward import Reward

ROOT = Path(__file__).resolve().parent.parent
# Below this cosine to a form's centroid, the draft is not really an instance of
# that form and the form's historical approval rate says nothing about it. The
# ten forms clearing 60% approval do not cover every task -- there is no support
# -ticket form among them -- so a task outside their span gets forced into the
# nearest one, and quoting "91% approved" for it would be inventing evidence.
FORM_MATCH = 0.40

# Half the downgraded templates carry Indic script while their stored `language`
# says ENGLISH_US, so the metadata cannot be trusted and the script is read from
# the body. A draft that answers a Kannada template in English is not a better
# template; it is one the recipient cannot read.
SCRIPTS = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F), "Gujarati": (0x0A80, 0x0AFF),
    "Odia": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF),
    "Malayalam": (0x0D00, 0x0D7F), "Arabic": (0x0600, 0x06FF),
}


def script_of(text):
    """The dominant non-Latin script in a template, or None."""
    counts = {}
    for character in text or "":
        point = ord(character)
        for name, (low, high) in SCRIPTS.items():
            if low <= point <= high:
                counts[name] = counts.get(name, 0) + 1
                break
    return max(counts, key=counts.get) if counts else None

PROMPT = """Write a WhatsApp Business template that Meta will classify as UTILITY.

Meta approved every one of these. Follow their shape:

{exemplars}

What makes Meta treat a template as utility, measured across this corpus:
- It reports on something the recipient ALREADY has -- an order, invoice,
  booking, appointment, agreement. It presupposes that thing exists.
- It names that specific thing rather than a generic topic.
- It reports what happened, often in the past tense: "has been processed",
  "we received", "was scheduled".
- It does NOT invite the recipient to start something new: no "get", "explore",
  "book now", "schedule now".
- A call to action is fine. Meta approves "Call Now or choose an option below".
  Links and buttons are not the problem.

Write the template for this:

{task}

Rules:
- Use {{{{placeholders}}}} for every value that varies per recipient.
- Include a reference that identifies the specific transaction.
- No discount, offer, or wording that promotes a purchase.
- If this task is inherently promotional and cannot be a service message, say so.
{budget}{language}

Return JSON only:
{{"possible": true|false, "name": "SHORT_TEMPLATE_NAME", "body": "the template",
 "buttons": "optional button label or empty", "reason": "one sentence"}}"""

REFINE = """Your previous draft scored {score:.2f} out of 1.00 against a model trained on
templates Meta approved as utility.

Your previous draft:
{previous}

{critique}

Write it again addressing that. Same rules: placeholders for varying values, a
reference to the specific transaction, nothing promotional.

Return JSON only:
{{"possible": true|false, "name": "SHORT_TEMPLATE_NAME", "body": "the template",
 "buttons": "optional button label or empty", "reason": "one sentence"}}"""


def critique(text, checks=None, budget=None, script=None):
    notes = []
    if script and checks is not None and not checks.get("kept_script", True):
        notes.append(f"It is not written in {script} script. The recipient reads {script}; "
                     f"write the template in that script, leaving placeholders in ASCII.")
    if budget is not None and checks is not None and not checks.get("within_budget", True):
        notes.append(f"It uses {checks['placeholders_used']} placeholders; the limit is {budget}. "
                     f"Remove the ones the description does not support -- do not assert a "
                     f"reference, booking or application the description never mentions.")
    if not TRANSACTION.search(text):
        notes.append("It names no transaction (order, invoice, booking, appointment).")
    if not REFERENCE.search(text):
        notes.append("It has no reference identifying which specific one.")
    if not PLACEHOLDER.search(text):
        notes.append("It contains no placeholders, so it cannot vary per recipient.")
    if promotion_findings({"body": text}):
        notes.append("It contains promotional wording.")
    if re.search(r"\b(?:get|grab|avail|explore)\b|\b(?:book|call|apply|schedule) now\b", text, re.I):
        notes.append("It invites a NEW action; report on an existing one instead.")
    return ("Problems:\n- " + "\n- ".join(notes)) if notes else (
        "No rule fired; the wording is simply further from the approved corpus than it could be.")


def gates(text):
    return {"has_anchor": bool(TRANSACTION.search(text)) and bool(REFERENCE.search(text)),
            "has_placeholder": bool(PLACEHOLDER.search(text)),
            "promotional": bool(promotion_findings({"body": text}))}


class Drafter:
    def __init__(self, store, min_approval=0.60):
        self.store = store
        self.reward = Reward(store)
        self.forms = json.loads((store.directory / "forms.json").read_text(encoding="utf-8"))["forms"]
        # Only forms Meta approves often enough to be worth imitating. Drafting
        # into the feedback form (17%) would be choosing a losing shape on purpose.
        self.safe = [f for f in self.forms if f["approval_rate"] >= min_approval and f["exemplars"]]
        if not self.safe:
            raise ValueError("No form clears the minimum approval rate.")
        self._centroids = None

    @property
    def centroids(self):
        if self._centroids is None:
            self._centroids = np.vstack([
                self.reward.embed(f["exemplars"][:3]).mean(0) for f in self.safe])
            norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
            self._centroids = self._centroids / np.clip(norms, 1e-9, None)
        return self._centroids

    def choose_form(self, task):
        """Pick exemplars to imitate, from a task description.

        Only ever used to choose what the model is shown. A task description and
        a template body sit in different regions of the encoder's space -- these
        similarities run 0.16-0.31 even when the eventual draft clearly belongs
        to a form -- so this choice guides wording and nothing more. What form a
        draft actually belongs to is measured afterwards, on the draft itself.
        """
        vector = self.reward.embed([task])[0]
        similarity = self.centroids @ vector
        rank = similarity * np.array([f["approval_rate"] for f in self.safe])
        index = int(np.argmax(rank))
        return self.safe[index], float(similarity[index])

    def classify_draft(self, body):
        """Which approved form the finished draft belongs to, if any.

        Comparing template to template rather than task to template: the same
        drafts that match a task at 0.20 match their own form at 0.51.
        """
        vector = self.reward.embed([body])[0]
        similarity = self.centroids @ vector
        index = int(np.argmax(similarity))
        return self.safe[index], float(similarity[index])

    def draft(self, task, rounds=3, target=0.60, placeholder_budget=None, script=None):
        """Draft a template for `task`.

        `placeholder_budget` caps how many distinct placeholders the result may
        carry. Without it, drafts invent the fields an ideal utility template
        would have: a viewing-confirmation drafted from "the recipient unlocked
        contact details" grew from 3 placeholders to 8, asserting an appointment
        reference and an enquiry id that the original never claimed existed.
        Every invented placeholder is either a field the business cannot fill or
        a statement that is not true.
        """
        task = str(task or "").strip()
        if not task:
            raise ValueError("Describe the message you need.")
        config = configuration()
        if config["backend"] == "disabled":
            raise Unavailable("No LLM backend configured.")
        cache = Cache(self.store.directory)

        form, task_similarity = self.choose_form(task)
        language = ""
        if script:
            language = (f"\n- Write the template in {script} script, the language the recipient reads."
                        f"\n  Placeholders stay in ASCII exactly as written.")
        budget = ""
        if placeholder_budget is not None:
            budget = (f"- Use at most {placeholder_budget} distinct placeholders. Do not introduce\n"
                      f"  references, identifiers or dates the description does not mention; if the\n"
                      f"  description does not say a booking or application exists, do not assert one.")
        prompt = PROMPT.format(
            exemplars="\n\n".join(f"--- approved example {i+1} ---\n{e}"
                                  for i, e in enumerate(form["exemplars"][:3])),
            task=task, budget=budget, language=language)

        best, trail = None, []
        for _ in range(rounds):
            key = Cache.key(prompt, config["model"])
            answer = cache.get(key)
            if not answer:
                answer = parse_rewrite(BACKENDS[config["backend"]](prompt, config["model"]))
                cache.put(key, answer)
            if not answer.get("possible") or not (answer.get("body") or "").strip():
                if best is None:
                    return {"possible": False, "reason": (answer.get("reason") or "")[:300],
                            "form_guess": form["terms"]}
                break
            body = answer["body"].strip()
            checks = gates(body)
            if placeholder_budget is not None:
                used = len(set(PLACEHOLDER.findall(body)))
                checks["placeholders_used"] = used
                checks["within_budget"] = used <= placeholder_budget
            if script:
                checks["script"] = script_of(body)
                checks["kept_script"] = checks["script"] == script
            score = self.reward.score(body)
            trail.append(round(score, 4))
            clean = (checks["has_anchor"] and checks["has_placeholder"]
                     and not checks["promotional"] and checks.get("within_budget", True)
                     and checks.get("kept_script", True))
            if clean and (best is None or score > best[1]):
                best = (body, score, answer, checks)
            if clean and score >= target:
                break
            prompt = REFINE.format(score=score, previous=body,
                                   critique=critique(body, checks, placeholder_budget, script))

        if best is None:
            return {"possible": False, "reason": "No draft satisfied the hard constraints.",
                    "score_trail": trail, "form_guess": form["terms"]}
        body, score, answer, checks = best
        # Evidence is measured on the finished draft, never on the task text.
        actual, similarity = self.classify_draft(body)
        return {"possible": True, "body": body,
                "name": str(answer.get("name") or "")[:120],
                "buttons": str(answer.get("buttons") or "")[:120],
                "reason": str(answer.get("reason") or "")[:300],
                "score": round(score, 4), "score_trail": trail, "rounds_used": len(trail),
                **self.form_evidence(actual, similarity, similarity >= FORM_MATCH), **checks}

    @staticmethod
    def form_evidence(form, similarity, matched):
        """Report the form's record only when the draft actually belongs to it."""
        evidence = {"form": form["terms"], "form_similarity": round(similarity, 4),
                    "form_matched": bool(matched)}
        if matched:
            evidence.update({
                "form_approval_rate": form["approval_rate"],
                "form_approved": form["approved"], "form_downgraded": form["downgraded"],
                "evidence": (f"Follows a form Meta approved {form['approved']} of "
                             f"{form['approved'] + form['downgraded']} times.")})
        else:
            evidence.update({
                "form_approval_rate": None, "form_approved": None, "form_downgraded": None,
                "evidence": ("No approved form closely matches this task, so its exemplars "
                             "guided the wording but no historical approval rate applies. "
                             "The score below is the only signal here.")})
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="What the message needs to say.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".data")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--target", type=float, default=0.60)
    args = parser.parse_args()
    result = Drafter(Store(args.data_dir)).draft(args.task, rounds=args.rounds, target=args.target)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
