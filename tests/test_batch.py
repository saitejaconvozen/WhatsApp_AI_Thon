import json

from templatelab.batch import load_done, summarise


def test_summary_counts_ratified_not_merely_converted():
    """Counting rewrites rewards a permissive filter; only ratified ones count."""
    rows = [
        {"id": "a", "verdict": "SPLIT_RECOMMENDED", "utility": "x", "after_category": "UTILITY",
         "before": .3, "after": .9},
        {"id": "b", "verdict": "SPLIT_RECOMMENDED", "utility": "y", "after_category": "MARKETING",
         "before": .2, "after": .25},
        {"id": "c", "verdict": "NEEDS_CONTEXT"},
        {"id": "d", "verdict": "ERROR", "error": "timeout"},
    ]
    s = summarise(rows)
    assert s["templates"] == 4
    assert s["produced_a_rewrite"] == 2
    assert s["ratified_as_utility"] == 1
    assert s["mean_score_gain"] == 0.6
    assert s["failures"] == 1
    assert s["verdicts"]["NEEDS_CONTEXT"] == 1


def test_a_half_written_line_does_not_break_resume(tmp_path):
    """An interrupted run leaves a truncated final line; it must be skipped."""
    path = tmp_path / "conversions.jsonl"
    path.write_text(json.dumps({"id": "a", "verdict": "NEEDS_CONTEXT"}) + "\n{\"id\": \"b\", trunc",
                    encoding="utf-8")
    done = load_done(path)
    assert list(done) == ["a"]


def test_missing_file_resumes_from_nothing(tmp_path):
    assert load_done(tmp_path / "absent.jsonl") == {}


def test_placeholder_loss_is_counted_over_conversions_not_refusals():
    """A refusal has no rewrite, so every placeholder trivially looks lost."""
    from templatelab import publish
    rows = [{"id": "a", "verdict": "NEEDS_CONTEXT", "utility": None, "placeholders_lost": ["{{1}}", "{{2}}"]},
            {"id": "b", "verdict": "SPLIT_RECOMMENDED", "utility": "x", "after_category": "UTILITY",
             "placeholders_lost": ["{{3}}"]}]
    counted = sum(1 for r in rows if r.get("utility") and r.get("placeholders_lost"))
    assert counted == 1
