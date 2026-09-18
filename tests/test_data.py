import io
import json

import pytest
from openpyxl import Workbook

from templatelab.data import Store, candidate, normalize_rows, parse_file, suggest_mapping, summarize, training_records


def draft(body="Your invoice {{id}} is ready.", observed="UTILITY", requested="UTILITY", status="VERIFIED", kind="TEXT"):
    row = {"_id": "example", "templateName": "Invoice", "messageBody": {
        "type": kind, "templateCategory": requested, "body": body,
        "interactionConfiguration": {"buttonConfiguration": {"ctaButtons": [{"text": "View invoice"}]}}
    }, "verificationStatus": status, "createdAt": {"$date": "2026-09-01T00:00:00Z"},
        "createdBy": "private@example.com", "whatsappPhoneNumberId": "private-account"}
    if observed:
        row["metaTemplateCategory"] = observed
    return row


def normalize(rows):
    return normalize_rows(rows, suggest_mapping(rows), "examples.json")


def test_observed_and_requested_labels_remain_separate():
    record = normalize([draft(observed="MARKETING")])[0]
    assert record["requested_category"] == "UTILITY"
    assert record["meta_category"] == "MARKETING"
    assert record["buttons"] == "View invoice"
    assert "createdBy" not in record
    assert "whatsappPhoneNumberId" not in record


def test_missing_observed_label_is_not_inferred_from_draft():
    record = normalize([draft(observed=None)])[0]
    assert record["meta_category"] == "UNKNOWN"
    assert not candidate(record)


@pytest.mark.parametrize("status,observed,kind", [("PENDING", "UTILITY", "TEXT"), ("REJECTED", "UTILITY", "TEXT"), ("VERIFIED", "AUTHENTICATION", "TEXT"), ("VERIFIED", "UTILITY", "MEDIA")])
def test_ineligible_records_stay_out_of_training(status, observed, kind):
    assert not candidate(normalize([draft(status=status, observed=observed, kind=kind)])[0])


def test_families_group_values_and_quarantine_conflicts():
    records = normalize([draft("Invoice 123 is ready.", "UTILITY"), draft("Invoice 456 is ready.", "MARKETING")])
    assert records[0]["family"] == records[1]["family"]
    clean, conflicts = training_records(records)
    assert not clean
    assert len(conflicts) == 1
    assert summarize(records)["conflicting_records"] == 2


def test_family_representatives_are_deduplicated():
    records = normalize([draft("Invoice {{invoice_id}} is ready."), draft("Invoice {{1}} is ready.")])
    assert len(training_records(records)[0]) == 1


def test_import_is_idempotent_and_durable(tmp_path):
    records = normalize([draft(), draft(observed="MARKETING")])
    store = Store(tmp_path)
    assert store.import_records(records)["added"] == 2
    assert store.import_records(records)["already_present"] == 2
    assert len(Store(tmp_path).records()) == 2


@pytest.mark.parametrize("name,content", [("a.json", b"not json"), ("a.json", b"[]"), ("a.json", b"[1,2]"), ("a.csv", b"")])
def test_invalid_uploads_raise_readable_error(name, content):
    with pytest.raises(ValueError):
        parse_file(content, name)


def test_json_txt_csv_jsonl_and_xlsx():
    rows = [{"body": "Your invoice is ready", "meta_category": "UTILITY", "status": "VERIFIED"}]
    assert parse_file(json.dumps(rows).encode(), "Pasted text.txt") == rows
    assert parse_file(json.dumps(rows[0]).encode(), "templates.jsonl") == rows
    assert parse_file(b"body,meta_category,status\nYour invoice is ready,UTILITY,VERIFIED\n", "templates.csv") == rows
    book = Workbook()
    book.active.append(list(rows[0].keys()))
    book.active.append(list(rows[0].values()))
    stream = io.BytesIO()
    book.save(stream)
    assert parse_file(stream.getvalue(), "templates.xlsx") == rows


def test_flat_category_is_requested_not_observed():
    rows = [{"text": "Invoice ready", "category": "UTILITY", "status": "VERIFIED"}]
    record = normalize(rows)[0]
    assert record["meta_category"] == "UNKNOWN"
    assert record["requested_category"] == "UTILITY"
