from datetime import datetime, timedelta, timezone

import pytest

from templatelab.data import Store, normalize_rows, suggest_mapping
from templatelab.experiments import (METHODS, Experiments, chronological_split, compare,
                                     date_value)


def dated_records():
    rows = []
    services = ['electricity', 'water', 'hosting', 'parking', 'cleaning', 'insurance', 'school', 'rental',
                'transport', 'repairs', 'laundry', 'software', 'storage', 'broadband', 'security',
                'maintenance', 'catering', 'consulting', 'membership', 'utilities']
    for i, service in enumerate(services):
        date = (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
        for category, body in [('UTILITY', f'Your {service} invoice {{{{id}}}} is due.'),
                               ('MARKETING', f'Buy {service} today and get exclusive discounts.')]:
            rows.append({"body": body, "meta_category": category, "status": "VERIFIED", "created_at": date})
    return normalize_rows(rows, suggest_mapping(rows), 'test.json')


def test_dates_boundary_and_invalid_exclusions():
    records = dated_records()
    records[0]["updated_at"] = '2025-01-30T00:00:00Z'
    records[1]["created_at"] = 'invalid'
    train, test, split = chronological_split(records)
    assert split["purged_boundary_families"] == 1
    assert split["excluded_invalid_date_families"] == 1
    assert split["group_overlap"] == 0
    assert records[0] not in train and records[0] not in test
    assert all(date_value(r["created_at"]) < date_value(split["cutoff"]) for r in train)
    assert all(date_value(r["created_at"]) >= date_value(split["cutoff"]) for r in test)
    assert date_value('bad') is None


def test_insufficient_dates_rejected():
    with pytest.raises(ValueError, match='30 clean families'):
        chronological_split([])


def test_comparison_persistence_and_staleness(tmp_path):
    store = Store(tmp_path)
    assert not Experiments(store).status()["available"]
    store.import_records(dated_records())
    result = Experiments(store).run()
    assert not result["stale"]
    report = result["report"]
    # Arms that cannot run here (no encoder installed) are recorded, not dropped.
    assert len(report["results"]) + len(report["skipped"]) == len(METHODS)
    assert {row["method"] for row in report["results"]} <= set(METHODS)
    assert report["split"]["test_families"] == 10
    for row in report["results"]:
        assert row["all"]["samples"] == 10
        assert sum(map(sum, row["all"]["confusion_matrix"])) == 10
    assert Experiments(store).status()["report"] == report
    extra = dated_records()[0].copy()
    extra["id"] = 'new-record'
    store.import_records([extra])
    assert Experiments(store).status()["stale"]
