import argparse
import json
from pathlib import Path

from .data import Store, normalize_rows, parse_file, suggest_mapping, summarize


def main():
    parser = argparse.ArgumentParser(description="Import a local template export into Template Lab.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".data")
    args = parser.parse_args()
    rows = parse_file(args.file.read_bytes(), args.file.name)
    records = normalize_rows(rows, suggest_mapping(rows), args.file.name)
    store = Store(args.data_dir)
    result = store.import_records(records)
    print(json.dumps({**result, "summary": summarize(store.records())}, indent=2))


if __name__ == "__main__":
    main()
