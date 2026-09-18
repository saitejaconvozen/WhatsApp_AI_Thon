"""Local WhatsApp template analysis workspace."""

import os
from pathlib import Path


def load_env(path=None):
    """Read .env into the environment, without adding a dependency.

    Values already exported in the shell win: an explicit `export` should not be
    silently overridden by a stale file. Returns the names that were set, never
    the values, so a caller can confirm configuration without logging a key.
    """
    path = Path(path or Path(__file__).resolve().parent.parent / ".env")
    if not path.exists():
        return []
    applied = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    return applied


load_env()
