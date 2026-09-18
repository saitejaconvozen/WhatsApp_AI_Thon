"""Keep the suite hermetic against a developer's local .env.

templatelab.load_env() reads .env on import, so a machine with real credentials
configured would otherwise run tests with a live backend: the suite would make
network calls, and the tests that assert a missing key is reported cleanly would
fail on the developer's machine while passing in CI. Clearing these for every
test makes the run identical either way; a test that wants a backend sets one
with monkeypatch.setenv, as several already do.
"""

import pytest

CREDENTIALS = ("TEMPLATELAB_LLM_API_KEY", "TEMPLATELAB_LLM_BACKEND",
               "TEMPLATELAB_LLM_MODEL", "TEMPLATELAB_LLM_BASE_URL",
               "TEMPLATELAB_LLM_ALLOW_EGRESS", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
