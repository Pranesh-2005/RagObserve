import pytest

import ragobserve
from ragobserve import client as rclient


@pytest.fixture
def local_client(tmp_path):
    """Fresh local SQLite-backed client per test; resets the global config."""
    db = str(tmp_path / "test.db")
    c = ragobserve.init(project="test-project", db_path=db)
    yield c
    c.store.close()
    rclient._config.client = None
    rclient._config.project = "default"
