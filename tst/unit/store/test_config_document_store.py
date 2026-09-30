"""Backend injection + AGENT_ENV_DOCUMENT_STORE / config.toml selection for get_document_store().

Mongo-free (the tst/unit socket guard forbids network): injection returns a
fake verbatim, and the ``local`` selector / a custom ``impl`` build real stores
under the test's state root.
"""

import pytest

from agent_env.store import ConfigError, Filter, LocalSqliteDocumentStore
from agent_env.config import Config, configure, get_config, set_document_store
from agent_env.config.paths import state_root
from tst.unit.store.fakes import FakeDocumentStore

_FAKE_SECTION = '[stores.document]\nimpl = "tst.unit.store.fakes:FakeDocumentStore"\n'


def _write_config(tmp_path, body):
    agentenv = tmp_path / ".agentenv"
    agentenv.mkdir(exist_ok=True)
    (agentenv / "config.toml").write_text(body)


def test_set_document_store_returns_it_verbatim():
    cfg = Config()
    fake = FakeDocumentStore()
    cfg.set_document_store(fake)
    assert cfg.get_document_store() is fake  # no Mongo built — override short-circuits


def test_module_level_set_document_store_overrides_singleton():
    fake = FakeDocumentStore()
    set_document_store(fake)
    assert get_config().get_document_store() is fake


def test_configure_carries_document_store():
    fake = FakeDocumentStore()
    configure(document_store=fake)
    assert get_config().get_document_store() is fake


def test_env_selector_builds_local_sqlite(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    store = Config().get_document_store()
    assert isinstance(store, LocalSqliteDocumentStore)
    store.insert("coll", {"id": "a", "x": 1})
    assert store.find_one("coll", Filter.of(id="a"))["x"] == 1
    assert (state_root() / "document_store" / "documents.db").exists()
    assert list(tmp_path.iterdir()) == []


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_DOCUMENT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    section = Config()._resolve_document_section()
    assert section["impl"] == "agent_env.store.document_store:LocalSqliteDocumentStore"


def test_mongo_alias_raises_actionable(monkeypatch):
    # No built-in coordinates: the error names the table AND the overriding env var.
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "mongo")
    with pytest.raises(ConfigError, match=r"\[stores\.document\].*AGENT_ENV_DOCUMENT_STORE"):
        Config().get_document_store()


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "bogus")
    with pytest.raises(ValueError, match="AGENT_ENV_DOCUMENT_STORE"):
        Config().get_document_store()


def test_config_toml_selects_custom_impl(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_DOCUMENT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    assert isinstance(Config().get_document_store(), FakeDocumentStore)


def test_env_override_beats_config_toml(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    assert isinstance(Config().get_document_store(), LocalSqliteDocumentStore)


def test_local_in_a_project_config_is_the_per_user_store(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_DOCUMENT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    first, second = tmp_path / "first", tmp_path / "second"
    for project in (first, second):
        project.mkdir()
        _write_config(project, '[stores]\ndocument = "local"\n')
    monkeypatch.chdir(first)
    Config().get_document_store().insert("coll", {"id": "a"})
    monkeypatch.chdir(second)
    assert Config().get_document_store().find_one("coll", Filter.of(id="a")) is not None
    assert (state_root() / "document_store" / "documents.db").exists()
    assert sorted(p.name for p in (first / ".agentenv").iterdir()) == ["config.toml"]


def test_a_configured_path_still_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_DOCUMENT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "mine" / "documents.db"
    _write_config(tmp_path, (
        '[stores.document]\nimpl = "agent_env.store.document_store:LocalSqliteDocumentStore"\n'
        f'[stores.document.config]\npath = "{path}"\n'
    ))
    Config().get_document_store().insert("coll", {"id": "a"})
    assert path.exists()
    assert not state_root().exists()


@pytest.mark.parametrize("blocked", ["state home", "store directory"])
def test_a_store_directory_that_cannot_be_created_is_a_config_error(monkeypatch, tmp_path, blocked):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    if blocked == "state home":
        blocker = tmp_path / "a-file"
        blocker.write_text("")
        monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    else:
        state_root().mkdir(parents=True)
        (state_root() / "document_store").write_text("")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    with pytest.raises(ConfigError, match="set XDG_STATE_HOME to a writable directory"):
        Config().get_document_store().insert("coll", {"id": "a"})
