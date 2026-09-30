"""A plugin's own settings: ``[plugins.<distribution name>]``, read with ``agent_env.plugins.settings``.

The namespace is a promise in two directions. To a plugin: this table is yours, found under your
distribution name however the file spells it, with references resolved like any other table. To
the operator: ``config show`` and ``config explain`` report it under the distribution it belongs to,
masked like core, and core makes no claim about what is inside it or whether anything reads it.
"""

import importlib
import json
import sys

import pytest
from click.testing import CliRunner

from agent_env import plugins
from agent_env.cli.config import config as config_group
from agent_env.cli.config import render, render_explain
from agent_env.config import ConfigError, get_config, reset_config, snapshot
from agent_env.config.describe import (
    MASK,
    SectionsReport,
    as_dict,
    describe_config,
    env_sources,
    explain_path,
)
from agent_env.config.runtime import Config
from agent_env.store.secret_store import LocalSecretStore


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for source in env_sources():
        monkeypatch.delenv(source.name, raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    yield
    reset_config()


def _use(tmp_path, monkeypatch, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    reset_config()
    return path


def _installed(tmp_path, monkeypatch, name: str, version: str, *, entry_points: bool = True):
    """A real installed distribution on sys.path. With ``entry_points`` it declares an
    ``agent_env.*`` entry point, which is what makes it a plugin; its module is never importable,
    so a report that imported plugin code would fail."""
    site = tmp_path / "site"
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if entry_points:
        (info / "entry_points.txt").write_text("[agent_env.envs]\ndemo = not_importable_demo:DemoEnv\n")
    (info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(site))
    importlib.invalidate_caches()


def _tables(report):
    return {t.owner.name: t for t in report.plugins.tables}


def _run(args):
    return CliRunner().invoke(config_group, args, catch_exceptions=False)


# ---------------------------------------------------------------- the read helper


def test_a_plugin_reads_its_own_table_and_nothing_else(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, """
[sandbox]
default = "local"
[plugins.agentenv-browser]
timeout = 30
[plugins.agentenv-browser.viewport]
width = 1280
[plugins.agentenv-grader]
rubric = "strict"
""")

    assert plugins.settings("agentenv-browser") == {"timeout": 30, "viewport": {"width": 1280}}
    assert plugins.settings("agentenv-grader") == {"rubric": "strict"}


@pytest.mark.parametrize("body", ["", '[sandbox]\ndefault = "local"\n', "[plugins.someone-else]\nx = 1\n"])
def test_no_table_is_an_empty_one(tmp_path, monkeypatch, body):
    _use(tmp_path, monkeypatch, body)

    assert plugins.settings("agentenv-browser") == {}


def test_with_no_config_file_there_are_no_settings(tmp_path):
    assert plugins.settings("agentenv-browser") == {}


@pytest.mark.parametrize("asked", ["agentenv-browser", "agentenv_browser", "AgentEnv.Browser"])
@pytest.mark.parametrize("written", ["agentenv-browser", "agentenv_browser", '"AgentEnv.Browser"'])
def test_the_table_is_found_by_canonical_name(tmp_path, monkeypatch, asked, written):
    """pip, uv and every installer treat these as one package, so the file and the plugin may
    spell it either way."""
    _use(tmp_path, monkeypatch, f"[plugins.{written}]\ntimeout = 30\n")

    assert plugins.settings(asked) == {"timeout": 30}


def test_two_spellings_of_one_package_are_refused_not_picked(tmp_path, monkeypatch):
    """Either pick would be a guess the report could not show, so neither is taken."""
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\ntimeout = 1\n[plugins.agentenv_browser]\ntimeout = 2\n")

    with pytest.raises(ConfigError, match=r"agentenv-browser's settings more than once, as "
                                          r"\[plugins.agentenv-browser\], \[plugins.agentenv_browser\]"):
        plugins.settings("agentenv-browser")


def test_a_plugins_value_that_is_not_a_table_fails_every_read(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "plugins = 5\n")

    with pytest.raises(ConfigError, match=r"\[plugins\] must be a table, got int"):
        plugins.settings("agentenv-browser")


def test_an_entry_that_is_not_a_table_fails_only_its_own_plugin(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[plugins]\nagentenv-browser = "fast"\n[plugins.agentenv-grader]\nrubric = "strict"\n')

    with pytest.raises(ConfigError, match=r"\[plugins.agentenv-browser\] must be a table, got str"):
        plugins.settings("agentenv-browser")
    assert plugins.settings("agentenv-grader") == {"rubric": "strict"}


def test_references_are_resolved_as_in_every_other_table(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, """
[plugins.agentenv-browser]
endpoint = "env:BROWSER_URL"
fallback = "env:BROWSER_UNSET?http://localhost:9222"
token = "secret:browser_token"
hosts = ["env:BROWSER_URL"]
""")
    monkeypatch.setenv("BROWSER_URL", "http://browser.invalid")
    get_config().set_secret_store(LocalSecretStore({"browser_token": "t0ken"}, use_env=False))

    assert plugins.settings("agentenv-browser") == {
        "endpoint": "http://browser.invalid",
        "fallback": "http://localhost:9222",
        "token": "t0ken",
        "hosts": ["http://browser.invalid"],
    }


def test_an_unresolvable_reference_is_a_config_error(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[plugins.agentenv-browser]\nendpoint = "env:BROWSER_UNSET"\n')

    with pytest.raises(ConfigError, match="Unresolved env:BROWSER_UNSET"):
        plugins.settings("agentenv-browser")


def test_each_read_is_a_copy(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser.viewport]\nwidth = 1280\n")

    plugins.settings("agentenv-browser")["viewport"]["width"] = 1
    assert plugins.settings("agentenv-browser") == {"viewport": {"width": 1280}}


def test_a_given_config_is_read_instead_of_the_process_one(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\ntimeout = 1\n")
    other = Config()
    other._snapshot = snapshot.Snapshot(path=None, _document={"plugins": {"agentenv-browser": {"timeout": 2}}})

    assert plugins.settings("agentenv-browser", config=other) == {"timeout": 2}
    assert plugins.settings("agentenv-browser") == {"timeout": 1}


def test_no_environment_variable_overrides_a_plugin_setting():
    """A distribution name holds `-` and `.`, so no variable name maps back to it unambiguously;
    a plugin that wants an override reads its own variable, or an `env:` reference."""
    assert not [s for s in env_sources() if s.shadows and s.shadows[0] == "plugins"]


# ---------------------------------------------------------------- config show


def test_show_lists_each_table_under_its_plugin_and_whether_it_is_installed(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch, "agentenv-browser", "1.2.0")
    _installed(tmp_path, monkeypatch, "requests-like", "2.0", entry_points=False)
    _use(tmp_path, monkeypatch, """
[plugins.agentenv_browser]
timeout = 30
[plugins.agentenv-grader]
rubric = "strict"
[plugins.requests-like]
x = 1
""")

    report = describe_config()
    tables = _tables(report)
    assert list(tables) == ["agentenv-browser", "agentenv-grader", "requests-like"]
    assert (tables["agentenv-browser"].owner.version, tables["agentenv-browser"].where) == \
        ("1.2.0", "[plugins.agentenv_browser]")
    assert tables["agentenv-grader"].owner.installed is None
    # installed, but declaring no agent_env entry point it is no plugin: the label says both
    assert (tables["requests-like"].owner.version, tables["requests-like"].owner.plugin) == ("2.0", False)

    out = render(report)
    assert "plugins:        agentenv-browser 1.2.0\n                  timeout=30\n" \
           "                  from [plugins.agentenv_browser]" in out
    assert "agentenv-grader (not installed)" in out
    assert "requests-like 2.0 (declares no agent_env entry point)" in out
    assert "not_importable_demo" not in sys.modules

    shown = as_dict(report)["plugins"]
    assert shown["error"] is None
    assert shown["tables"][0] == {"name": "agentenv-browser", "installed": True, "version": "1.2.0", "plugin": True,
                                  "keys": ["agentenv_browser"], "value": {"timeout": 30}, "error": None}
    assert shown["tables"][1]["installed"] is False and shown["tables"][1]["version"] is None
    assert (shown["tables"][2]["installed"], shown["tables"][2]["plugin"]) == (True, False)


def test_show_with_no_plugin_tables_says_so(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[sandbox]\ndefault = "local"\n')

    report = describe_config()
    assert "plugins:        (none)" in render(report)
    assert as_dict(report)["plugins"] == {"error": None, "tables": []}


def test_the_plugins_label_does_not_widen_the_layout(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\ntimeout = 30\n")

    out = render(describe_config())
    assert "sandbox:        (unset)" in out and "plugins:        agentenv-browser" in out


def test_plugin_tables_are_masked_one_table_at_a_time(tmp_path, monkeypatch):
    """Masking `[plugins]` whole would scope on the distribution names: `agentenv-oauth-bridge`
    holds `auth`, and every one of its settings would print as `***`."""
    _use(tmp_path, monkeypatch, """
[plugins.agentenv-oauth-bridge]
region = "us-west-2"
client_secret = "s3cr3t"
endpoint = "env:BRIDGE_URL"
dsn = "postgres://user:pw@db/x"
[plugins.agentenv-oauth-bridge.credentials]
user = "someone"
""")

    report = describe_config()
    value = _tables(report)["agentenv-oauth-bridge"].value
    assert value == {"region": "us-west-2", "client_secret": MASK, "endpoint": "env:BRIDGE_URL",
                     "dsn": f"postgres://{MASK}@db/x", "credentials": {"user": MASK}}
    rendered, serialized = render(report), json.dumps(as_dict(report))
    for literal in ("s3cr3t", "someone", "user:pw"):
        assert literal not in rendered and literal not in serialized


def test_a_plugins_value_that_is_not_a_table_is_reported_in_place(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "plugins = 5\n")

    report = describe_config()
    assert "must be a table, got int" in report.plugins.error
    assert "plugins:        (unresolved) config.toml [plugins] must be a table" in render(report)
    assert as_dict(report)["plugins"]["tables"] == []


def test_a_bad_entry_is_reported_and_the_others_still_are(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, """
[plugins]
agentenv-browser = "fast"
[plugins.agentenv-grader]
rubric = "strict"
[plugins.agentenv-web]
a = 1
[plugins.agentenv_web]
a = 2
""")

    tables = _tables(describe_config())
    assert "[plugins.agentenv-browser] must be a table, got str" in tables["agentenv-browser"].error
    assert tables["agentenv-grader"].value == {"rubric": "strict"} and tables["agentenv-grader"].error is None
    assert tables["agentenv-web"].value is None
    assert tables["agentenv-web"].where == "[plugins.agentenv-web], [plugins.agentenv_web]"
    assert "more than once" in tables["agentenv-web"].error


def test_an_unreadable_file_is_reported_rather_than_listed_as_no_tables(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "plugins = [[[\n")

    report = describe_config()
    assert "Malformed" in report.plugins.error


def test_core_makes_no_claim_about_the_keys_in_a_plugin_table(tmp_path, monkeypatch):
    """The `config` check is a statement about what core reads, so a plugin's `config` table and
    its keys named like core ones are the plugin's business; `[explorer.plugins]` is core's own."""
    _use(tmp_path, monkeypatch, """
[explorer.plugins]
impls = []

[plugins.agentenv-browser]
impl = "agentenv_browser.pool:Pool"
size = 1
model = "a-model"
envs = ["x"]
[plugins.agentenv-browser.config]
size = 2
""")

    assert describe_config().warnings == []


def test_the_same_checks_still_cover_core_tables(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, """
[stores.document]
impl = "agent_env.store.document_store:MongoDocumentStore"
database = "beside"
[stores.document.config]
database = "under"

[task_steps]
impls = []
envs = "pkg:VmEnv"

[plugins.agentenv-browser]
x = 1
""")

    warnings = describe_config().warnings
    assert len(warnings) == 2
    assert any("both beside its config table" in w for w in warnings)
    assert any("Did you mean [envs]?" in w for w in warnings)


def test_a_plugin_table_is_shown_as_written_not_read_as_a_seam(tmp_path, monkeypatch):
    """`impl` and `config` mean something in agent-env's tables, and nothing agent-env knows in a
    plugin's: reading them as a class and its kwargs would hide `size = 1` behind `size = 2`."""
    _use(tmp_path, monkeypatch, """
[plugins.agentenv-browser]
impl = "agentenv_browser.pool:Pool"
size = 1
[plugins.agentenv-browser.config]
size = 2
""")

    shown = render(describe_config())
    assert "impl=agentenv_browser.pool:Pool  size=1\n                  config:\n" \
           "                    size=2\n" in shown
    explained = render_explain(explain_path("plugins.agentenv-browser"))
    assert "impl=agentenv_browser.pool:Pool  size=1" in explained and "  config:\n    size=2" in explained
    children = render_explain(explain_path("plugins"))
    assert "impl=agentenv_browser.pool:Pool  size=1" in children


def test_a_dotted_key_is_named_the_way_the_file_must_spell_it(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[plugins."AgentEnv.Browser"]\ntimeout = 30\n')

    (table,) = describe_config().plugins.tables
    assert (table.owner.name, table.where) == ("agentenv-browser", '[plugins."AgentEnv.Browser"]')


def test_a_distribution_whose_metadata_cannot_be_read_is_not_installed(tmp_path, monkeypatch):
    """Its entry points cannot be read, so none of its plugins loads; the others are still found."""
    _installed(tmp_path, monkeypatch, "agentenv-browser", "1.2.0")
    torn = tmp_path / "torn" / "agentenv_torn-1.0.dist-info"
    torn.mkdir(parents=True)
    (torn / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-torn\nVersion: 1.0\n")
    (torn / "entry_points.txt").write_text("[agent_env.envs]\nno equals sign\n")
    monkeypatch.syspath_prepend(str(tmp_path / "torn"))
    _use(tmp_path, monkeypatch, "[plugins.agentenv-torn]\nx = 1\n[plugins.agentenv-browser]\nx = 1\n")

    tables = _tables(describe_config())
    assert tables["agentenv-torn"].owner.installed is None
    assert tables["agentenv-browser"].owner.version == "1.2.0"


def test_show_json_through_the_cli(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\napi_key = 12345\nsince = 2026-09-27\n")

    shown = json.loads(_run(["show", "--json"]).output)
    assert shown["plugins"]["tables"][0]["value"] == {"api_key": MASK, "since": "2026-09-27"}
    assert [s["name"] for s in shown["sections"]][-1] == "explorer"   # core sections unchanged


def test_a_broken_plugins_table_breaks_nothing_until_a_plugin_reads_it(tmp_path, monkeypatch):
    """Checked where it is read, like every other table: agent-env itself never reads it."""
    _use(tmp_path, monkeypatch, 'plugins = 5\n[sandbox]\ndefault = "local"\n')

    config = get_config()
    config.env_registry(), config.task_step_registry(), config.sandbox_registry()
    assert config.section("sandbox") == {"default": "local"}
    report = describe_config(config)
    assert all(s.error is None for s in report.sections) and report.warnings == []
    assert _run(["show"]).exit_code == 0


# ---------------------------------------------------------------- config explain


def test_explain_names_the_plugin_that_reads_a_key(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch, "agentenv-browser", "1.2.0")
    _use(tmp_path, monkeypatch, "[plugins.agentenv_browser]\ntimeout = 30\n")

    report = explain_path("plugins.agentenv-browser.timeout")
    assert (report.value, report.section, report.file_only) == (30, "plugins", False)
    assert report.winner.where == "[plugins.agentenv_browser]"
    assert (report.plugin.name, report.plugin.version) == ("agentenv-browser", "1.2.0")
    out = render_explain(report)
    assert "; in the table of the plugin agentenv-browser 1.2.0; agent-env does not read or check it" in out
    assert "no resolver owns this path" not in out

    shown = json.loads(_run(["explain", "plugins.agentenv_browser.timeout", "--json"]).output)
    assert shown["plugin"] == {"name": "agentenv-browser", "installed": True, "version": "1.2.0", "plugin": True}
    assert (shown["section"], shown["file_only"], shown["value"]) == ("plugins", False, 30)


def test_explain_says_when_the_plugin_is_not_installed(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-grader]\nrubric = \"strict\"\n")

    report = explain_path("plugins.agentenv-grader")
    assert report.value == {"rubric": "strict"}
    assert "; in the table of agentenv-grader, which is not installed; agent-env does not read it" in render_explain(report)


def test_explain_of_an_absent_plugin_key_is_unset_and_still_names_the_plugin(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[sandbox]\ndefault = "local"\n')

    report = explain_path("plugins.agentenv-grader.rubric")
    assert report.winner is None and report.value is None
    out = render_explain(report)
    assert "(unset) — no layer supplies this path" in out and "agentenv-grader" in out


def test_explain_masks_plugin_values_on_the_same_terms(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, """
[plugins.agentenv-oauth-bridge]
region = "us-west-2"
[plugins.agentenv-oauth-bridge.credentials]
user = "someone"
""")

    assert explain_path("plugins.agentenv-oauth-bridge.region").value == "us-west-2"
    assert explain_path("plugins.agentenv-oauth-bridge.credentials.user").value == MASK
    assert explain_path("plugins.agentenv-oauth-bridge").value["region"] == "us-west-2"


def test_explain_of_plugins_lists_every_table(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-a]\nx = 1\n[plugins.agentenv-oauth-b]\nregion = \"r\"\n")

    report = explain_path("plugins")
    assert isinstance(report, SectionsReport)
    assert [(c.path, c.value) for c in report.children] == [
        ("plugins.agentenv-a", {"x": 1}), ("plugins.agentenv-oauth-b", {"region": "r"})]


def test_explain_of_plugins_with_none_is_unset(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[sandbox]\ndefault = "local"\n')

    report = explain_path("plugins")
    assert report.winner is None and report.error is None


@pytest.mark.parametrize("body, path, error", [
    ("plugins = 5\n", "plugins.agentenv-browser.x", "[plugins] must be a table"),
    ('[plugins]\nagentenv-browser = 5\n', "plugins.agentenv-browser.x", "[plugins.agentenv-browser] must be a table"),
    ("[plugins.agentenv-browser]\nx = 1\n[plugins.agentenv_browser]\nx = 2\n", "plugins.agentenv-browser.x",
     "more than once"),
])
def test_explain_reports_a_bad_table_instead_of_a_value(tmp_path, monkeypatch, body, path, error):
    _use(tmp_path, monkeypatch, body)

    report = explain_path(path)
    assert error in report.error and report.value is None
    assert "(unresolved)" in render_explain(report)


# ---------------------------------------------------------------- review follow-ups


def test_a_package_named_like_a_core_section_is_a_plugin_table_not_a_misplaced_one(tmp_path, monkeypatch):
    """Every key in [plugins] is a package name: TOML cannot tell [plugins.sandbox] from a bare
    `sandbox` bound under [plugins], and either way it is listed as that package's table."""
    _use(tmp_path, monkeypatch, "[plugins.sandbox]\nx = 1\n[plugins]\nmodel = { y = 2 }\n")

    report = describe_config()
    assert report.warnings == []
    assert [t.owner.name for t in report.plugins.tables] == ["sandbox", "model"]


def test_plugins_bound_under_another_table_by_a_bare_key_warns(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[sandbox]\ndefault = "local"\nplugins = { acme = { x = 1 } }\n')

    assert describe_config().warnings == [
        "[sandbox] has a key named 'plugins', which is also a top-level section — a bare key binds "
        "to the table above it. Did you mean [plugins]?"
    ]
    assert plugins.settings("acme") == {}


@pytest.mark.parametrize("table", ["plugin", "Plugins", "plugns"])
def test_a_table_one_letter_from_plugins_gets_a_did_you_mean(tmp_path, monkeypatch, table):
    _use(tmp_path, monkeypatch, f"[{table}.agentenv-browser]\ntimeout = 60\n")

    assert f"agent-env does not read [{table}]: a plugin's settings go under [plugins.<package>]. Did you mean [plugins]?" \
        in describe_config().warnings
    assert plugins.settings("agentenv-browser") == {}


def test_plugins_and_explorer_plugins_get_no_did_you_mean(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\nx = 1\n[explorer.plugins]\nimpls = []\n")

    assert describe_config().warnings == []


def test_the_core_distribution_is_agent_env_itself_not_a_plugin(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-framework]\nx = 1\n")

    (table,) = describe_config().plugins.tables
    assert table.owner.installed is not None and table.owner.plugin is False
    assert table.owner.label.endswith("(agent-env itself)")
    assert "which is agent-env itself, not a plugin; agent-env does not read it" in \
        render_explain(explain_path("plugins.agentenv-framework.x"))


@pytest.mark.parametrize(("body", "key", "kept"), [
    ('[plugins.agentenv-toy]\napi_key = "env:TOY_KEY?sk-literal"\n', "api_key", "env:TOY_KEY?***"),
    ('[plugins.agentenv-toy.credentials]\nuser = "env:U?sk-literal"\n', "credentials", "env:U?***"),
])
def test_a_reference_keeps_its_name_but_its_default_under_a_secret_key_is_masked(
    tmp_path, monkeypatch, body, key, kept
):
    _use(tmp_path, monkeypatch, body)

    report = describe_config()
    shown = json.dumps(as_dict(report)) + render(report)
    shown += render_explain(explain_path(f"plugins.agentenv-toy.{key}"))
    assert "sk-literal" not in shown and kept in shown


def test_a_core_secret_default_is_masked_and_a_uri_default_hides_its_userinfo(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[model]\napi_key = "env:K?sk-core-literal"\n'
                                '[plugins.agentenv-toy]\nurl = "env:DB?postgres://u:hunter2@db/app"\n')

    shown = json.dumps(as_dict(describe_config()))
    assert "sk-core-literal" not in shown and "hunter2" not in shown
    assert "env:DB?postgres://***@db/app" in shown


def test_a_quoted_table_label_from_show_round_trips_through_explain(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[plugins."Acme.Toy"]\ntimeout = 30\n')

    (table,) = describe_config().plugins.tables
    assert table.where == '[plugins."Acme.Toy"]'
    report = explain_path('plugins."Acme.Toy".timeout')
    assert (report.value, report.plugin.name) == (30, "acme-toy")


def test_a_dotted_name_as_plugin_list_prints_it_is_found(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch, "AgentEnv.Toy", "1.2.3")
    _use(tmp_path, monkeypatch, '[plugins.agentenv-toy]\nregion = "upper"\n')

    report = explain_path("plugins.AgentEnv.Toy.region")
    assert (report.value, report.plugin.name, report.plugin.version) == ("upper", "agentenv-toy", "1.2.3")


def test_the_first_segment_is_the_table_when_one_is_named_so(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, '[plugins.agentenv-toy]\nregion = "a"\n[plugins.agentenv-toy-region]\nx = 1\n')

    assert explain_path("plugins.agentenv-toy.region").value == "a"


def test_explain_plugins_lists_each_table_with_its_owner(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch, "agentenv-browser", "1.2.0")
    _use(tmp_path, monkeypatch, "[plugins.agentenv-browser]\nx = 1\n[plugins.agentenv-grader]\ny = 2\n")

    out = render_explain(explain_path("plugins"))
    assert "covers 2 plugin tables" in out
    assert "(agentenv-browser 1.2.0)" in out and "(agentenv-grader (not installed))" in out
    children = json.loads(_run(["explain", "plugins", "--json"]).output)["children"]
    assert [c["plugin"]["name"] for c in children] == ["agentenv-browser", "agentenv-grader"]
    assert [c["plugin"]["plugin"] for c in children] == [True, False]


def test_show_json_lists_the_keys_as_written(tmp_path, monkeypatch):
    _use(tmp_path, monkeypatch, "[plugins.agentenv-web]\nx = 1\n[plugins.agentenv_web]\nx = 2\n")

    (table,) = json.loads(_run(["show", "--json"]).output)["plugins"]["tables"]
    assert table["keys"] == ["agentenv-web", "agentenv_web"]
    assert set(table) == {"name", "installed", "version", "plugin", "keys", "value", "error"}


def test_a_config_that_cannot_be_found_is_the_plugins_block_error_too(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "nope.toml"))
    reset_config()

    report = describe_config()
    assert report.plugins.error and "does not point to an existing file" in report.plugins.error


def test_an_unresolved_reference_names_the_table_it_is_in(tmp_path, monkeypatch):
    monkeypatch.delenv("ACME_URL", raising=False)
    _use(tmp_path, monkeypatch, '[plugins.agentenv-toy]\nendpoint = "env:ACME_URL"\n')

    with pytest.raises(ConfigError, match=r"\[plugins.agentenv-toy\]: Unresolved env:ACME_URL reference"):
        plugins.settings("agentenv-toy")
