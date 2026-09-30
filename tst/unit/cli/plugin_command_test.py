"""``agent-env plugin list / show / check``: the installed plugins, from the CLI.

Type plugins come from ``agent_env.plugins.inventory()``; CLI plugins from what the CLI loaders
recorded for the root group, so nothing is loaded twice. ``check`` exits 1 when any contribution did
not take effect, and a config replacement is not a failure.
"""

import importlib
import json
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.cli.plugin import _KINDS, environment, plugin
from agent_env.config import get_config
from agent_env.plugins import _cli as cli_plugins_mod
from agent_env.plugins import _discovery, _inventory
from agent_env.plugins._cli import load_cli_plugins, load_cli_root_options
from tst.unit.plugins_test import _EP, _HERE, _fresh, _install, _use_config  # noqa: F401  (_fresh: autouse)

_THIS = "tst.unit.cli.plugin_command_test"

tools = click.Group(name="tools", help="Demo tools.")
# The usual click idiom, `@click.group() def cli()`, names the group "cli".
cli_named_group = click.Group(name="cli", help="Demo group.")
shadow = click.Command(name="plugin")
tenant = click.Option(["--tenant"], expose_value=False)


class _CliEP:
    """A CLI entry point from a named distribution."""

    def __init__(self, name: str, attr: str, group: str, dist: str = "agentenv-tools", version: str = "0.2.0",
                 requires: list[str] | None = None):
        self.name = name
        self.value = f"{_THIS}:{attr}"
        self.group = group
        self.dist = type("Dist", (), {"name": dist, "version": version, "requires": requires})()

    def load(self):
        module, _, attr = self.value.partition(":")
        return getattr(importlib.import_module(module), attr)


class _NoisyCliEP(_CliEP):
    """A CLI entry point whose package prints while it is imported."""

    def load(self):
        print("hello from a CLI plugin")
        return super().load()


def _root_with_cli_plugins(monkeypatch, commands=(), options=()) -> click.Group:
    """A root group shaped like the real one: core groups first, then the plugin loaders."""
    root = click.Group()
    root.add_command(plugin)
    by_group = {cli_plugins_mod.CLI_PLUGINS_GROUP: list(commands), cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP: list(options)}
    monkeypatch.setattr(cli_plugins_mod, "entry_points", lambda *, group: by_group.get(group, []))
    load_cli_plugins(root)
    load_cli_root_options(root)
    return root


def _run(root: click.Group, *args: str):
    return CliRunner().invoke(root, ["plugin", *args], catch_exceptions=False)


def _broken(name: str, **kwargs) -> _EP:
    ep = _EP(name, "_BrowserEnv", **kwargs)
    ep.value = "agentenv_broken.nowhere:Missing"
    return ep


# ---------------------------------------------------------------- list


def test_list_shows_each_package_what_it_provides_and_its_status(monkeypatch):
    _install(monkeypatch,
             envs=[_EP("browser", "_BrowserEnv", dist="agentenv-browser", version="1.0.0")],
             task_steps=[_EP("plugin_step", "_PluginStep", dist="agentenv-browser", version="1.0.0"),
                         _broken("broken_step", dist="agentenv-grader", version="0.3.1")],
             env_providers=[_EP("plugin_env", "_PluginEnvProvider", dist="agentenv-browser", version="1.0.0")])
    root = _root_with_cli_plugins(monkeypatch)

    result = _run(root, "list")

    assert result.exit_code == 0
    rows = [line.split() for line in result.output.splitlines() if line.startswith("agentenv-")]
    assert rows[0][:2] == ["agentenv-browser", "1.0.0"] and rows[0][-1] == "ok"
    assert "env browser, task step plugin_step, environment provider plugin_env" in result.output
    assert rows[1][:2] == ["agentenv-grader", "0.3.1"] and rows[1][-2:] == ["1", "failed"]


def test_list_includes_the_cli_plugins_the_cli_loaded(monkeypatch):
    _install(monkeypatch)
    root = _root_with_cli_plugins(
        monkeypatch,
        commands=[_CliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP),
                  _CliEP("plugin", "shadow", cli_plugins_mod.CLI_PLUGINS_GROUP)],
        options=[_CliEP("tenant", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP)],
    )

    result = _run(root, "list")

    (row,) = [line for line in result.output.splitlines() if line.startswith("agentenv-tools")]
    assert "CLI command plugin, CLI command tools, root option tenant" in row
    assert row.endswith("1 skipped")
    assert root.commands["plugin"] is plugin


def test_list_reports_a_conflict_without_taking_its_group_down(monkeypatch):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-browser", version="1.0.0"),
        _EP("browser", "_OtherBrowserEnv", dist="agentenv-web", version="2.1.0"),
        _EP("mcp_server", "_BrowserEnv", dist="agentenv-web", version="2.1.0"),
    ])

    result = _run(_root_with_cli_plugins(monkeypatch), "list")

    assert result.exit_code == 0
    rows = {line.split()[0]: line for line in result.output.splitlines() if line.startswith("agentenv-")}
    assert rows["agentenv-browser"].endswith("1 conflict") and rows["agentenv-web"].endswith("1 conflict, 1 skipped")
    assert "nothing in agent_env.envs loads" not in result.output


def test_list_with_no_plugins_says_so(monkeypatch):
    _install(monkeypatch)

    result = _run(_root_with_cli_plugins(monkeypatch), "list")

    assert result.exit_code == 0 and "No plugins installed." in result.output


def test_list_json_is_the_whole_report(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])

    payload = json.loads(_run(_root_with_cli_plugins(monkeypatch), "list", "--json").stdout)

    assert list(payload) == [
        "format_version", "agent_env", "config", "loaded", "group_errors", "discovery_errors", "plugins",
    ]
    assert payload["format_version"] == 1
    (dist,) = payload["plugins"]
    assert dist["name"] == "agentenv-demo"
    assert dist["contributions"][0] == {
        "group": "agent_env.envs", "name": "browser", "value": f"{_HERE}:_BrowserEnv", "status": "active",
        "code": None, "reason": None, "replaced_by": None, "conflicts_with": [],
    }


def test_a_command_is_listed_run_and_reported_under_its_entry_point_name(monkeypatch):
    root = _root_with_cli_plugins(monkeypatch, commands=[_CliEP("demo", "cli_named_group", cli_plugins_mod.CLI_PLUGINS_GROUP)])

    listing = CliRunner().invoke(root, ["--help"]).output
    ran = CliRunner().invoke(root, ["demo", "--help"])
    payload = json.loads(_run(root, "list", "--json").stdout)

    assert "demo" in listing and "cli " not in listing
    assert ran.exit_code == 0 and "Demo group." in ran.output
    (tools,) = [plugin for plugin in payload["plugins"] if plugin["name"] == "agentenv-tools"]
    (contribution,) = tools["contributions"]
    assert (contribution["name"], contribution["status"]) == ("demo", "active")


def test_list_no_load_imports_no_type_plugin(tmp_path, monkeypatch):
    (tmp_path / "agentenv_cli_demo.py").write_text(
        "from agent_env.env.env import Env\n\n\nclass DemoEnv(Env):\n    type = 'cli_demo'\n\n"
        "    @classmethod\n    def from_dict(cls, data):\n        return cls(data['id'], data.get('version'))\n"
    )
    dist_info = tmp_path / "agentenv_cli_demo-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-cli-demo\nVersion: 0.1.0\n")
    (dist_info / "entry_points.txt").write_text("[agent_env.envs]\ncli_demo = agentenv_cli_demo:DemoEnv\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    try:
        result = _run(_root_with_cli_plugins(monkeypatch), "list", "--no-load")

        (row,) = [line for line in result.output.splitlines() if line.startswith("agentenv-cli-demo")]
        assert row.endswith("not loaded")
        assert "agentenv_cli_demo" not in sys.modules
    finally:
        sys.modules.pop("agentenv_cli_demo", None)


# ---------------------------------------------------------------- show


def test_show_lists_each_contribution_with_why(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")], task_steps=[_broken("broken_step")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_OtherBrowserEnv"]
    """)
    monkeypatch.setattr(sys.modules["agent_env.cli.plugin"], "config_effect", lambda name: "importing it leaves AGENT_ENV_CONFIG unset")

    result = _run(_root_with_cli_plugins(monkeypatch), "show", "agentenv_demo")

    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "agentenv-demo 1.0"
    assert f"replaced: config names '{_HERE}:_OtherBrowserEnv' in [envs] of " in result.output
    assert result.output.count("[replaced-by-config]") == 1
    assert "failed: failed to load: ModuleNotFoundError" in result.output
    assert result.output.count("[load-failed]") == 1
    assert "config:   importing it leaves AGENT_ENV_CONFIG unset" in result.output


def test_show_names_the_installed_packages_when_none_matches(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])

    result = CliRunner().invoke(_root_with_cli_plugins(monkeypatch), ["plugin", "show", "nope"])

    assert result.exit_code == 1
    assert "no installed plugin package named 'nope' (installed: agentenv-demo)" in result.output


def _dist(root: Path, name: str, entry_points: str, requires: tuple[str, ...] = (), files: dict | None = None) -> None:
    """An installed distribution on disk: its dist-info, and its modules."""
    for path, body in (files or {}).items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(body)
    dist_info = root / f"{name.replace('-', '_')}-0.1.0.dist-info"
    dist_info.mkdir()
    lines = ["Metadata-Version: 2.1", f"Name: {name}", "Version: 0.1.0", *(f"Requires-Dist: {r}" for r in requires)]
    (dist_info / "METADATA").write_text("\n".join(lines) + "\n")
    (dist_info / "entry_points.txt").write_text(entry_points)
    (dist_info / "RECORD").write_text("")


def test_the_config_probe_survives_a_plugin_that_prints_and_blames_no_innocent_plugin(tmp_path, monkeypatch):
    _dist(tmp_path, "agentenv-loud", "[agent_env.cli_plugins]\nloud = agentenv_loud:loud\n", files={
        "agentenv_loud/__init__.py": (
            "import atexit, sys, click\nsys.stdout.write('partial')\natexit.register(print, 'bye')\n"
            "import agent_env.cli\n\n\n@click.group()\ndef loud():\n    pass\n"
        ),
    })
    _dist(tmp_path, "agentenv-switcher", "[agent_env.cli_plugins]\nswitch = agentenv_switcher:switch\n", files={
        "agentenv_switcher/__init__.py": (
            "import os, click\nos.environ.setdefault('AGENT_ENV_CONFIG', '/switcher/config.toml')\n\n\n"
            "@click.group()\ndef switch():\n    pass\n"
        ),
    })
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    from agent_env.cli.plugin import config_effect

    assert config_effect("agentenv-loud") == "importing it leaves AGENT_ENV_CONFIG unset"
    assert config_effect("agentenv-switcher") == "importing it sets AGENT_ENV_CONFIG to /switcher/config.toml"


def test_show_finds_the_agent_env_requirement_under_any_spelling(tmp_path, monkeypatch):
    _dist(tmp_path, "agentenv-spelled", "[agent_env.envs]\nspelled = agentenv_spelled:Spelled\n",
          requires=("agentenv_Framework>=0.9.1188", "click>=8"))
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    from agent_env.cli.plugin import _requires_core

    assert _requires_core("agentenv-spelled") == ["agentenv_Framework>=0.9.1188"]


def test_show_detects_a_package_that_sets_agent_env_config_on_import(tmp_path, monkeypatch):
    package = tmp_path / "agentenv_tenant_demo"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import os\nos.environ.setdefault('AGENT_ENV_CONFIG', '/tenant/config.toml')\n"
    )
    (package / "cli.py").write_text("import click\n\n\n@click.group()\ndef tenant_tools():\n    pass\n")
    dist_info = tmp_path / "agentenv_tenant_demo-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-tenant-demo\nVersion: 0.1.0\n")
    (dist_info / "entry_points.txt").write_text("[agent_env.cli_plugins]\ntenant-tools = agentenv_tenant_demo.cli:tenant_tools\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    from agent_env.cli.plugin import config_effect

    assert config_effect("agentenv-tenant-demo") == "importing it sets AGENT_ENV_CONFIG to /tenant/config.toml"
    assert config_effect("agentenv-framework") == "importing it leaves AGENT_ENV_CONFIG unset"


# ---------------------------------------------------------------- check


def test_check_passes_when_every_contribution_is_in_effect(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])

    result = _run(_root_with_cli_plugins(monkeypatch), "check")

    assert result.exit_code == 0
    assert result.output.strip() == "ok: 1 plugin package(s), 1 contribution(s), all in effect"


def test_check_treats_a_config_replacement_as_intended(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_OtherBrowserEnv"]
    """)

    result = _run(_root_with_cli_plugins(monkeypatch), "check")

    assert result.exit_code == 0 and "1 replaced by config" in result.output


def test_check_does_not_fail_on_a_config_warning(monkeypatch, tmp_path):
    """An unread table is `config show`'s to report; `check` gates what plugins contribute."""
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, """
        [sanbox]
        default = "modal"
    """)

    result = _run(_root_with_cli_plugins(monkeypatch), "check")

    assert result.exit_code == 0
    assert result.output.strip() == "ok: 1 plugin package(s), 1 contribution(s), all in effect"


@pytest.mark.parametrize("groups, expected", [
    ({"envs": [_broken("browser")]}, "env browser: failed: failed to load: ModuleNotFoundError"),
    ({"envs": [_EP("website", "_BrowserEnv")]}, "env website: skipped: clashes with a built-in"),
    ({"task_steps": [_EP("bare", "_BareStep")]},
     "task step bare: failed: failed to load: TypeError('_BareStep must implement execute and from_dict') [invalid-plugin]"),
    ({"envs": [_EP("browser", "_BrowserEnv", dist="a"), _EP("browser", "_OtherBrowserEnv", dist="b")]},
     "env browser: conflict: also registered by 'browser' from b 1.0"),
])
def test_check_fails_on_every_contribution_that_did_not_take_effect(monkeypatch, groups, expected):
    _install(monkeypatch, **groups)

    result = CliRunner().invoke(_root_with_cli_plugins(monkeypatch), ["plugin", "check"])

    assert result.exit_code == 1
    assert expected in result.output
    assert "did not take effect" in result.output


def test_check_fails_on_a_status_the_build_did_not_record(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    real_groups = _inventory._groups

    def groups_that_lose_track():
        table = real_groups()
        builtins_of, build = table[_inventory._registration.ENVS]

        def build_then_forget(probe):
            build(probe)
            probe.registrations[_inventory._registration.ENVS].added.pop("browser")

        return {**table, _inventory._registration.ENVS: (builtins_of, build_then_forget)}

    monkeypatch.setattr(_inventory, "_groups", groups_that_lose_track)

    result = CliRunner().invoke(_root_with_cli_plugins(monkeypatch), ["plugin", "check"])

    assert result.exit_code == 1
    assert "env browser: unloaded: its status could not be determined" in result.output


def test_a_command_another_plugin_already_added_names_that_plugin(monkeypatch):
    _install(monkeypatch)
    root = _root_with_cli_plugins(monkeypatch, commands=[
        _CliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP, dist="agentenv-a"),
        _CliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP, dist="agentenv-b"),
    ])

    result = CliRunner().invoke(root, ["plugin", "check"])
    (problem,) = json.loads(CliRunner().invoke(root, ["plugin", "check", "--json"]).stdout)["problems"]

    assert "clashes with existing command 'tools' from plugin 'tools' from 'agentenv-a' [name-conflict]" in result.output
    assert (problem["package"], problem["status"], problem["code"]) == ("agentenv-b", "skipped", "name-conflict")
    assert problem["conflicts_with"] == [{"package": "agentenv-a", "version": "0.2.0", "value": f"{_THIS}:tools"}]


def test_a_root_option_attached_twice_is_still_reported(monkeypatch):
    _install(monkeypatch)
    root = _root_with_cli_plugins(monkeypatch, options=[
        _CliEP("tenant", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP),
        _CliEP("tenant_again", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP),
    ])

    result = _run(root, "show", "agentenv-tools", "--no-load")

    assert "tenant_again" in result.output and "active: the same option object is already attached [already-attached]" in result.output


help_again = click.Option(["--help"], expose_value=False)


@pytest.mark.parametrize(("group", "attr", "status", "code", "reason"), [
    (cli_plugins_mod.CLI_PLUGINS_GROUP, "missing", "failed", "load-failed", "failed to load: AttributeError"),
    (cli_plugins_mod.CLI_PLUGINS_GROUP, "tenant", "failed", "invalid-plugin", "is not a click.Command"),
    (cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, "missing", "failed", "load-failed", "failed to load: AttributeError"),
    (cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, "shadow", "failed", "invalid-plugin", "is not a click.Option"),
    (cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, "help_again", "skipped", "builtin-name",
     "clashes with core root option '--help'"),
])
def test_each_way_a_cli_plugin_is_refused_has_its_code(monkeypatch, group, attr, status, code, reason):
    _install(monkeypatch)
    ep = _CliEP("refused", attr, group)
    commands, options = ([ep], []) if group == cli_plugins_mod.CLI_PLUGINS_GROUP else ([], [ep])
    root = _root_with_cli_plugins(monkeypatch, commands=commands, options=options)

    (problem,) = json.loads(CliRunner().invoke(root, ["plugin", "check", "--json"]).stdout)["problems"]

    assert (problem["group"], problem["status"], problem["code"]) == (group, status, code)
    assert problem["reason"].startswith(reason)


def test_a_command_the_root_group_refuses_to_add_is_failed(monkeypatch):
    class _Refusing(click.Group):
        def add_command(self, cmd, name=None):
            if name == "tools":
                raise RuntimeError("no more commands")
            super().add_command(cmd, name)

    _install(monkeypatch)
    root = _Refusing()
    root.add_command(plugin)
    monkeypatch.setattr(cli_plugins_mod, "entry_points",
                        lambda *, group: [_CliEP("tools", "tools", group)] if group == cli_plugins_mod.CLI_PLUGINS_GROUP else [])
    load_cli_plugins(root)

    (problem,) = json.loads(CliRunner().invoke(root, ["plugin", "check", "--json"]).stdout)["problems"]

    assert (problem["status"], problem["code"]) == ("failed", "invalid-plugin")
    assert problem["reason"] == "could not be registered: RuntimeError('no more commands')"


def test_cli_plugins_whose_agent_env_requirement_is_not_met_are_not_loaded(monkeypatch):
    _install(monkeypatch)
    needs_more = ["agentenv-framework>=999"]
    root = _root_with_cli_plugins(
        monkeypatch,
        commands=[_CliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP, requires=needs_more)],
        options=[_CliEP("tenant", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, requires=needs_more)],
    )

    result = CliRunner().invoke(root, ["plugin", "check"])

    assert "tools" not in root.commands and tenant not in root.params
    assert result.exit_code == 1
    assert result.output.count("failed: needs agentenv-framework>=999 (installed: ") == 2
    assert result.output.count("[incompatible-core]") == 2


def test_one_option_object_two_packages_export_is_attached_once_not_a_conflict(monkeypatch):
    """One object is one callback, so there is no winner to pick between."""
    _install(monkeypatch)
    root = _root_with_cli_plugins(monkeypatch, options=[
        _CliEP("tenant", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, dist="agentenv-a"),
        _CliEP("tenant_too", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP, dist="agentenv-b"),
    ])

    result = CliRunner().invoke(root, ["plugin", "check", "--json"])

    report = json.loads(result.stdout)
    assert result.exit_code == 0 and report["problems"] == []
    assert [(d["name"], c["status"], c["code"]) for d in report["plugins"] for c in d["contributions"]] == [
        ("agentenv-a", "active", None), ("agentenv-b", "active", "already-attached"),
    ]
    assert root.params.count(tenant) == 1


def test_check_fails_on_a_cli_plugin_the_cli_skipped(monkeypatch):
    _install(monkeypatch)
    root = _root_with_cli_plugins(monkeypatch, commands=[_CliEP("plugin", "shadow", cli_plugins_mod.CLI_PLUGINS_GROUP)])

    result = CliRunner().invoke(root, ["plugin", "check"])

    assert result.exit_code == 1
    assert "agentenv-tools 0.2.0: CLI command plugin: skipped: clashes with existing command 'plugin' [builtin-name]" in result.output


@pytest.mark.parametrize("broken", ["missing", "malformed"])
def test_check_fails_when_the_config_cannot_be_read(monkeypatch, tmp_path, broken):
    _install(monkeypatch)
    if broken == "missing":
        monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "missing.toml"))
    else:
        _use_config(monkeypatch, tmp_path, "[envs\n")

    result = CliRunner().invoke(_root_with_cli_plugins(monkeypatch), ["plugin", "check"])

    assert result.exit_code == 1
    assert result.output.rstrip().endswith("Error: the config could not be read")


def test_json_stays_valid_when_a_plugin_prints_while_it_is_imported(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv", on_load=lambda: print("hello from a plugin"))])

    result = _run(_root_with_cli_plugins(monkeypatch), "list", "--json")

    assert json.loads(result.stdout)["plugins"][0]["contributions"][0]["status"] == "active"


def _chatty(ctx, param, value):
    print(f"callback saw {value!r}")


chatty = click.Option(["--chatty"], is_flag=True, default=None, expose_value=False, callback=_chatty)


def test_json_stays_valid_when_a_root_option_callback_prints_with_its_flag_absent(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    root = _root_with_cli_plugins(monkeypatch, options=[_CliEP("chatty", "chatty", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP)])

    absent = CliRunner().invoke(root, ["plugin", "list", "--json"])
    given = CliRunner().invoke(root, ["--chatty", "plugin", "list"])

    assert json.loads(absent.stdout)["format_version"] == 1
    assert "callback saw None" in absent.stderr
    assert "callback saw True" in given.stdout


def test_a_package_with_unreadable_metadata_is_one_row(monkeypatch):
    step = _EP("one", "_PluginStep", dist="", version="")
    step.dist._path = "/site/broken_a-1.0.dist-info"
    _install(monkeypatch, task_steps=[step])
    command = _CliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP, dist="", version="")
    command.dist._path = "/site/broken_a-1.0.dist-info"

    payload = json.loads(_run(_root_with_cli_plugins(monkeypatch, commands=[command]), "list", "--json").stdout)

    (dist,) = payload["plugins"]
    assert dist["name"] == "(unreadable metadata: broken_a-1.0.dist-info)"
    assert [c["name"] for c in dist["contributions"]] == ["one", "tools"]


def test_a_cli_plugin_that_prints_while_it_is_imported_prints_to_stderr(monkeypatch, capsys):
    _root_with_cli_plugins(
        monkeypatch,
        commands=[_NoisyCliEP("tools", "tools", cli_plugins_mod.CLI_PLUGINS_GROUP)],
        options=[_NoisyCliEP("tenant", "tenant", cli_plugins_mod.CLI_ROOT_OPTIONS_GROUP)],
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("hello from a CLI plugin") == 2


def test_check_fails_when_installed_entry_points_cannot_be_read(monkeypatch):
    def unreadable(*, group):
        raise TypeError("Pair.__new__() missing 1 required positional argument: 'value'")

    monkeypatch.setattr(_discovery, "entry_points", unreadable)
    monkeypatch.setattr(cli_plugins_mod, "entry_points", unreadable)
    root = click.Group()
    root.add_command(plugin)
    load_cli_plugins(root)
    load_cli_root_options(root)

    result = _run(root, "check")
    payload = json.loads(CliRunner().invoke(root, ["plugin", "check", "--json"]).stdout)
    listing = _run(root, "list").output

    assert result.exit_code == 1
    assert result.output.count("installed entry points could not be read for agent_env.envs, ") == 1
    assert "agent_env.cli_root_options, agent_env.bundles: TypeError: Pair.__new__()" in result.output
    assert result.output.rstrip().endswith("Error: installed entry points could not be read")
    assert payload["ok"] is False and set(payload["discovery_errors"]) == set(_KINDS)
    assert listing.count("(error) installed entry points could not be read for agent_env.envs, ") == 1


def test_check_json_reports_problems_and_exits_1(monkeypatch):
    _install(monkeypatch, envs=[_broken("browser")])

    result = CliRunner().invoke(_root_with_cli_plugins(monkeypatch), ["plugin", "check", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert list(payload)[0] == "format_version" and payload["ok"] is False
    assert [(p["package"], p["name"], p["status"], p["code"]) for p in payload["problems"]] == [
        ("agentenv-demo", "browser", "failed", "load-failed"),
    ]
    assert payload["plugins"][0]["contributions"][0]["code"] == "load-failed"


def test_the_commands_leave_the_process_config_and_the_working_directory_untouched(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_broken("browser")], task_steps=[_EP("plugin_step", "_PluginStep")])
    root = _root_with_cli_plugins(monkeypatch)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    for args in (["list"], ["list", "--no-load"], ["check"]):
        CliRunner().invoke(root, ["plugin", *args])

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert get_config()._registries == {}


# ---------------------------------------------------------------- environment and the root CLI


def test_environment_says_how_agent_env_was_installed(tmp_path):
    def prefix(name: str, *markers: str) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True)
        for marker in markers:
            (path / marker).write_text("")
        return path

    base = tmp_path / "base"
    assert environment(prefix("tool", "uv-receipt.toml"), base)[0] == "uv tool"
    assert environment(prefix("pipx", "pipx_metadata.json"), base)[0] == "pipx"
    project = tmp_path / "proj"
    (project / ".venv").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    (project / "uv.lock").write_text("")
    assert environment(project / ".venv", base) == ("uv project", project)
    assert environment(prefix("venv"), base)[0] == "virtualenv"
    assert environment(base, base)[0] == "system Python"


def test_core_owns_the_plugin_command_name():
    assert cli.commands["plugin"] is plugin
    assert sorted(plugin.commands) == ["add", "check", "list", "remove", "show"]

