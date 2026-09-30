"""``agent_env.plugins.inventory()``: every installed plugin's contributions and their status.

The registries are built on a throwaway Config reading the caller's document, so the Config in use
keeps its registries and its ``load_failures()``. A conflict is reported for both claimants, and the
rest of the group loads, as in a real build.
"""

import importlib
import sys

import pytest
from fastapi import APIRouter

from agent_env import plugins
from agent_env.config import get_config, reset_config
from agent_env.explorer.plugin import ExplorerPlugin, load_plugins
from agent_env.plugins import Claimant, Replacement, _inventory, inventory
from tst.unit.plugins_test import _EP, _HERE, _Env, _ReentrantEP, _fresh, _install, _use_config  # noqa: F401  (_fresh: autouse)

_THIS = "tst.unit.plugin_inventory_test"


class _ExtraEnv(_Env):
    type = "extra_env"


class _ExitingRoutes(ExplorerPlugin):
    type = "exiting_routes"

    def __init__(self):
        raise SystemExit(3)

    @property
    def router(self) -> APIRouter:
        return APIRouter()


def _ep(name: str, attr: str, **kwargs) -> _EP:
    """An entry point whose class lives in this module rather than in plugins_test."""
    ep = _EP(name, attr, **kwargs)
    ep.value = f"{_THIS}:{attr}"
    return ep


def _broken(name: str, **kwargs) -> _EP:
    ep = _EP(name, "_BrowserEnv", **kwargs)
    ep.value = "agentenv_broken.nowhere:Missing"
    return ep


def _contribution(inv, group: str, name: str, dist: str = "agentenv-demo"):
    (found,) = [
        c for d in inv.distributions if d.name == dist for c in d.contributions if (c.group, c.name) == (group, name)
    ]
    return found


# ---------------------------------------------------------------- one status per shape


def test_a_plugin_in_every_group_is_active(monkeypatch):
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv")],
        task_steps=[_EP("plugin_step", "_PluginStep")],
        artifacts=[_EP("plugin_art", "_PluginArtifact")],
        sandbox_providers=[_EP("plugin_box", "_OtherRecordingProvider")],
        state_providers=[_EP("plugin_state", "_PluginState")],
        explorer_plugins=[_EP("plugin_routes", "_Routes")],
    )

    inv = inventory()

    (dist,) = inv.distributions
    assert (dist.name, dist.version) == ("agentenv-demo", "1.0")
    assert [(c.group, c.name, c.status) for c in dist.contributions] == [
        (plugins.ENVS, "browser", "active"),
        (plugins.ARTIFACTS, "plugin_art", "active"),
        (plugins.TASK_STEPS, "plugin_step", "active"),
        (plugins.SANDBOX_PROVIDERS, "plugin_box", "active"),
        (plugins.STATE_PROVIDERS, "plugin_state", "active"),
        (plugins.EXPLORER_PLUGINS, "plugin_routes", "active"),
    ]
    assert inv.loaded and not inv.group_errors and inv.config_error is None


def test_a_plugin_that_fails_to_import_is_failed_with_the_error(monkeypatch):
    _install(monkeypatch, envs=[_broken("browser")])

    found = _contribution(inventory(), plugins.ENVS, "browser")

    assert (found.status, found.code) == ("failed", "load-failed")
    assert found.reason.startswith("failed to load: ModuleNotFoundError")


@pytest.mark.parametrize("load", [True, False])
def test_a_plugin_whose_agent_env_requirement_is_not_met_is_failed_without_importing_it(monkeypatch, load):
    def imported():
        raise AssertionError("the plugin was imported")

    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv", on_load=imported, requires=["agentenv-framework>=999"])])

    found = _contribution(inventory(load=load), plugins.ENVS, "browser")

    assert (found.status, found.code) == ("failed", "incompatible-core")
    assert found.reason.startswith("needs agentenv-framework>=999 (installed: ")


@pytest.mark.parametrize("load", [True, False])
def test_a_claim_that_cannot_load_takes_no_part_in_a_conflict(monkeypatch, load):
    needs_more = ["agentenv-framework>=999"]
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-a"),
        _EP("browser", "_OtherBrowserEnv", dist="agentenv-b", requires=needs_more),
    ], task_steps=[
        _EP("plugin_step", "_PluginStep", dist="agentenv-c"),
        _EP("plugin_step", "_PluginStep", dist="agentenv-d"),
        _EP("plugin_step", "_PluginStep", dist="agentenv-e", requires=needs_more),
    ])

    inv = inventory(load=load)

    statuses = {(d.name, c.name): (c.status, c.code) for d in inv.distributions for c in d.contributions}
    assert statuses[("agentenv-a", "browser")] == (("active", None) if load else ("unloaded", "not-loaded"))
    assert statuses[("agentenv-b", "browser")] == ("failed", "incompatible-core")
    assert plugins.ENVS not in inv.group_errors
    assert statuses[("agentenv-c", "plugin_step")] == statuses[("agentenv-d", "plugin_step")] == ("conflict", "name-conflict")
    assert statuses[("agentenv-e", "plugin_step")] == ("failed", "incompatible-core")
    conflicting = _contribution(inv, plugins.TASK_STEPS, "plugin_step", "agentenv-c")
    assert [c.package for c in conflicting.conflicts_with] == ["agentenv-d"]
    assert plugins.TASK_STEPS not in inv.group_errors
    assert "agentenv-e" not in conflicting.reason


def test_a_plugin_that_fails_validation_is_failed(monkeypatch):
    _install(monkeypatch, envs=[_EP("not_browser", "_BrowserEnv")])

    found = _contribution(inventory(), plugins.ENVS, "not_browser")

    assert (found.status, found.code) == ("failed", "invalid-plugin")
    assert "its entry point must be named 'browser'" in found.reason


def test_a_plugin_that_leaves_a_required_method_unimplemented_is_failed(monkeypatch):
    _install(monkeypatch, task_steps=[_EP("bare", "_BareStep")])

    found = _contribution(inventory(), plugins.TASK_STEPS, "bare")
    unloaded = _contribution(inventory(load=False), plugins.TASK_STEPS, "bare")

    assert (found.status, found.code) == ("failed", "invalid-plugin")
    assert "_BareStep must implement execute and from_dict" in found.reason
    assert (unloaded.status, unloaded.code) == ("unloaded", "not-loaded")


def test_a_built_in_name_is_skipped_for_every_claimant_before_any_conflict(monkeypatch):
    _install(monkeypatch, envs=[_EP("website", "_BrowserEnv", dist="a"), _EP("website", "_BrowserEnv", dist="b")])

    inv = inventory()

    assert [(c.status, c.code) for c in (_contribution(inv, plugins.ENVS, "website", dist) for dist in ("a", "b"))] == [
        ("skipped", "builtin-name"), ("skipped", "builtin-name"),
    ]
    assert not inv.group_errors


def test_an_explorer_plugin_that_fails_to_construct_is_failed(monkeypatch):
    _install(monkeypatch, explorer_plugins=[_EP("broken_routes", "_BrokenCtorRoutes")])

    found = _contribution(inventory(), plugins.EXPLORER_PLUGINS, "broken_routes")

    assert (found.status, found.code) == ("failed", "load-failed")
    assert found.reason == "failed to construct: RuntimeError('boom in ctor')"


def test_an_explorer_plugin_that_exits_while_constructed_is_failed_not_fatal(monkeypatch):
    _install(monkeypatch, explorer_plugins=[_ep("exiting_routes", "_ExitingRoutes")])

    found = _contribution(inventory(), plugins.EXPLORER_PLUGINS, "exiting_routes")

    assert (found.status, found.code, found.reason) == ("failed", "load-failed", "failed to construct: SystemExit(3)")


def test_an_artifact_legacy_name_config_strands_is_failed(monkeypatch, tmp_path):
    _install(monkeypatch, artifacts=[_EP("plugin_art", "_PluginArtifact"), _EP("plugin_art_legacy", "_PluginArtifact")])
    _use_config(monkeypatch, tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_OtherPluginArtifact"]
    """)

    inv = inventory()

    assert _contribution(inv, plugins.ARTIFACTS, "plugin_art").status == "replaced"
    stranded = _contribution(inv, plugins.ARTIFACTS, "plugin_art_legacy")
    assert stranded.status == "failed" and "resolves to _OtherPluginArtifact" in stranded.reason
    assert stranded.code == "invalid-plugin"


# ---------------------------------------------------------------- config taking a name over


def test_config_naming_a_different_class_replaces_the_plugin_and_says_where(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_OtherBrowserEnv"]
    """)

    found = _contribution(inventory(), plugins.ENVS, "browser")

    config = tmp_path / ".agentenv" / "config.toml"
    assert (found.status, found.code) == ("replaced", "replaced-by-config")
    assert found.replaced_by == Replacement(config, "envs", f"{_HERE}:_OtherBrowserEnv")
    assert found.reason == f"config names '{_HERE}:_OtherBrowserEnv' in [envs] of {config}"


def test_config_naming_the_plugins_own_class_leaves_it_active(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_BrowserEnv"]
    """)

    assert _contribution(inventory(), plugins.ENVS, "browser").status == "active"


def test_a_provider_table_with_impl_replaces_a_provider_plugin(monkeypatch, tmp_path):
    _install(monkeypatch, sandbox_providers=[_EP("plugin_box", "_OtherRecordingProvider")])
    _use_config(monkeypatch, tmp_path, f"""
        [sandbox.providers.plugin_box]
        impl = "{_HERE}:_GuardedProvider"
    """)

    found = _contribution(inventory(), plugins.SANDBOX_PROVIDERS, "plugin_box")

    assert (found.status, found.code) == ("replaced", "replaced-by-config")
    assert found.replaced_by == Replacement(
        tmp_path / ".agentenv" / "config.toml", "sandbox.providers.plugin_box", f"{_HERE}:_GuardedProvider"
    )


def test_a_config_only_provider_table_configures_the_plugin_without_replacing_it(monkeypatch, tmp_path):
    _install(monkeypatch, state_providers=[_EP("plugin_state", "_PluginState")])
    _use_config(monkeypatch, tmp_path, """
        [state.providers.plugin_state.config]
        secret_name = "demo"
    """)

    assert _contribution(inventory(), plugins.STATE_PROVIDERS, "plugin_state").status == "active"


# ---------------------------------------------------------------- conflicts are strict, and contained to the name


def test_a_conflict_is_reported_and_the_rest_of_its_group_loads(monkeypatch):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="agentenv-browser"),
        _EP("browser", "_OtherBrowserEnv", dist="agentenv-web", version="2.1.0"),
        _ep("extra_env", "_ExtraEnv", dist="agentenv-browser"),
    ], task_steps=[_EP("plugin_step", "_PluginStep", dist="agentenv-web", version="2.1.0")])

    inv = inventory()

    browser = _contribution(inv, plugins.ENVS, "browser", "agentenv-browser")
    assert (browser.status, browser.code) == ("conflict", "name-conflict")
    assert browser.conflicts_with == (Claimant("agentenv-web", "2.1.0", f"{_HERE}:_OtherBrowserEnv"),)
    assert browser.reason == (f"also registered by 'browser' from agentenv-web 2.1.0 ({_HERE}:_OtherBrowserEnv); "
                              "remove all but one with `agent-env plugin remove <package>`")
    assert _contribution(inv, plugins.ENVS, "browser", "agentenv-web").status == "conflict"
    assert _contribution(inv, plugins.ENVS, "extra_env", "agentenv-browser").status == "active"
    assert _contribution(inv, plugins.TASK_STEPS, "plugin_step", "agentenv-web").status == "active"
    assert not inv.group_errors
    registry = get_config().env_registry()
    assert "browser" not in registry and registry["extra_env"] is _ExtraEnv


def test_without_loading_conflicts_and_built_in_clashes_are_still_reported(monkeypatch):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="a"),
        _EP("browser", "_OtherBrowserEnv", dist="b"),
        _ep("extra_env", "_ExtraEnv", dist="a"),
        _EP("website", "_BrowserEnv", dist="a"),
    ], task_steps=[_EP("plugin_step", "_PluginStep", dist="a")])

    inv = inventory(load=False)

    assert not inv.loaded
    assert _contribution(inv, plugins.ENVS, "browser", "a").status == "conflict"
    assert _contribution(inv, plugins.ENVS, "extra_env", "a").status == "unloaded"
    assert not inv.group_errors
    assert _contribution(inv, plugins.ENVS, "website", "a").status == "skipped"
    step = _contribution(inv, plugins.TASK_STEPS, "plugin_step", "a")
    assert (step.status, step.code) == ("unloaded", "not-loaded")


# ---------------------------------------------------------------- isolation from the Config in use


def test_the_config_in_use_keeps_its_registries_and_its_failures(monkeypatch):
    _install(monkeypatch, envs=[_broken("browser")], task_steps=[_EP("plugin_step", "_PluginStep")])
    config = get_config()

    inventory(config)

    assert plugins.load_failures(config) == {}
    assert config._registries == {}


def test_the_callers_pinned_document_is_read_even_if_a_plugin_repoints_the_config(monkeypatch, tmp_path):
    elsewhere = tmp_path / "elsewhere.toml"
    elsewhere.write_text("")
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", on_load=lambda: monkeypatch.setenv("AGENT_ENV_CONFIG", str(elsewhere)))
    ])
    config = get_config()
    pinned = config.config_path()

    inv = inventory(config)

    assert inv.config_path == pinned != elsewhere


def test_a_plugin_that_reenters_its_registry_on_import_leaves_no_failure_behind(monkeypatch):
    _install(monkeypatch, envs=[_ReentrantEP("browser", "_BrowserEnv", reset=False)])
    config = get_config()

    inv = inventory(config)

    assert _contribution(inv, plugins.ENVS, "browser").status == "active"
    assert plugins.load_failures(config) == {}


def test_without_loading_no_plugin_module_is_imported(tmp_path, monkeypatch):
    (tmp_path / "agentenv_inventory_demo.py").write_text(
        "from agent_env.env.env import Env\n\n\nclass DemoEnv(Env):\n    type = 'inventory_demo'\n\n"
        "    @classmethod\n    def from_dict(cls, data):\n        return cls(data['id'], data.get('version'))\n"
    )
    dist_info = tmp_path / "agentenv_inventory_demo-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-inventory-demo\nVersion: 0.1.0\n")
    (dist_info / "entry_points.txt").write_text("[agent_env.envs]\ninventory_demo = agentenv_inventory_demo:DemoEnv\n")
    (dist_info / "RECORD").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    try:
        unloaded = _contribution(inventory(load=False), plugins.ENVS, "inventory_demo", "agentenv-inventory-demo")

        assert (unloaded.status, unloaded.value) == ("unloaded", "agentenv_inventory_demo:DemoEnv")
        assert "agentenv_inventory_demo" not in sys.modules
        assert _contribution(inventory(), plugins.ENVS, "inventory_demo", "agentenv-inventory-demo").status == "active"
    finally:
        sys.modules.pop("agentenv_inventory_demo", None)


# ---------------------------------------------------------------- errors are reported, not raised


def test_a_config_file_that_cannot_be_found_blocks_every_group_but_plugins_are_still_checked(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")], task_steps=[_broken("broken_step")])
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "missing.toml"))

    inv = inventory()

    assert inv.config_error.code == "config-not-found" and "missing.toml" in inv.config_error.reason
    assert inv.config_path is None
    browser = _contribution(inv, plugins.ENVS, "browser")
    assert (browser.status, browser.code) == ("blocked", "config-not-found") and "missing.toml" in browser.reason
    assert _contribution(inv, plugins.TASK_STEPS, "broken_step").status == "failed"


def test_a_malformed_config_file_is_reported_even_with_no_plugins(monkeypatch, tmp_path):
    _install(monkeypatch)
    _use_config(monkeypatch, tmp_path, "[envs\n")

    inv = inventory()

    assert inv.config_error.code == "config-unreadable" and "Malformed" in inv.config_error.reason
    assert inv.config_path == tmp_path / ".agentenv" / "config.toml"


def test_a_malformed_config_file_blocks_what_its_real_build_would_not_reach(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="a"), _EP("browser", "_OtherBrowserEnv", dist="b"),
    ], task_steps=[_EP("plugin_step", "_PluginStep")])
    _use_config(monkeypatch, tmp_path, "[envs\n")

    inv = inventory()

    # A conflict is no group error, so the file's is, as in a real build.
    envs, steps = inv.group_errors[plugins.ENVS], inv.group_errors[plugins.TASK_STEPS]
    assert envs == steps and steps.code == "config-unreadable" and steps.reason.startswith("ConfigError: Malformed")
    assert _contribution(inv, plugins.ENVS, "browser", "a").status == "conflict"
    assert _contribution(inv, plugins.TASK_STEPS, "plugin_step").code == "config-unreadable"


_REAL_BUILDS = {
    plugins.ENVS: lambda config: config.env_registry(),
    plugins.TASK_STEPS: lambda config: config.task_step_registry(),
    plugins.ARTIFACTS: lambda config: config.artifact_registry(),
    plugins.SANDBOX_PROVIDERS: lambda config: config.sandbox_registry(),
    plugins.STATE_PROVIDERS: lambda config: config.state_registry(),
    plugins.EXPLORER_PLUGINS: lambda config: load_plugins(source=config),
}


@pytest.mark.parametrize(
    "config_body",
    [None, "", "[envs\n", f'[envs]\nimpls = ["{_HERE}:_OtherBrowserEnv"]\n', '[explorer.plugins]\nimpls = "x"\n',
     f'[sandbox.providers.plugin_box]\nimpl = "{_HERE}:_RecordingProvider"\n',
     '[artifacts.type_aliases]\nplugin_art = "file"\n', '[artifacts.type_aliases]\nplugin_art_old = "plugin_art"\n',
     '[state.providers.plugin_state.config]\nsecret_name = "demo"\n'],
    ids=["no-file", "empty", "malformed", "config-names-the-conflicted-class", "explorer-reads-config-first",
         "provider-table-names-the-conflicted-name", "alias-from-the-conflicted-name", "alias-to-the-conflicted-name",
         "state-table-names-the-conflicted-name"],
)
def test_group_errors_are_the_errors_a_real_build_raises(monkeypatch, tmp_path, config_body):
    _install(
        monkeypatch,
        envs=[_EP("browser", "_BrowserEnv", dist="a"), _EP("browser", "_OtherBrowserEnv", dist="b")],
        task_steps=[_EP("plugin_step", "_PluginStep")],
        # The explorer reads its config before merging, the others after: each order is checked.
        sandbox_providers=[_EP("plugin_box", "_RecordingProvider", dist="a"),
                           _EP("plugin_box", "_OtherRecordingProvider", dist="b")],
        artifacts=[_EP("plugin_art", "_PluginArtifact", dist="a"), _EP("plugin_art", "_OtherPluginArtifact", dist="b")],
        state_providers=[_EP("plugin_state", "_PluginState", dist="a"), _EP("plugin_state", "_OtherPluginState", dist="b")],
        explorer_plugins=[_EP("plugin_routes", "_Routes", dist="a"), _EP("plugin_routes", "_OtherRoutes", dist="b")],
    )
    if config_body is None:
        monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "missing.toml"))
        reset_config()
    else:
        _use_config(monkeypatch, tmp_path, config_body)

    inv = inventory()

    for group, build in _REAL_BUILDS.items():
        reset_config()
        try:
            build(get_config())
        except Exception as exc:
            expected = f"{type(exc).__name__}: {exc}"
        else:
            expected = None
        error = inv.group_errors.get(group)
        assert (error.reason if error else None) == expected, group


def test_without_loading_a_malformed_config_blocks_nothing(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[
        _EP("browser", "_BrowserEnv", dist="a"), _EP("browser", "_OtherBrowserEnv", dist="b"),
    ], task_steps=[_EP("plugin_step", "_PluginStep")])
    _use_config(monkeypatch, tmp_path, "[envs\n")

    inv = inventory(load=False)

    assert "Malformed" in inv.config_error.reason
    assert not inv.group_errors
    assert _contribution(inv, plugins.ENVS, "browser", "a").status == "conflict"
    step = _contribution(inv, plugins.TASK_STEPS, "plugin_step")
    assert (step.status, step.code) == ("unloaded", "not-loaded")


def test_a_status_the_build_did_not_record_is_visible_not_silent(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    real_groups = _inventory._groups

    def groups_that_lose_track():
        table = real_groups()
        builtins_of, build = table[plugins.ENVS]

        def build_then_forget(probe):
            build(probe)
            probe.registrations[plugins.ENVS].added.pop("browser")

        return {**table, plugins.ENVS: (builtins_of, build_then_forget)}

    monkeypatch.setattr(_inventory, "_groups", groups_that_lose_track)

    found = _contribution(inventory(), plugins.ENVS, "browser")

    assert (found.status, found.code, found.reason) == ("unloaded", "status-unknown", "its status could not be determined")


def test_a_config_error_in_a_group_blocks_its_plugins(monkeypatch, tmp_path):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, """
        [envs]
        impls = "not-a-list"
    """)

    inv = inventory()

    assert inv.group_errors[plugins.ENVS].code == "config-invalid"
    assert inv.group_errors[plugins.ENVS].reason.startswith("ConfigError: [envs] impls must be a list")
    assert _contribution(inv, plugins.ENVS, "browser").status == "blocked"


def test_a_config_impl_that_exits_while_imported_blocks_its_group_not_the_report(monkeypatch, tmp_path):
    (tmp_path / "agentenv_exits_demo.py").write_text("raise SystemExit('exits on import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv")])
    _use_config(monkeypatch, tmp_path, """
        [envs]
        impls = ["agentenv_exits_demo:Anything"]
    """)

    inv = inventory()

    assert inv.group_errors[plugins.ENVS].code == "group-build-failed"
    assert _contribution(inv, plugins.ENVS, "browser").code == "group-build-failed"


def test_packages_with_unreadable_metadata_are_not_merged(monkeypatch):
    first, second = _EP("one", "_PluginStep", dist="", version=""), _EP("two", "_PluginStep", dist="", version="")
    first.dist._path, second.dist._path = "/site/broken_a-1.0.dist-info", "/site/broken_b-2.0.dist-info"
    _install(monkeypatch, task_steps=[first, second])

    names = [d.name for d in inventory().distributions]

    assert names == ["(unreadable metadata: broken_a-1.0.dist-info)", "(unreadable metadata: broken_b-2.0.dist-info)"]


def test_entry_points_that_cannot_be_read_are_reported_with_the_package_at_fault(tmp_path, monkeypatch):
    info = tmp_path / "site" / "agentenv_torn-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: agentenv-torn\nVersion: 1.0\n")
    (info / "entry_points.txt").write_text("[agent_env.envs]\nno equals sign\n")
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    inv = inventory()

    assert set(inv.discovery_errors) == {
        plugins.ENVS, plugins.ARTIFACTS, plugins.TASK_STEPS,
        plugins.SANDBOX_PROVIDERS, plugins.STATE_PROVIDERS, plugins.ENV_PROVIDERS, plugins.EXPLORER_PLUGINS,
    }
    assert inv.discovery_errors[plugins.ENVS].code == "entry-points-unreadable"
    assert str(info) in inv.discovery_errors[plugins.ENVS].reason
    assert inv.distributions == ()


def test_a_name_one_package_declares_twice_says_so(monkeypatch):
    _install(monkeypatch, envs=[_EP("browser", "_BrowserEnv"), _EP("browser", "_OtherBrowserEnv")])

    found = [c for d in inventory().distributions for c in d.contributions]

    assert {(c.status, c.code) for c in found} == {("conflict", "name-conflict")}
    assert all(c.reason.startswith("this package declares the name more than once: 'browser' from agentenv-demo 1.0")
               for c in found)
    assert [c.conflicts_with[0].value for c in found] == [f"{_HERE}:_OtherBrowserEnv", f"{_HERE}:_BrowserEnv"]


def test_no_plugins_installed_is_an_empty_inventory(monkeypatch):
    _install(monkeypatch)

    inv = inventory()

    assert inv.distributions == () and not inv.group_errors


def test_the_public_types_name_the_public_module():
    for public in (plugins.Inventory, plugins.Distribution, plugins.Contribution, plugins.Diagnostic, plugins.Claimant,
                   plugins.Replacement):
        assert public.__module__ == "agent_env.plugins"
