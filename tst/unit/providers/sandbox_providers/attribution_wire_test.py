"""What attribution emits on the Modal wire.

The deployment-wide config.toml ``[sandbox.attribution]`` defaults are the Modal App tags; a
run's attribution, with unset keys filled from those defaults, is its sandbox tags. Neither names
a key: any dimension a deployment uses passes through. Precedence is
``explicit value -> [sandbox.attribution]``, tested ``is not None``.
"""

from types import SimpleNamespace

import pytest

from agent_env.attribution import PIPELINE_STEP_KEY, RUN_ID_KEY
from agent_env.providers.sandbox_providers.modal_sandbox import _build_app_tags, _build_sandbox_tags


def _use_config(tmp_path, monkeypatch, attribution: dict[str, str]) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox.attribution]\n" + "".join(f'{k} = "{v}"\n' for k, v in attribution.items()))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))


@pytest.fixture(autouse=True)
def _configured_defaults(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {"group": "g0", "owner": "o0"})


# --- App tags: the configured defaults only ------------------------------------------


def test_app_tags_are_the_configured_defaults():
    assert _build_app_tags() == {"group": "g0", "owner": "o0"}


def test_app_tags_without_configured_defaults_are_empty(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {})
    assert _build_app_tags() == {}


def test_a_configured_default_may_be_an_env_reference(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {"group": "env:ATTRIBUTION_TEST_GROUP?g0"})
    monkeypatch.delenv("ATTRIBUTION_TEST_GROUP", raising=False)
    assert _build_app_tags() == {"group": "g0"}
    monkeypatch.setenv("ATTRIBUTION_TEST_GROUP", "from-env")
    assert _build_app_tags() == {"group": "from-env"}


def test_modal_vm_shares_the_modal_builders():
    from agent_env.providers.sandbox_providers import modal_vm_sandbox

    assert modal_vm_sandbox._build_app_tags is _build_app_tags
    assert modal_vm_sandbox._build_sandbox_tags is _build_sandbox_tags


# --- sandbox tags: the run's attribution over the defaults ----------------------------


def test_any_attribution_key_reaches_the_sandbox_tags():
    tags = _build_sandbox_tags({"batch": "b1", PIPELINE_STEP_KEY: "t_s", RUN_ID_KEY: "inst-1"})
    assert tags == {"batch": "b1", "group": "g0", "owner": "o0", PIPELINE_STEP_KEY: "t_s", RUN_ID_KEY: "inst-1"}


def test_an_explicit_value_beats_the_configured_default():
    assert _build_sandbox_tags({"group": "explicit"})["group"] == "explicit"


def test_an_empty_string_keeps_the_default_out_and_emits_no_tag():
    """``is not None``: an explicit empty value is not replaced by the default, and Modal has no
    empty tag, so the key is left out."""
    assert "group" not in _build_sandbox_tags({"group": ""})


def test_no_attribution_and_no_defaults_means_no_tags(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {})
    assert _build_sandbox_tags({}) == {}


def test_keys_and_values_are_made_modal_valid():
    tags = _build_sandbox_tags({"my key/v2": "a value/with spaces", "long": "x" * 80})
    assert tags["my-key-v2"] == "a-value-with-spaces"
    assert len(tags["long"]) == 63


# --- the dict is the only carried form -------------------------------------------------


@pytest.mark.asyncio
async def test_a_chain_forwards_the_dict_as_is():
    """ChainedSandboxProvider takes **kwargs; the dict reaches the backend untouched."""
    from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider

    seen = {}

    class Provider:
        type = "modern"

        async def create_sandbox(self, *, attribution=None, **kwargs):
            seen["attribution"] = attribution
            seen["kwargs"] = kwargs
            return SimpleNamespace(type="modern", network_policy=None)

    attribution = {"group": "g"}
    await ChainedSandboxProvider([Provider()]).create_sandbox(attribution=attribution, cpu=2.0)
    assert seen == {"attribution": attribution, "kwargs": {"cpu": 2.0}}


@pytest.mark.asyncio
async def test_the_quartet_is_no_longer_a_keyword_on_any_backend():
    """A caller still on the pre-dict signature fails loud rather than billing unattributed."""
    from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider

    with pytest.raises(TypeError, match="team"):
        await LocalSandboxProvider().create_vm(team="t")
