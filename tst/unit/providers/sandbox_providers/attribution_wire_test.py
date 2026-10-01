"""Byte-identity net for the emitted compute-attribution wire.

Attribution lands in the Modal App `tags` dict (the `labels` sink of the out-of-tree platform
sandbox provider is pinned by that provider's tests). These pin what comes out, so a refactor of
how attribution is carried announces itself instead of quietly re-bucketing spend.

Precedence is `explicit arg -> config.toml [sandbox.attribution]`, tested `is not None`: an
empty string is a value.
"""

from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.modal_sandbox import _build_cost_attribution_tags

_DEFAULTS = {"product": "p0", "customer": "c0", "team": "t0"}


def _use_config(tmp_path, monkeypatch, attribution: dict[str, str]) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text("[sandbox.attribution]\n" + "".join(f'{k} = "{v}"\n' for k, v in attribution.items()))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))


@pytest.fixture(autouse=True)
def _configured_defaults(tmp_path, monkeypatch):
    # Pin config discovery to a fixture so no ambient .agentenv leaks into the assertions.
    _use_config(tmp_path, monkeypatch, _DEFAULTS)


def _tags(**kwargs):
    return _build_cost_attribution_tags(kwargs)


# --- the emitted dicts, exactly -------------------------------------------------


def test_modal_tags_with_nothing_supplied():
    assert _tags() == {"product": "p0", "customer": "c0", "team": "t0"}


def test_modal_tags_with_everything_supplied():
    assert _tags(product="p", customer="c", team="t") == {"product": "p", "customer": "c", "team": "t"}


def test_modal_vm_shares_the_modal_sink():
    from agent_env.providers.sandbox_providers import modal_vm_sandbox

    assert modal_vm_sandbox._build_cost_attribution_tags is _build_cost_attribution_tags



# --- precedence -----------------------------------------------------------------


@pytest.mark.parametrize("field", ["product", "customer", "team"])
def test_explicit_beats_the_configured_default(field):
    assert _tags(**{field: "explicit"})[field] == "explicit"


def test_a_configured_default_may_be_an_env_reference(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {**_DEFAULTS, "product": "env:ATTRIBUTION_TEST_PRODUCT?p0"})
    monkeypatch.delenv("ATTRIBUTION_TEST_PRODUCT", raising=False)
    assert _tags()["product"] == "p0"
    monkeypatch.setenv("ATTRIBUTION_TEST_PRODUCT", "from-env")
    assert _tags()["product"] == "from-env"


def test_without_configured_defaults_modal_omits_unset_dimensions(tmp_path, monkeypatch):
    _use_config(tmp_path, monkeypatch, {})
    assert _tags() == {}
    assert _tags(team="t") == {"team": "t"}


@pytest.mark.parametrize("field", ["product", "customer", "team"])
def test_an_empty_string_is_a_value_not_an_absence(field):
    """`is not None`, not truthiness — pinned because flipping it is silent on Modal."""
    assert _tags(**{field: ""})[field] == ""


# --- the dict is the only carried form -------------------------------------------


@pytest.mark.asyncio
async def test_a_chain_forwards_the_dict_as_is():
    """ChainedSandboxProvider takes **kwargs; the dict reaches the backend untouched, open
    keys included."""
    from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider

    seen = {}

    class Provider:
        type = "modern"

        async def create_sandbox(self, *, attribution=None, **kwargs):
            seen["attribution"] = attribution
            seen["kwargs"] = kwargs
            return SimpleNamespace(type="modern", network_policy=None)

    attribution = {"team": "t"}
    await ChainedSandboxProvider([Provider()]).create_sandbox(attribution=attribution, cpu=2.0)
    assert seen == {"attribution": attribution, "kwargs": {"cpu": 2.0}}


@pytest.mark.asyncio
async def test_the_quartet_is_no_longer_a_keyword_on_any_backend():
    """A caller still on the pre-dict signature fails loud rather than billing unattributed."""
    from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider

    with pytest.raises(TypeError, match="team"):
        await LocalSandboxProvider().create_vm(team="t")
