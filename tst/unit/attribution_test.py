"""`attribution_of` reads a step's `metadata["attribution"]` and nothing else."""

from agent_env.attribution import attribution_of


def _step(**attrs):
    return type("Step", (), attrs)()


def test_attribution_of_is_empty_when_metadata_is_unset():
    assert attribution_of(_step()) == {}
    assert attribution_of(_step(metadata=None)) == {}
    assert attribution_of(_step(metadata={"run_label": "nightly"})) == {}


def test_attribution_of_reads_the_reserved_metadata_key_only():
    step = _step(metadata={"attribution": {"cost_center": "research"}, "run_label": "nightly"})
    assert attribution_of(step) == {"cost_center": "research"}


def test_attribution_of_ignores_flat_attributes_on_the_step():
    step = _step(cost_center="research", metadata={"attribution": {"team": "t"}})
    assert attribution_of(step) == {"team": "t"}


def test_attribution_of_returns_a_copy():
    metadata = {"attribution": {"team": "t"}}
    attribution_of(_step(metadata=metadata))["team"] = "mutated"
    assert metadata["attribution"] == {"team": "t"}
