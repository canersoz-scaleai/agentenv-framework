"""Unit tests for the generic ``reset_env`` task step (Env.reset()).

The reset itself lives in the env's Env.reset override; this only covers
the step's registration, (de)serialization, and required-env_id contract."""
import pytest

from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep


def test_reset_env_step_registered_and_roundtrips():
    assert get_task_step_registry().get("reset_env") is ResetEnvTaskStep
    s = ResetEnvTaskStep(id="r", version=1, env_id="warm-env")
    d = s.to_dict()
    assert d["type"] == "reset_env" and d["env_id"] == "warm-env"
    s2 = ResetEnvTaskStep.from_dict(d)
    assert s2.env_id == "warm-env"
    assert s2.fail_task_on_error is False  # best-effort default


def test_reset_env_requires_env_id():
    with pytest.raises(ValueError):
        ResetEnvTaskStep(id="r", version=1, env_id="")
    with pytest.raises(ValueError):
        ResetEnvTaskStep.from_dict({"id": "r", "version": 1})  # no env_id


def test_reset_env_from_dict_defaults_to_best_effort():
    # A serialized step that omits fail_task_on_error must stay non-fatal (matching
    # __init__'s default), not inherit _base_from_dict's fatal default of True.
    s = ResetEnvTaskStep.from_dict({"id": "r", "version": 1, "env_id": "warm-env"})
    assert s.fail_task_on_error is False
    # An explicit True still round-trips.
    s2 = ResetEnvTaskStep.from_dict(
        {"id": "r", "version": 1, "env_id": "warm-env", "fail_task_on_error": True})
    assert s2.fail_task_on_error is True
