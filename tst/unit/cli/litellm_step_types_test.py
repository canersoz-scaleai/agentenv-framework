from agent_env.cli.task.run import _LITELLM_USING_STEP_TYPES
from agent_env.task_step.registry import _builtin_registry


def test_litellm_step_types_are_built_in_step_types():
    assert _LITELLM_USING_STEP_TYPES <= set(_builtin_registry())
