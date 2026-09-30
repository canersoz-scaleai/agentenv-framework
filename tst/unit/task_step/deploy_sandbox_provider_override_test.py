"""A run's ``sandbox`` override moves a deploy_sandbox step to that provider, as ``agent-env run --sandbox`` asks."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider():
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "modal_vm"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock()
    provider.create_vm.return_value = sandbox
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("step_type, overrides, built", [
    ("local", {"sandbox": "modal"}, "modal"),
    ("local", {}, "local"),
    (None, {"sandbox": "modal"}, "modal"),
    ("local", {"env_sandbox": "modal", "agent_sandbox": "modal"}, "local"),
], ids=["override-wins", "step-without-override", "override-without-step-type", "other-overrides-ignored"])
async def test_the_sandbox_override_picks_the_provider(step_type, overrides, built):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm", sandbox_type=step_type)
    provider = _provider()
    context = TaskStepContext(metadata={"user_overrides": overrides})

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider) as build:
        await step.execute(context)

    build.assert_called_once_with(built)
    assert context.deployed_sandboxes[0].sandbox_type == "modal_vm"


@pytest.mark.asyncio
async def test_without_either_the_configured_default_deploys_it():
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm")
    provider = _provider()

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider) as default:
        await step.execute(TaskStepContext())

    default.assert_called_once_with()
