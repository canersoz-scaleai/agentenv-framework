"""A deploy_sandbox step that sets no cpu reserves one vCPU, whether built in code or loaded from a stored dict, and
a stored cpu is kept as written."""

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


def _stored(**fields) -> dict:
    return {"id": "s", "version": 1, "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", **fields}


@pytest.mark.parametrize("step", [
    DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm"),
    DeploySandboxTaskStep.from_dict(_stored()),
], ids=["constructed", "from-dict"])
def test_an_unsized_step_reserves_one_cpu(step):
    assert step.cpu == 1.0


def test_a_stored_cpu_is_kept():
    assert DeploySandboxTaskStep.from_dict(_stored(cpu=2.0)).cpu == 2.0


@pytest.mark.asyncio
async def test_the_provider_is_asked_for_one_cpu():
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm", sandbox_type="modal_vm")
    provider = _provider()

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    assert provider.create_vm.call_args.kwargs["cpu"] == 1.0
