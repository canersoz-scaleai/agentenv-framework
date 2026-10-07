"""A failed agent task is kept with the tail of the agent's container logs, on the local defaults, and a URL the agent
logged is kept only as its scheme and host: an agent may log the grant URLs it was sent. The agent is the echo agent
(tst/data/a2a_agent), told to log a signed URL and fail. Needs Docker and a throwaway local registry."""

import re
import shutil
import socket
import subprocess
import time
import uuid

import httpx
import pytest

from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.task import Task
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.a2a_test_agent import put_test_agent

pytestmark = pytest.mark.integration

REGISTRY_READY_SECONDS = 45


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_registry(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    port = _free_port()
    name = f"failure-logs-registry-{uuid.uuid4().hex[:8]}"
    started = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2")
    assert started.returncode == 0, started.stderr
    host = f"localhost:{port}"
    deadline, ready = time.time() + REGISTRY_READY_SECONDS, False
    while not ready and time.time() < deadline:
        try:
            ready = httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401)
        except httpx.HTTPError:
            pass
        if not ready:
            time.sleep(0.5)
    if not ready:
        _docker("rm", "-f", name)
        pytest.fail(f"the local registry did not answer within {REGISTRY_READY_SECONDS}s")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        # Only this test's containers: every local sandbox it deployed has a work dir in its own sandbox root, and its
        # container is named after that sandbox. Images are shared, so matching by image would reach other runs.
        for work_dir in sandboxes.glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                _docker("rm", "-f", f"agent-{m.group(1)}")
        _docker("rm", "-f", name)
        shutil.rmtree(sandboxes, ignore_errors=True)
        # The agent trusted this process's grant server, whose certificate is from this test's state root; a later
        # test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


@pytest.mark.asyncio
async def test_a_failed_agents_logged_url_is_kept_with_the_run_as_its_host_only(local_registry):
    suffix = uuid.uuid4().hex[:8]
    signature = uuid.uuid4().hex
    signed = f"https://bucket.s3.amazonaws.com/run/x.png?X-Amz-Credential=AKIA{suffix}&X-Amz-Signature={signature}"
    agent = put_test_agent(f"failure-logs-agent-{suffix}")
    task = Task.put(id=f"failure-logs-{suffix}", steps=[
        DeployAgentTaskStep(id="deploy", version=None, env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                            agent_name="solver", sandbox_type="local"),
        PromptAgentTaskStep(id="ask", version=None, prompt_id="ask", agent_name="solver", poll_interval_seconds=1,
                            prompt=f"fail-with fetched {signed}", fail_task_on_error=False),
    ])

    ctx = await task.run()

    (failed,) = ctx.metadata["failed_steps"]
    assert "the prompt asked the agent to fail" in failed["error"] and "[container " in failed["error"]
    assert "fetched https://bucket.s3.amazonaws.com/<redacted>" in failed["error"]
    assert signature not in failed["error"] and f"AKIA{suffix}" not in failed["error"]
