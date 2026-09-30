"""A task scores against a bare local sandbox with no Docker, model or network.

A local VM-mode sandbox is a host work dir driven by subprocesses, so this needs `bash`
and `python3` on PATH and nothing else.
"""

import shutil
import uuid

import pytest

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.task import Task
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep

pytestmark = pytest.mark.integration

_CHECK_PY = "import pathlib, sys\nsys.exit(0 if 'hello' in pathlib.Path('hello.txt').read_text() else 1)\n"


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path, sandboxes
    finally:
        shutil.rmtree(sandboxes, ignore_errors=True)
        reset_artifact_store()
        reset_config()


def _greeting_universe(src_dir, suffix):
    src_dir.mkdir()
    (src_dir / "hello.txt").write_text("hello, world\n")
    (src_dir / "check.py").write_text(_CHECK_PY)
    return FileArtifactUniverse.put(
        id=f"greeting-{suffix}",
        file_artifacts={
            name: FileArtifact.put(id=f"greeting-{suffix}-{name}", description=name, file_path=str(src_dir / name))
            for name in ("hello.txt", "check.py")
        },
    )


def _verify(verifier_id, expected, bash_cmd):
    return VerifySandboxTaskStep(
        id=verifier_id, version=None, sandbox_name="box", base_dir="/app/greeting", verifier_id=verifier_id,
        criteria=[
            {"type": "probe_file_contains", "criterion": "greets", "paths": ["hello.txt"], "expected": expected},
            {"type": "bash_cmd_succeeds", "criterion": "check passes", "bash_cmd": bash_cmd},
        ],
    )


@pytest.mark.asyncio
async def test_verify_sandbox_scores_a_bare_local_sandbox(local_backends):
    tmp_path, sandboxes = local_backends
    suffix = uuid.uuid4().hex[:8]
    universe = _greeting_universe(tmp_path / "greeting", suffix)
    task = Task.put(id=f"hello-{suffix}", steps=[
        DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local"),
        LoadArtifactTaskStep(
            id="load", version=None, artifact_id=universe.id, sandbox_name="box", destination_path="/app/greeting",
        ),
        _verify("hello", "hello", "python3 check.py"),
        _verify("control", "goodbye", "python3 -c 'raise SystemExit(1)'"),
    ])

    ctx = await task.run()

    verifications = ctx.metadata["verifications"]
    assert verifications["hello"]["score"] == 1.0
    assert [r["result"] for r in verifications["hello"]["results"]] == [True, True]
    assert verifications["control"]["score"] == 0.0
    assert [r["result"] for r in verifications["control"]["results"]] == [False, False]
    (deployed,) = ctx.deployed_sandboxes
    work_dir = LocalSandbox.find_work_dir(deployed.sandbox_id)
    assert work_dir.parent == sandboxes
    assert (work_dir / "greeting" / "hello.txt").read_text() == "hello, world\n"
    assert Task.get_instance(ctx.instance_id).status == "completed"


@pytest.mark.asyncio
async def test_file_artifacts_stage_beside_each_other(local_backends):
    tmp_path, _ = local_backends
    suffix = uuid.uuid4().hex[:8]
    files = _greeting_universe(tmp_path / "greeting", suffix).get_file_artifacts()
    task = Task.put(id=f"hello-files-{suffix}", steps=[
        DeploySandboxTaskStep(id="deploy", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local"),
        *(
            LoadArtifactTaskStep(
                id=f"load-{i}", version=None, artifact_id=file.id, sandbox_name="box", destination_path="/app/greeting",
            )
            for i, file in enumerate(files.values())
        ),
        _verify("hello", "hello", "python3 check.py"),
    ])

    ctx = await task.run()

    assert ctx.metadata["verifications"]["hello"]["score"] == 1.0
    loaded = ctx.metadata["loaded_file_artifact_universes"]
    assert [(entry["artifact_type"], entry["files"]) for entry in loaded] == [("file", ["hello.txt"]), ("file", ["check.py"])]
