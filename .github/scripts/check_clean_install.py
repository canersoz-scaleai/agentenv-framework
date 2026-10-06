"""Check, with the installed wheel's own code, the install and what ``agent-env run BUNDLE`` left in the local store.

``imports`` fails unless boto3 is absent, as without the ``aws`` extra, and every agent_env module imports or fails
only for want of a module an extra of agent-env installs. ``snapshot PATH`` records the ``@local`` writes after the
first run. ``verify`` fails unless agent-env was imported from this interpreter's environment rather than a checkout, both packages byte-compile on this interpreter's Python
(an install skips a file that doesn't), no config file was found, no other distribution registers an agent-env
plugin, each run of each of the bundle's tasks completed every step and scored 1 on every
verifier, every instance sits under the bundle's id root, the runs left no sandbox work folder, and the later runs
changed no artifact, task, ledger row or stored object.

Run by the clean venv's interpreter; see ``clean_install.py``.
"""

from __future__ import annotations

import argparse
import compileall
import importlib.metadata
import importlib.util
import json
import os
import pkgutil
import sys
from pathlib import Path

import agent_env
import agentenv_protocol
from agent_env.artifact.store import ARTIFACTS_COLLECTION
from agent_env.bundle import BundleKind
from agent_env.bundle.installed import checked, find_bundle
from agent_env.bundle.ledger import LEDGER_COLLECTION
from agent_env.config import get_config
from agent_env.config.loader import discover_config_path, missing_extra
from agent_env.config.paths import state_root
from agent_env.store import Filter
from agent_env.task.store import TASK_INSTANCES_COLLECTION, TASKS_COLLECTION, TaskInstance, TaskStepStatus

REUSED = (ARTIFACTS_COLLECTION, TASKS_COLLECTION, LEDGER_COLLECTION)
OWN_DISTRIBUTIONS = {"agentenv-framework", "agentenv-framework-protocol"}
AWS_SDK = ("boto3", "botocore")
# Not the library's: the gateway runs only in its container image, and the code runner is a script that reads its
# arguments on import.
NOT_LIBRARY = ("agent_env.env.gateway", "agent_env.task_step.task_steps.run_code_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("imports")
    commands.add_parser("snapshot").add_argument("path", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--runs", type=int, required=True)
    verify.add_argument("--since", type=Path, required=True, help="the snapshot taken after the first run")
    args = parser.parse_args()
    if args.command == "imports":
        return imports()
    if args.command == "snapshot":
        args.path.write_text(json.dumps(writes()))
        return 0
    found = problems(args.bundle, args.runs, json.loads(args.since.read_text()))
    for problem in found:
        print(f"::error::{problem}")
    if not found:
        print(f"clean install: {args.runs} runs of {args.bundle} completed with every score 1 and left no sandbox "
              "work folder, and the later runs changed no artifact, task, ledger row or stored object")
    return 1 if found else 0


def imports() -> int:
    found, needs = import_problems()
    for problem in found:
        print(f"::error::{problem}")
    if not found:
        print("clean install: no AWS SDK, and every agent_env module imports except those that need an extra: "
              + "; ".join(f"{extra}: {', '.join(modules)}" for extra, modules in sorted(needs.items())))
    return 1 if found else 0


def import_problems() -> tuple[list[str], dict[str, list[str]]]:
    """Why this isn't an install without the aws extra, or a module doesn't import; and the modules that need an
    extra, by extra."""
    found = [f"{name} is installed; the gate tests agent-env without the aws extra"
             for name in AWS_SDK if importlib.util.find_spec(name) is not None]
    needs: dict[str, list[str]] = {}
    # A package that fails to import is recorded by the loop, before the walk tries to import it again.
    for module in pkgutil.walk_packages(agent_env.__path__, "agent_env.", onerror=lambda name: None):
        if module.name.startswith(NOT_LIBRARY):
            continue
        try:
            importlib.import_module(module.name)
        except ImportError as e:
            if (extra := missing_extra(e)) is None:
                found.append(f"{module.name}: {e}")
            else:
                needs.setdefault(extra, []).append(module.name)
        except Exception as e:
            found.append(f"{module.name}: {type(e).__name__}: {e}")
    return found, needs


def writes() -> dict[str, list[str]]:
    """The ``@local`` documents and stored objects a rerun must leave unchanged, each object with its mtime."""
    store = get_config().local_namespace_document_store()
    found = {collection: sorted(json.dumps(doc, sort_keys=True, default=str)
                                for doc in store.query(collection, Filter.of()))
             for collection in REUSED}
    objects = state_root() / "object_store"
    found["objects"] = sorted(f"{path.relative_to(objects)} {path.stat().st_mtime_ns}"
                              for path in objects.rglob("*") if path.is_file())
    return found


def problems(name: str, runs: int, before: dict[str, list[str]]) -> list[str]:
    found = []
    if not Path(agent_env.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
        found.append(f"agent_env was imported from {agent_env.__file__}, not from the environment at {sys.prefix}")
    for package in (agent_env, agentenv_protocol):
        if not compileall.compile_dir(Path(package.__file__).parent, quiet=1, force=True):
            found.append(f"{package.__name__} doesn't byte-compile on Python {sys.version.split()[0]}; "
                         "see the errors above")
    if (config := discover_config_path()) is not None:
        found.append(f"agent-env found the config file {config}; the gate runs with none")
    for plugin in plugins():
        found.append(f"{plugin} is installed; the gate tests agent-env without plugins")
    bundle = find_bundle(name)
    store = get_config().local_namespace_document_store()
    for task in [entry for entry in checked(bundle).entries if entry.kind is BundleKind.TASK]:
        steps = [step["id"] for step in json.loads(task.path.read_text())]
        instances = [TaskInstance.from_dict(doc)
                     for doc in store.query(TASK_INSTANCES_COLLECTION, Filter.of(task_id=task.id))]
        if len(instances) != runs:
            found.append(f"{task.id}: {len(instances)} instances, not {runs}")
        for instance in instances:
            where = instance.instance_id
            if not where.startswith(f"{bundle.id_root}/"):
                found.append(f"{where}: not under {bundle.id_root}/")
            if instance.status != "completed":
                found.append(f"{where}: {instance.status} ({instance.error})")
            done = [result.step_id for result in instance.completed_steps if result.status == TaskStepStatus.SUCCESS]
            if done != steps:
                found.append(f"{where}: completed {done}, not {steps}")
            verifications = (instance.context or {}).get("metadata", {}).get("verifications", {})
            scores = {verifier: result.get("score") for verifier, result in verifications.items()}
            if not scores or any(score != 1.0 for score in scores.values()):
                found.append(f"{where}: scores {scores or 'none'}, not all 1")
    if leftovers := sorted(path.name for path in Path(os.environ["AGENT_ENV_LOCAL_SANDBOX_DIR"]).glob("*")):
        found.append(f"the runs left {len(leftovers)} sandbox work folder(s): {', '.join(leftovers[:3])}")
    after = writes()
    for kind in before:
        if changed := sorted(set(after[kind]) ^ set(before[kind])):
            found.append(f"a later run changed {kind}: {', '.join(_described(entry) for entry in changed[:3])}")
    return found


def plugins() -> list[str]:
    """Entry points in an ``agent_env.*`` group from any distribution other than agent-env's own."""
    return sorted(f"{dist.name}'s {ep.group} entry point {ep.name}"
                  for dist in importlib.metadata.distributions()
                  if dist.name not in OWN_DISTRIBUTIONS
                  for ep in dist.entry_points if ep.group.startswith("agent_env."))


def _described(entry: str) -> str:
    """A stored document by its id, version and status; an object by its path and mtime."""
    if not entry.startswith("{"):
        return entry
    doc = json.loads(entry)
    return " ".join(str(doc[key]) for key in ("id", "version", "status") if key in doc) or entry[:80]


if __name__ == "__main__":
    sys.exit(main())
