"""Write a planned bundle's entities, tasks and evals, reusing each version whose inputs haven't changed.

Materializing first refuses every write this release has no writer for, so nothing is written for a bundle
that can't be written whole. It needs the CLI's namespace routing, which sends ``@local`` writes to the
``@local`` namespace's store. Holding the bundle's lock, it writes the entities in the plan's order, then
builds and preflights every task before writing any of them, and writes the evals last, since they name the
tasks. Each write goes through the ledger, so one whose inputs haven't changed reuses the version the bundle
last wrote. A reused task keeps whatever its steps took from config when it was first written, such as a
rubrics verifier's default judge model.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.entity_refs import EntityRef, RefRole, ref_sites
from agent_env.eval import Eval, EvalTask
from agent_env.store.routing import namespace_routing_enabled
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry

from ._fs import relative, with_article
from .authoring import AuthoringContext
from .ledger import Ledger, materializing
from .parse import BundleError, BundleKind
from .plan import Plan, Write, folder_walk
from .resolve import BuiltImage, build_step

@dataclass(frozen=True)
class Materialized:
    """One write: the version it left in the store, and whether that is the one the bundle last wrote."""

    write: Write
    version: int
    reused: bool
    reasons: tuple[str, ...]  # why a new version was written; empty when reused


@dataclass(frozen=True)
class Materialization:
    """What materializing a plan left in the store, one ``Materialized`` per write."""

    plan: Plan
    writes: tuple[Materialized, ...]  # in the plan's order

    def version_of(self, store: str, id: str) -> int:
        for done in self.writes:
            if (done.write.kind.store, done.write.id) == (store, id):
                return done.version
        raise KeyError(f"{store} {id!r} isn't one of this plan's writes")


def materialize(
    plan: Plan,
    *,
    on_wait: Callable[[], None] | None = None,
    on_write: Callable[[Materialized], None] | None = None,
) -> Materialization:
    """Write ``plan``'s entities, tasks and evals. The first write that fails stops it, and every earlier
    one stays: they're in the ledger, so the next run reuses them. ``on_wait`` is called when another run
    holds a lock this one needs, and ``on_write`` after each write, reused or not."""
    _refuse_unwritable(plan)
    if not namespace_routing_enabled():
        raise RuntimeError("materializing a bundle needs namespace routing, which the agent-env CLI turns on; "
                           "call it inside agent_env.store.routing.namespace_routing()")
    ledger = Ledger.for_plan(plan)
    entities = [write for write in plan.writes if write.kind not in (BundleKind.TASK, BundleKind.EVAL)]
    tasks = [write for write in plan.writes if write.kind is BundleKind.TASK]
    evals = [write for write in plan.writes if write.kind is BundleKind.EVAL]
    done: dict[tuple[str, str], Materialized] = {}

    def through_ledger(write: Write, write_fn: Callable[[], int]) -> None:
        done[_key(write)] = _through_ledger(plan, ledger, write, write_fn)
        if on_write is not None:
            on_write(done[_key(write)])

    with materializing(plan.bundle.bundle, on_wait):
        for write in entities:
            through_ledger(write, lambda: _WRITERS[write.kind](plan, write))
        built, problems = {}, []
        for write in tasks:
            with _noted(plan, write, "preflighting"):
                built[write.id] = _build(write)
                problems.extend(f"{_path(plan, write)}: {problem}" for problem in _preflight(write, built[write.id]))
        if problems:
            raise BundleError(problems)
        for write in tasks:
            through_ledger(write, lambda: Task.put(id=write.id, steps=built[write.id].steps).version)
        for write in evals:
            through_ledger(write, lambda: _WRITERS[write.kind](plan, write))
    return Materialization(plan, tuple(done[_key(write)] for write in plan.writes))


def _write_artifact(plan: Plan, write: Write) -> int:
    entry = write.source.entry
    cls = get_artifact_registry()[canonical_type(entry.type)]
    return cls.from_toml(_pinned(plan, write, cls.toml_refs), AuthoringContext(plan.bundle.bundle, entry)).version


def _write_agent(plan: Plan, write: Write) -> int:
    entry = write.source.entry
    return A2AAgent.from_toml(_pinned(plan, write, A2AAgent.toml_refs),
                              AuthoringContext(plan.bundle.bundle, entry)).version


def _pinned(plan: Plan, write: Write, refs: tuple[EntityRef, ...]) -> Any:
    """A copy of ``write``'s resolved toml with each store ref that names no version pinned to the version
    the plan checked, which the ledger hashed, so one the store gains before the write isn't written."""
    config = copy.deepcopy(write.source.config)
    for site in ref_sites(refs, config, inline_pins=True):
        planned = plan.store_latest.get((site.ref.kind, site.value)) if site.version is None else None
        if planned is None:
            continue
        if site.version_key is None:
            site.owner[site.key] = {site.ref.kind.value: site.value, "version": planned}
        else:
            site.rewrite(site.value, planned)
    return config


def _write_eval(plan: Plan, write: Write) -> int:
    """An unpinned task ref, the bundle's own or a store's, is written without a version: the eval runs
    its latest."""
    tasks = [EvalTask(ref.id, ref.version) for ref in write.source.references]
    return Eval.put(id=write.id, tasks=tasks).version


# The writers this release has, by kind; any other kind is refused before anything is written. Tasks are
# written separately, once every one of them is preflighted, and evals after them, since they name the tasks.
_WRITERS: dict[BundleKind, Callable[[Plan, Write], int]] = {
    BundleKind.ARTIFACT: _write_artifact, BundleKind.AGENT: _write_agent, BundleKind.EVAL: _write_eval,
}


def _refuse_unwritable(plan: Plan) -> None:
    problems = [f"{_path(plan, write)}: writing {what} isn't supported yet"
                for write in plan.writes if (what := _unwritable(write))]
    if problems:
        raise BundleError(problems)


def _unwritable(write: Write) -> str | None:
    """What ``write`` would write, when this release has no writer for it."""
    if isinstance(write.source, BuiltImage):
        return f"an image built from {write.source.dockerfile}"
    if write.kind is BundleKind.TASK:
        return None
    if write.kind not in _WRITERS:
        return with_article(write.kind.value.removesuffix("s"))
    if write.kind is BundleKind.ARTIFACT:
        type_ = write.source.entry.type
        if folder_walk(get_artifact_registry().get(canonical_type(type_))) is None:
            return with_article(f"{type_} artifact")
    return None


def _through_ledger(plan: Plan, ledger: Ledger, write: Write, write_fn: Callable[[], int]) -> Materialized:
    with _noted(plan, write, "writing"):
        check = ledger.check(write)
        version = check.version if check.unchanged else ledger.record(check, write_fn)
    return Materialized(write, version, check.unchanged, check.reasons)


@contextmanager
def _noted(plan: Plan, write: Write, doing: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        e.add_note(f"while {doing} {_path(plan, write)} ({write.id})")
        raise


def _build(write: Write) -> Task:
    return Task(id=write.id, version=None, steps=[build_step(step) for step in write.source.config])


def _preflight(write: Write, task: Task) -> list[str]:
    """``task``'s preflight problems. A step reading one of the task's own outputs is skipped: the output
    only exists once the task runs."""
    outputs = {output.id for output in write.source.outputs}
    problems = []
    for config, step in zip(write.source.config, task.steps):
        if not _reads_any(config, outputs):
            problems.extend(step.preflight())
    return problems


def _reads_any(step: dict, ids: set[str]) -> bool:
    refs = get_task_step_registry()[step["type"]].entity_refs or ()
    return any(site.value in ids for site in ref_sites(refs, step) if site.ref.role is RefRole.INPUT)


def _key(write: Write) -> tuple[str, str]:
    return (write.kind.store, write.id)


def _path(plan: Plan, write: Write) -> str:
    return relative(plan.bundle.bundle.root, write.source.entry.path)
