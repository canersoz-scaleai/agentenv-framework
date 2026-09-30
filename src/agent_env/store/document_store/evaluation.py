"""In-Python evaluation of the Filter, Sort and UpdateSpec algebra with Mongo semantics,
for backends with no native query language to push it down to."""

from __future__ import annotations

from typing import Optional

from agent_env.store.document_store.document_store import (
    AbsentOrNull,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    Lte,
    LteOrAbsent,
    Ne,
    Predicate,
    Sort,
    UpdateSpec,
)


def _resolve(doc: dict, path: str) -> tuple[bool, list]:
    """Resolve a dotted path to (present, values): a numeric segment indexes an
    array; a non-numeric segment into an array traverses its elements."""
    cursors = [doc]
    for seg in path.split("."):
        nxt = []
        for cur in cursors:
            if isinstance(cur, dict):
                if seg in cur:
                    nxt.append(cur[seg])
            elif isinstance(cur, list):
                if seg.isdigit():
                    idx = int(seg)
                    if 0 <= idx < len(cur):
                        nxt.append(cur[idx])
                else:
                    for el in cur:
                        if isinstance(el, dict) and seg in el:
                            nxt.append(el[seg])
        cursors = nxt
        if not cursors:
            return False, []
    return True, cursors


def _eq(rv, value) -> bool:
    return rv == value or (isinstance(rv, list) and value in rv)


def _cmp(rv, value, op: str) -> bool:
    try:
        return rv >= value if op == "ge" else rv <= value
    except TypeError:
        return False


def _pred_matches(doc: dict, field: str, pred: Predicate) -> bool:
    present, values = _resolve(doc, field)
    if isinstance(pred, Eq):
        return any(_eq(rv, pred.value) for rv in values) if present else pred.value is None
    if isinstance(pred, Ne):
        return present and not any(_eq(rv, pred.value) for rv in values)
    if isinstance(pred, In):
        return present and any(
            rv in pred.values or (isinstance(rv, list) and any(e in pred.values for e in rv))
            for rv in values
        )
    if isinstance(pred, Gte):
        return any(_cmp(rv, pred.value, "ge") for rv in values)
    if isinstance(pred, Lte):
        return any(_cmp(rv, pred.value, "le") for rv in values)
    if isinstance(pred, Exists):
        return present == pred.present
    if isinstance(pred, LteOrAbsent):
        if not present:
            return True
        return any(rv is None for rv in values) or any(
            _cmp(rv, pred.value, "le") for rv in values
        )
    if isinstance(pred, AbsentOrNull):
        return (not present) or any(rv is None for rv in values)
    raise TypeError(f"Unsupported predicate: {pred!r}")


def matches(doc: dict, filter: Filter) -> bool:
    return all(
        _pred_matches(doc, field, pred)
        for field, preds in filter.conditions.items()
        for pred in preds
    )


def _sort_key(doc: dict, field: str):
    present, values = _resolve(doc, field)
    if not present or values[0] is None:
        return (0,)
    return (1, values[0])


def sort_docs(docs: list[dict], sort: Optional[Sort]) -> list[dict]:
    if sort is None or not sort.keys:
        return docs
    for key in reversed(sort.keys):
        docs = sorted(docs, key=lambda d, k=key: _sort_key(d, k.field), reverse=key.descending)
    return docs


def _get_child(container, seg: str):
    if isinstance(container, dict):
        return container.get(seg)
    if isinstance(container, list) and seg.isdigit() and 0 <= int(seg) < len(container):
        return container[int(seg)]
    return None


def _step_write(container, seg: str):
    if isinstance(container, list):
        return container[int(seg)]
    child = container.get(seg)
    if not isinstance(child, (dict, list)):
        child = {}
        container[seg] = child
    return child


def _assign(container, seg: str, value) -> None:
    if isinstance(container, list):
        container[int(seg)] = value
    else:
        container[seg] = value


def _set_path(doc: dict, path: str, value) -> None:
    segs = path.split(".")
    cur = doc
    for seg in segs[:-1]:
        cur = _step_write(cur, seg)
    _assign(cur, segs[-1], value)


def _unset_path(doc: dict, path: str) -> None:
    segs = path.split(".")
    cur = doc
    for seg in segs[:-1]:
        cur = _get_child(cur, seg)
        if cur is None:
            return
    last = segs[-1]
    if isinstance(cur, dict):
        cur.pop(last, None)
    elif isinstance(cur, list) and last.isdigit() and 0 <= int(last) < len(cur):
        cur[int(last)] = None


def _ensure_list(doc: dict, path: str) -> list:
    segs = path.split(".")
    cur = doc
    for seg in segs[:-1]:
        cur = _step_write(cur, seg)
    existing = _get_child(cur, segs[-1])
    if not isinstance(existing, list):
        existing = []
        _assign(cur, segs[-1], existing)
    return existing


def apply_update(doc: dict, spec: UpdateSpec) -> None:
    spec.validate()
    for path, value in spec.set.items():
        _set_path(doc, path, value)
    for path in spec.unset:
        _unset_path(doc, path)
    for path, delta in spec.inc.items():
        segs = path.split(".")
        cur = doc
        for seg in segs[:-1]:
            cur = _step_write(cur, seg)
        current = _get_child(cur, segs[-1])
        _assign(cur, segs[-1], (current or 0) + delta)
    for path, items in spec.add_to_set.items():
        lst = _ensure_list(doc, path)
        for it in items:
            if it not in lst:
                lst.append(it)
    for path, items in spec.push.items():
        _ensure_list(doc, path).extend(items)


def synthesize(filter: Filter, spec: UpdateSpec) -> dict:
    doc: dict = {}
    for field, preds in filter.conditions.items():
        for pred in preds:
            if isinstance(pred, Eq):
                _set_path(doc, field, pred.value)
    apply_update(doc, spec)
    return doc
