"""LoadArtifactTaskStep `urls` entries: an object entry ``{"url", "filename"}`` is saved under its
filename verbatim, survives a round-trip, and a bad or colliding entry fails at construction."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

OPAQUE_URL = "https://files.example/objects/obj-4f9c2a"


def _step(urls) -> LoadArtifactTaskStep:
    return LoadArtifactTaskStep(id="load", version=None, sandbox_name="box", urls=urls)


def test_object_entries_survive_a_round_trip():
    urls = ["https://a.example/data.csv", {"url": OPAQUE_URL, "filename": "report.txt"}]
    data = _step(urls).to_dict()
    assert data["urls"] == urls
    assert LoadArtifactTaskStep.from_dict(data).urls == urls


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"url": OPAQUE_URL}, "non-empty string 'filename'"),
        ({"url": OPAQUE_URL, "name": "a.txt"}, r"unknown key\(s\) \['name'\]"),
        ({"url": OPAQUE_URL, "filename": "../a.txt"}, r"must not contain '\.\.' segments"),
        ({"url": OPAQUE_URL, "filename": "/etc/passwd"}, "must be relative"),
        ({"url": OPAQUE_URL, "filename": "sub/a.txt"}, "plain file name, not a path"),
    ],
)
def test_a_bad_object_entry_fails_at_construction(entry, match):
    with pytest.raises(ValueError, match=match):
        _step([entry])


def test_a_filename_given_twice_fails():
    with pytest.raises(ValueError, match="'a.txt' is given to more than one urls entry"):
        _step([{"url": OPAQUE_URL, "filename": "a.txt"}, {"url": OPAQUE_URL + "2", "filename": "a.txt"}])


def test_a_bare_url_named_like_an_explicit_filename_fails():
    with pytest.raises(ValueError, match=r"'https://a.example/data.csv' would be saved as 'data.csv'"):
        _step(["https://a.example/data.csv", {"url": OPAQUE_URL, "filename": "data.csv"}])
