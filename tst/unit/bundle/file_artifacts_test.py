"""Authoring file and file_artifact_universe artifacts from a bundle folder: what the folder holds,
which links count, and what gets written."""

import os
import re
import unicodedata

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.authoring import AuthoringContext
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.ids import key_segment

ROOT = "@local/~/triage"
BRING_IT_IN = "copy what it points to into the bundle, or put it in a store and refer to it by id"


class Link:
    """A symlink to ``target``, relative to the link's folder unless absolute."""

    def __init__(self, target):
        self.target = target


@pytest.fixture
def make(tmp_path, monkeypatch):
    """A bundle at ``~/triage`` holding ``files``: text, or a ``Link``."""
    monkeypatch.setenv("HOME", str(tmp_path))

    def make(files):
        for rel, content in files.items():
            path = tmp_path / "triage" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, Link):
                path.symlink_to(content.target)
            else:
                path.write_text(content)
        return parse_bundle(tmp_path / "triage")

    return make


def context(bundle, name):
    return AuthoringContext(bundle, next(entry for entry in bundle.entries if entry.name == name))


def contents(bundle, name):
    return {key: path.read_text() for key, path in context(bundle, name).files().items()}


def problems(call):
    with pytest.raises(BundleError) as caught:
        call()
    return caught.value.problems


# The files a folder holds


def test_files_are_keyed_by_path_in_order_dot_files_included_less_leavings_and_the_folders_toml(make):
    bundle = make({
        "artifacts/data/b.txt": "b",
        "artifacts/data/a/z.txt": "z",
        "artifacts/data/artifact.toml": "",
        "artifacts/data/a/artifact.toml": "nested",
        "artifacts/data/.env.example": "e",
        "artifacts/data/a/.git/config": "g",
        "artifacts/data/.DS_Store": "",
        "artifacts/data/a/Thumbs.db": "",
        "artifacts/data/a/desktop.ini": "",
        "artifacts/data/__pycache__/m.pyc": "",
        "artifacts/data/a/__pycache__/n.pyc": "",
    })
    found = contents(bundle, "data")
    assert found == {".env.example": "e", "a/.git/config": "g", "a/artifact.toml": "nested", "a/z.txt": "z",
                     "b.txt": "b"}
    assert list(found) == sorted(found)


def test_keys_are_nfc_however_the_disk_spells_them(make):
    decomposed = unicodedata.normalize("NFD", "café.txt")
    bundle = make({f"artifacts/data/{decomposed}": "x", "artifacts/data/b.txt": "b"})
    assert set(contents(bundle, "data")) == {"café.txt", "b.txt"}


def test_names_equal_once_normalized_are_refused(make, tmp_path):
    bundle = make({"artifacts/data/café.txt": "composed", "artifacts/data/b.txt": "b"})
    folder = tmp_path / "triage" / "artifacts" / "data"
    (folder / unicodedata.normalize("NFD", "café.txt")).write_text("decomposed")
    if len(os.listdir(folder)) < 3:
        pytest.skip("this filesystem stores both spellings as one name")
    (problem,) = problems(context(bundle, "data").files)
    assert problem.endswith("once normalized to NFC; rename one")


def test_links_inside_the_bundle_are_followed(make):
    bundle = make({
        "artifacts/shared/x.txt": "shared",
        "artifacts/data/y.txt": "y",
        "artifacts/data/x.txt": Link("../shared/x.txt"),
        "artifacts/data/sub": Link("../shared"),
        "vendor/pack/p.txt": "packed",
        "artifacts/pack": Link("../vendor/pack"),
    })
    assert contents(bundle, "data") == {"sub/x.txt": "shared", "x.txt": "shared", "y.txt": "y"}
    assert contents(bundle, "pack") == {"p.txt": "packed"}


@pytest.mark.parametrize(("rel", "target"), [
    ("artifacts/data/secret.txt", "secret.txt"),
    ("artifacts/data/stash", "stash"),
])
def test_a_link_leaving_the_bundle_is_refused(make, tmp_path, rel, target):
    outside = tmp_path / "outside"
    (outside / "stash").mkdir(parents=True)
    (outside / "secret.txt").write_text("s")
    bundle = make({"artifacts/data/kept.txt": "k", rel: Link(outside / target)})
    assert problems(context(bundle, "data").files) == (
        f"{rel}: links to {os.path.realpath(outside / target)}, outside the bundle; {BRING_IT_IN}",
    )


def test_an_artifact_folder_linked_from_outside_the_bundle_is_refused(make, tmp_path):
    outside = tmp_path / "datasets" / "big"
    outside.mkdir(parents=True)
    (outside / "rows.csv").write_text("r")
    bundle = make({"artifacts/big": Link(outside)})
    assert problems(context(bundle, "big").files) == (
        f"artifacts/big: links to {os.path.realpath(outside)}, outside the bundle; {BRING_IT_IN}",
    )


def test_a_folder_link_back_to_a_folder_on_the_way_down_is_refused(make):
    bundle = make({"artifacts/data/a/x.txt": "x", "artifacts/data/b/y.txt": "y",
                   "artifacts/data/a/l": Link("../b"), "artifacts/data/b/m": Link("../a")})
    assert problems(context(bundle, "data").files) == (
        "artifacts/data/a/l/m: links back to a folder that holds it",
        "artifacts/data/b/m/l: links back to a folder that holds it",
    )


def test_a_link_spelling_the_bundle_root_in_another_case_is_inside(make, tmp_path):
    bundle = make({"artifacts/shared/x.txt": "shared", "artifacts/data/y.txt": "y"})
    other_case = tmp_path / "TRIAGE" / "artifacts" / "shared" / "x.txt"
    if not other_case.exists():
        pytest.skip("this filesystem is case-sensitive")
    (tmp_path / "triage" / "artifacts" / "data" / "x.txt").symlink_to(other_case)
    assert contents(bundle, "data") == {"x.txt": "shared", "y.txt": "y"}


def test_an_artifact_folder_under_an_artifacts_link_leaving_the_bundle_is_refused(make, tmp_path):
    outside = tmp_path / "elsewhere"
    (outside / "data").mkdir(parents=True)
    (outside / "data" / "a.txt").write_text("a")
    bundle = make({"artifacts": Link(outside)})
    assert problems(context(bundle, "data").files) == (
        f"artifacts/data: resolves to {os.path.realpath(outside / 'data')}, outside the bundle; {BRING_IT_IN}",
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_a_file_that_cant_be_read_is_refused(make, tmp_path):
    bundle = make({"artifacts/data/a.txt": "a", "artifacts/data/b.txt": "b"})
    (tmp_path / "triage" / "artifacts" / "data" / "b.txt").chmod(0)
    assert problems(context(bundle, "data").files) == ("artifacts/data/b.txt: permission denied",)


@pytest.mark.parametrize("target", [".", "..", "../.."])
def test_a_folder_link_back_to_a_holder_is_refused(make, target):
    bundle = make({"artifacts/data/kept.txt": "k", "artifacts/data/loop": Link(target)})
    assert problems(context(bundle, "data").files) == ("artifacts/data/loop: links back to a folder that holds it",)


def test_a_broken_link_is_refused(make):
    bundle = make({"artifacts/data/kept.txt": "k", "artifacts/data/sub/gone.txt": Link("missing.txt")})
    assert problems(context(bundle, "data").files) == ("artifacts/data/sub/gone.txt: a broken symlink",)


def test_something_neither_a_file_nor_a_folder_is_refused(make, tmp_path):
    bundle = make({"artifacts/data/kept.txt": "k", "artifacts/data/sub/kept.txt": "k"})
    os.mkfifo(tmp_path / "triage" / "artifacts" / "data" / "sub" / "pipe")
    assert problems(context(bundle, "data").files) == ("artifacts/data/sub/pipe: neither a regular file nor a folder",)


def test_a_folder_with_nothing_to_write_is_refused(make):
    bundle = make({"artifacts/data/sub/.DS_Store": "", "artifacts/data/sub/__pycache__/m.pyc": ""})
    assert problems(context(bundle, "data").files) == (
        "artifacts/data: has no files to write (.DS_Store, Thumbs.db, __pycache__, desktop.ini don't count)",
    )


def test_a_declared_file_holds_one_file(make):
    bundle = make({"artifacts/pair/artifact.toml": 'type = "file"\n', "artifacts/pair/a.txt": "a",
                   "artifacts/pair/b.txt": "b"})
    assert problems(context(bundle, "pair").file) == (
        "artifacts/pair: a file artifact holds one file, and this folder has 2 ('a.txt', 'b.txt'); leave out its "
        "declared type to write the folder as a file_artifact_universe",
    )


@pytest.mark.parametrize(("cls", "data", "problem"), [
    (FileArtifact, {"descripton": "x"}, "unknown key 'descripton'; a file artifact takes description, type and id"),
    (FileArtifact, {"description": 3}, "description must be a string, not 3"),
    (FileArtifactUniverse, {"b": 1, "a": 2}, "unknown keys 'a', 'b'; a file_artifact_universe artifact takes type and id"),
])
def test_a_toml_key_the_type_doesnt_take_is_refused(make, cls, data, problem):
    bundle = make({"artifacts/one/a.txt": "a", "artifacts/many/a.txt": "a", "artifacts/many/b.txt": "b"})
    name = "one" if cls is FileArtifact else "many"
    assert problems(lambda: cls.from_toml(data, context(bundle, name))) == (
        f"artifacts/{name}: artifact.toml: {problem}",
    )


def test_a_toml_key_named_like_accepts_own_parameter_is_accepted(make):
    bundle = make({"artifacts/one/a.txt": "a"})
    assert context(bundle, "one").accept({"data": "x", "type": "file"}, data=str) == {"data": "x"}


# What gets written


@pytest.fixture
def stores(local_stores, cli_routing):
    return local_stores


def test_a_file_artifact_is_written_from_its_one_file(make, stores):
    bundle = make({"artifacts/greeting/hello.txt": "hello", "artifacts/note/artifact.toml": 'description = "a note"\n',
                   "artifacts/note/note.md": "# note"})
    greeting = FileArtifact.from_toml({}, context(bundle, "greeting"))
    note_entry = context(bundle, "note")
    note = FileArtifact.from_toml(note_entry.entry.config, note_entry)
    assert (greeting.id, greeting.version, greeting.filename, greeting.description, greeting.load()) == (
        f"{ROOT}/greeting", 1, "hello.txt", "hello.txt", b"hello")
    assert (note.filename, note.description, FileArtifact.get(f"{ROOT}/note").load()) == ("note.md", "a note", b"# note")


def test_a_file_write_that_fails_partway_doesnt_block_the_next(make, stores, monkeypatch, tmp_path):
    bundle = make({"artifacts/big/data.bin": "payload"})
    put_file, keys = LocalFilesystemObjectStore.put_file, []

    def interrupted(self, key, file_path, content_type="application/octet-stream"):
        keys.append(key)
        if len(keys) == 1:
            partial = tmp_path / "partial"
            partial.write_text("pay")
            put_file(self, key, str(partial), content_type)
            raise ConnectionError("interrupted")
        return put_file(self, key, file_path, content_type)

    monkeypatch.setattr(LocalFilesystemObjectStore, "put_file", interrupted)
    with pytest.raises(ConnectionError):
        FileArtifact.from_toml({}, context(bundle, "big"))
    written = FileArtifact.from_toml({}, context(bundle, "big"))
    prefix = stores.get_object_store().object_url(f"artifacts/file/{key_segment(f'{ROOT}/big')}")
    assert (written.version, written.load()) == (1, b"payload")
    assert re.fullmatch(rf"{re.escape(prefix)}/1-[0-9a-f]{{8}}/data\.bin", written.object_url)
    assert keys[0] != keys[1]


def test_written_file_names_are_nfc(make, stores):
    decomposed = unicodedata.normalize("NFD", "café.txt")
    bundle = make({f"artifacts/menu/{decomposed}": "x", f"artifacts/menus/{decomposed}": "x", "artifacts/menus/b": "b"})
    universe = FileArtifactUniverse.from_toml({}, context(bundle, "menus"))
    assert FileArtifact.from_toml({}, context(bundle, "menu")).filename == "café.txt"
    assert universe.get_file_artifacts()["café.txt"].filename == "café.txt"


def test_a_universe_is_written_from_its_files_and_rewritten_under_a_new_prefix(make, stores, tmp_path):
    bundle = make({"artifacts/docs/a.txt": "A", "artifacts/docs/sub/b.txt": "B"})
    first = FileArtifactUniverse.from_toml({}, context(bundle, "docs"))
    (tmp_path / "triage" / "artifacts" / "docs" / "a.txt").write_text("A2")
    (tmp_path / "triage" / "artifacts" / "docs" / "c.txt").write_text("C")
    second = FileArtifactUniverse.from_toml({}, context(bundle, "docs"))

    def loaded(version):
        universe = FileArtifactUniverse.get(f"{ROOT}/docs", version)
        return {key: artifact.load() for key, artifact in universe.get_file_artifacts().items()}

    prefix = stores.get_object_store().object_url(f"artifacts/file_artifact_universe/{key_segment(f'{ROOT}/docs')}")
    assert [first.version, second.version] == [1, 2]
    assert re.fullmatch(rf"{re.escape(prefix)}/1-[0-9a-f]{{8}}/", first.bundle_object_url)
    assert re.fullmatch(rf"{re.escape(prefix)}/2-[0-9a-f]{{8}}/", second.bundle_object_url)
    assert loaded(1) == {"a.txt": b"A", "sub/b.txt": b"B"}
    assert loaded(2) == {"a.txt": b"A2", "c.txt": b"C", "sub/b.txt": b"B"}
