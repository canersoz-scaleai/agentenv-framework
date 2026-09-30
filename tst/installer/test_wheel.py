"""What the agentenv-framework wheel built from the checkout ships."""

from __future__ import annotations

import zipfile

import pytest

from tst.installer.conftest import requires_tool

pytestmark = [pytest.mark.installer, requires_tool("uv")]


def test_the_wheel_marks_agent_env_as_typed(checkout_wheels):
    framework = next(checkout_wheels.glob("agentenv_framework-*.whl"))

    assert "agent_env/py.typed" in zipfile.ZipFile(framework).namelist()
