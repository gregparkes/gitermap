"""Test file to run pytest."""

import os
import pathlib

import pytest

os.chdir(pathlib.Path.cwd() / "tests")

pytest.main()
