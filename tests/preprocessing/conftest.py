from dataclasses import replace
from pathlib import Path

import pytest

from src.preprocessing import PreprocessConfig
from src.preprocessing.decode import decode_frames
from src.preprocessing.probe import probe_clip


@pytest.fixture(scope="session")
def fixture_path():
    return Path(__file__).parents[1] / "fixtures" / "room_short.mp4"


@pytest.fixture(scope="session")
def clip(fixture_path):
    return probe_clip(fixture_path)


@pytest.fixture(scope="session")
def config():
    default = PreprocessConfig()
    return replace(default, revisits=replace(default.revisits, enabled=False))


@pytest.fixture(scope="session")
def frames(clip, config):
    return list(decode_frames(clip, config.decode))
