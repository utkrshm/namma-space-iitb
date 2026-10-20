"""Model-independent smartphone-video preprocessing."""

from .config import PreprocessConfig
from .contracts import (
    ClipInfo,
    Edge,
    FrameId,
    FrameScores,
    GapItem,
    PreprocessResult,
    RotationTransform,
    SelectedFrame,
)
from .pipeline import load_result, preprocess_capture

__all__ = [
    "ClipInfo",
    "Edge",
    "FrameId",
    "FrameScores",
    "GapItem",
    "PreprocessConfig",
    "PreprocessResult",
    "RotationTransform",
    "SelectedFrame",
    "load_result",
    "preprocess_capture",
]
