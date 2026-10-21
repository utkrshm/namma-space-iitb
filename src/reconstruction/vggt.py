"""Lazy, reusable CUDA-only VGGT inference for one scene per call."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np

from .contracts import ImageBatch, ReconstructionResult
from .geometry import decode_cameras, unproject_depth
from .images import ImageInput, load_images


def _upstream_modules():
    """Use an installed VGGT or the unmodified repository submodule."""
    if importlib.util.find_spec("vggt") is None:
        root = Path(__file__).parent / "deps" / "vggt"
        if not (root / "vggt" / "models" / "vggt.py").is_file():
            raise ImportError(
                "VGGT source is missing. Run: git submodule update --init src/reconstruction/deps/vggt"
            )
        sys.path.insert(0, str(root))

    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    return VGGT, pose_encoding_to_extri_intri


class VGGTInference:
    """Load once, infer ordered views jointly, and return CPU NumPy reconstruction data.

    A call represents a single scene, not a batch of unrelated scenes. All heads
    are enabled; tracking outputs require query points in the first model image.
    Inference requires a sufficiently large CUDA GPU; there is no CPU fallback.
    """

    def __init__(
        self,
        model_id: str | Path = "facebook/VGGT-1B",
        *,
        device: str = "cuda",
        precision: Literal["auto", "float32", "float16", "bfloat16"] = "auto",
        image_size: int = 518,
        local_files_only: bool = False,
    ):
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("VGGT inference targets CUDA GPUs; use device='cuda' or 'cuda:N'")
        if precision not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("Unsupported precision")
        if not isinstance(image_size, int) or image_size < 14 or image_size % 14:
            raise ValueError("image_size must be a positive multiple of 14")

        self.model_id = str(model_id)
        self.device = device
        self.precision = precision
        self.image_size = image_size
        self.local_files_only = local_files_only
        self._model = None
        self._decode_pose = None
        self._resolved_precision = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> VGGTInference:
        """Download/read pretrained weights and put the evaluation model on CUDA."""
        if self.is_loaded:
            return self
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "Install inference dependencies: uv sync --extra reconstruction"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run inference on the larger GPU machine.")
        with torch.cuda.device(self.device):
            supports_bf16 = torch.cuda.is_bf16_supported()
        precision = self.precision
        if precision == "auto":
            precision = "bfloat16" if supports_bf16 else "float16"
        if precision == "bfloat16" and not supports_bf16:
            raise ValueError("This CUDA device does not support bfloat16")

        model_type, decode_pose = _upstream_modules()
        model = model_type.from_pretrained(self.model_id, local_files_only=self.local_files_only)
        self._model = model.eval().to(self.device)
        self._decode_pose = decode_pose
        self._resolved_precision = precision
        return self

    def unload(self) -> None:
        """Release this wrapper's model reference; CPU results remain usable."""
        self._model = None
        self._decode_pose = None
        self._resolved_precision = None

    def predict(
        self,
        images: ImageInput | Sequence[ImageInput] | ImageBatch,
        *,
        query_points: np.ndarray | None = None,
    ) -> ReconstructionResult:
        """Infer one image or multiple views; query_points is finite N,2 model-pixel XY.

        Use batch.transforms[0].transform_points(source_xy) for upright source
        queries. ImageBatch inputs can carry preprocessing frame provenance.
        CUDA OOM propagates: views are never silently dropped or independently chunked.
        """
        batch = (
            images
            if isinstance(images, ImageBatch)
            else load_images(images, image_size=self.image_size)
        )
        self._validate_batch(batch)
        queries = self._validate_queries(query_points, batch)
        self.load()

        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(batch.images)).to(self.device)
        query_tensor = None if queries is None else torch.from_numpy(queries).to(self.device)
        autocast = (
            nullcontext()
            if self._resolved_precision == "float32"
            else torch.autocast("cuda", dtype=getattr(torch, self._resolved_precision))
        )
        with torch.inference_mode():
            with autocast:
                predictions = self._model(tensor, query_points=query_tensor)
            extrinsics, intrinsics = self._decode_pose(
                predictions["pose_enc"].float(), tensor.shape[-2:]
            )

        def array(value):
            return value[0].detach().float().cpu().numpy()

        cameras = decode_cameras(array(extrinsics), array(intrinsics), batch.transforms)
        depth = array(predictions["depth"])[..., 0]
        return ReconstructionResult(
            images=np.rint(batch.images.transpose(0, 2, 3, 1) * 255).astype(np.uint8),
            valid_mask=batch.valid_mask,
            transforms=batch.transforms,
            frames=batch.frames,
            cameras=cameras,
            pose_encoding=array(predictions["pose_enc"]),
            pose_encoding_iterations=np.stack([array(p) for p in predictions["pose_enc_list"]]),
            depth=depth,
            depth_confidence=array(predictions["depth_conf"]),
            world_points=array(predictions["world_points"]),
            world_points_confidence=array(predictions["world_points_conf"]),
            depth_world_points=unproject_depth(depth, cameras),
            model_id=self.model_id,
            precision=self._resolved_precision,
            tracks=array(predictions["track"]) if queries is not None else None,
            track_visibility=array(predictions["vis"]) if queries is not None else None,
            track_confidence=array(predictions["conf"]) if queries is not None else None,
        )

    @staticmethod
    def _validate_batch(batch: ImageBatch) -> None:
        images = batch.images
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
            raise ValueError("ImageBatch images must have shape S,3,H,W")
        size, _, height, width = images.shape
        if height < 14 or width < 14 or height % 14 or width % 14:
            raise ValueError("Model image dimensions must be positive multiples of 14")
        if (
            images.dtype != np.float32
            or not np.isfinite(images).all()
            or images.min() < 0
            or images.max() > 1
        ):
            raise ValueError("ImageBatch images must be finite float32 RGB in [0,1]")
        if batch.valid_mask.shape != (size, height, width) or batch.valid_mask.dtype != bool:
            raise ValueError("ImageBatch valid_mask must be S,H,W bool")
        if len(batch.frames) != size or len(batch.transforms) != size:
            raise ValueError("ImageBatch provenance must match the image count")
        if any(t.model_size != (width, height) for t in batch.transforms):
            raise ValueError("ImageBatch transforms must match the model image dimensions")

    @staticmethod
    def _validate_queries(query_points, batch: ImageBatch) -> np.ndarray | None:
        if query_points is None:
            return None
        queries = np.asarray(query_points, dtype=np.float32)
        if (
            queries.ndim != 2
            or queries.shape[1] != 2
            or not len(queries)
            or not np.isfinite(queries).all()
        ):
            raise ValueError("query_points must be a nonempty finite N,2 array")
        _, _, height, width = batch.images.shape
        if (
            (queries < 0).any()
            or (queries[:, 0] > width - 1).any()
            or (queries[:, 1] > height - 1).any()
        ):
            raise ValueError("Query points must be inside the first model image")
        xy = np.rint(queries).astype(int)
        if not batch.valid_mask[0, xy[:, 1], xy[:, 0]].all():
            raise ValueError("Query points must not lie in padding")
        return np.ascontiguousarray(queries)
