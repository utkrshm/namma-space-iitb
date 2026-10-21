"""VGGT-specific preparation; never alter the frozen preprocessing exports."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .contracts import FrameProvenance, ImageBatch, ImageTransform

ImageInput = str | Path | np.ndarray


def load_images(images: ImageInput | Sequence[ImageInput], *, image_size: int = 518) -> ImageBatch:
    """Load paths or uint8 HWC RGB arrays into one ordered scene, padded to squares.

    Match VGGT's pad-mode resize: longest side image_size, other side rounded
    to a multiple of 14, bicubic PIL resize, centred white padding. Inputs are
    already upright; EXIF/display rotation is not applied a second time.
    """
    from PIL import Image

    if not isinstance(image_size, int) or image_size < 14 or image_size % 14:
        raise ValueError("image_size must be a positive multiple of VGGT's patch size 14")
    if isinstance(images, (str, Path, np.ndarray)):
        images = [images]
    if not images:
        raise ValueError("At least one image is required")

    tensors, masks, transforms, frames = [], [], [], []
    for index, source in enumerate(images):
        path = None
        if isinstance(source, (str, Path)):
            path = str(Path(source).resolve())
            with Image.open(path) as image:
                if image.mode == "RGBA" or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba)
                rgb = image.convert("RGB")
        elif isinstance(source, np.ndarray):
            if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
                raise ValueError("Array inputs must be uint8 H,W,3 RGB (not BGR)")
            if not source.shape[0] or not source.shape[1]:
                raise ValueError("Images must have nonzero dimensions")
            rgb = Image.fromarray(source)
        else:
            raise TypeError("Images must be paths or uint8 RGB arrays")

        width, height = rgb.size
        scale = image_size / max(width, height)
        resized = tuple(max(14, round(value * scale / 14) * 14) for value in (width, height))
        left, top = ((image_size - value) // 2 for value in resized)
        transform = ImageTransform((width, height), resized, (image_size, image_size), (left, top))

        canvas = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        canvas[top : top + resized[1], left : left + resized[0]] = np.asarray(
            rgb.resize(resized, Image.Resampling.BICUBIC)
        )
        mask = np.zeros((image_size, image_size), dtype=bool)
        mask[top : top + resized[1], left : left + resized[0]] = True

        tensors.append(canvas.transpose(2, 0, 1).astype(np.float32) / 255)
        masks.append(mask)
        transforms.append(transform)
        frames.append(FrameProvenance(str(index), path))

    return ImageBatch(np.stack(tensors), np.stack(masks), tuple(transforms), tuple(frames))
