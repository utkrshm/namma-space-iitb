"""Atomic checkpoints and standard binary 3DGS PLY with all SH coefficients."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from .model import GaussianModel

CHECKPOINT_VERSION = 1


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def write_json(path: str | Path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path: str | Path, model: GaussianModel, **state):
    model.validate()
    payload = {
        **state,
        "schema_version": CHECKPOINT_VERSION,
        "sh_degree": model.sh_degree,
        "params": {key: value.detach().cpu() for key, value in model.params.items()},
    }
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(cpu_tree(payload), temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GaussianModel, dict]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported splatting checkpoint schema")
    model = GaussianModel(state["params"], state["sh_degree"]).to(device)
    return model, state


def export_ply(path: str | Path, model: GaussianModel, metadata: dict | None = None):
    """Graphdeco layout: log scales, opacity logits, wxyz and channel-major SH."""
    from plyfile import PlyData, PlyElement

    model.validate()
    params = {key: value.detach().cpu().numpy() for key, value in model.params.items()}
    sh0 = params["sh0"][:, 0, :]
    shn = params["shn"].transpose(0, 2, 1).reshape(model.count, -1)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(shn.shape[1])]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = np.concatenate(
        (
            params["means"],
            np.zeros((model.count, 3), dtype=np.float32),
            sh0,
            shn,
            params["opacities"][:, None],
            params["scales"],
            params["quats"],
        ),
        axis=1,
    )
    vertices = np.empty(model.count, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(names):
        vertices[name] = values[:, index]

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(temporary)
    temporary.replace(path)
    if metadata is not None:
        write_json(path.with_suffix(path.suffix + ".json"), metadata)


def load_ply(path: str | Path, device: str = "cpu") -> tuple[GaussianModel, dict]:
    from plyfile import PlyData

    path = Path(path)
    vertices = PlyData.read(path)["vertex"]
    names = vertices.data.dtype.names
    rest = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.removeprefix("f_rest_")),
    )
    if rest != [f"f_rest_{i}" for i in range(len(rest))] or len(rest) % 3:
        raise ValueError("Invalid SH fields in Gaussian PLY")
    basis_count = len(rest) // 3 + 1
    degree = math.isqrt(basis_count) - 1
    if (degree + 1) ** 2 != basis_count or not 0 <= degree <= 3:
        raise ValueError("PLY must use SH degree 0..3")

    def columns(keys):
        return np.column_stack([vertices[key] for key in keys]).astype(np.float32)

    shn = (
        columns(rest).reshape(len(vertices), 3, -1).transpose(0, 2, 1)
        if rest
        else np.empty((len(vertices), 0, 3), dtype=np.float32)
    )
    arrays = {
        "means": columns(["x", "y", "z"]),
        "scales": columns([f"scale_{i}" for i in range(3)]),
        "quats": columns([f"rot_{i}" for i in range(4)]),
        "opacities": np.asarray(vertices["opacity"], dtype=np.float32),
        "sh0": columns([f"f_dc_{i}" for i in range(3)])[:, None, :],
        "shn": shn,
    }
    model = GaussianModel({k: torch.from_numpy(v.copy()) for k, v in arrays.items()}, degree).to(
        device
    )
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return model, metadata


def load_model(path: str | Path, device: str = "cpu") -> tuple[GaussianModel, dict]:
    path = Path(path)
    if path.suffix == ".ply":
        return load_ply(path, device)
    if path.suffix == ".pt":
        model, state = load_checkpoint(path, device)
        # Renderers need run metadata, not another CPU copy of every parameter
        # and optimizer moment in a training checkpoint.
        for key in (
            "params",
            "optimizers",
            "strategy_state",
            "sampler_rng",
            "torch_rng",
            "cuda_rng",
        ):
            state.pop(key, None)
        return model, state
    raise ValueError("Expected a .pt checkpoint or a standard Gaussian .ply")
