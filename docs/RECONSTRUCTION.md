# Running VGGT reconstruction

## Deferred experimental direction

Per-view camera/depth refinement is a promising direction, but was explored too
early; current work is focused on stabilizing base reconstruction. The experiment
is preserved on `fix/recon-mv-refinement` at commit `6c7d168`, not implemented here.
It improved measured correspondence and some living-wall appearance, but unsupported
components and entrance layering remain unresolved. These exploratory results do not
establish superiority or readiness for integration. Revisit after base reconstruction
is stable. Detailed handoff: `git show fix/recon-mv-refinement:docs/REFINEMENT.md`.

The reconstruction stage jointly infers cameras, depth and point maps from one or
more images. `get_recon.py` consumes a frozen preprocessing directory, saves the
processed outputs, and displays a coloured point cloud with Open3D.

## Setup and execution

Use a **larger CUDA GPU** with enough memory for all selected views in one call.
The wrapper does not distribute inference across GPUs or align independent chunks.
Inputs follow the [frozen preprocessing contract](PREPROCESSING.md).

Run from the repository root on the GPU machine:

```sh
git submodule update --init src/reconstruction/deps/vggt
uv sync --extra reconstruction --extra viewer
uv run --extra reconstruction --extra viewer python get_recon.py outputs/room_m_block/preprocessing
```

On a headless GPU machine, save without opening a window:

```sh
uv run --extra reconstruction --extra viewer python get_recon.py \
  outputs/room_m_block/preprocessing --device cuda:0 --no-view
```

The default destination is `outputs/room_m_block/reconstruction/`. Override it with
`--output-dir outputs/YOUR_RUN/reconstruction`. The CLI rejects output destinations
inside the frozen preprocessing directory.

The equivalent console entry point is:

```sh
uv run --extra reconstruction --extra viewer namma-reconstruct \
  outputs/room_m_block/preprocessing --no-view
```

The wrapper lazily imports an installed `vggt` package, if present, otherwise the
checked-in submodule. The reconstruction extra declares the used runtime dependencies
directly, avoiding the upstream demo stack, `numpy<2` pin and GUI OpenCV package.
The existing project uses NumPy 2 and headless OpenCV. Pretrained inference and
Open3D PLY export have run successfully on both supplied captures.

On Linux, the project sources PyTorch and torchvision from the official CUDA 12.6
index. This avoids the default CUDA 13 wheel's driver incompatibility on the RTX
A4000 machine (driver 535.309.01). CUDA 12.x minor-version compatibility is used;
the system CUDA version displayed by `nvidia-smi` need not equal the wheel runtime.
To retain all pipeline dependencies in one environment:

```sh
uv sync --extra revisits --extra reconstruction --extra viewer --group dev
```

Weights for `facebook/VGGT-1B` download on the first `load()` or `predict()` call.
`model_id` can instead identify a local pretrained directory. Use
`local_files_only=True` / `--local-files-only` to require cached/local weights.

## Viewing an exported cloud

Copy `point_cloud.ply` to a machine with a graphical display. This path needs only
the viewer extra and never loads VGGT or weights:

```sh
uv sync --extra viewer
uv run --extra viewer python view_point_cloud.py outputs/room_m_block/reconstruction/point_cloud.ply
uv run --extra viewer python view_point_cloud.py outputs/coorg_bnb/reconstruction/point_cloud.ply
```

`view_point_cloud.py` is a standalone root entry point using the existing Open3D
viewer module. It accepts a point-cloud path and optional `--title`, without loading
Torch, VGGT or model weights. `get_recon.py --view PATH.ply` remains supported.
If all extras are already installed, use `uv run --no-sync python view_point_cloud.py
PATH.ply` to preserve that environment. The standalone viewer has not been executed;
interactive qualitative inspection is left to the user.

Open3D supports mouse orbit, pan and zoom; press `H` for controls. The viewer
preserves exported XYZ without estimating gravity or flipping axes.

### Generated capture clouds (21-10-26)

Both captures were preprocessed separately, then reconstructed using
`facebook/VGGT-1B`, CUDA on an RTX A4000 (16 GB), automatic **bfloat16** precision,
the complete geometry pool, stride 1 and no frame-count limit:

```sh
uv run --no-sync python get_recon.py outputs/room_m_block/preprocessing --device cuda:0 --no-view
uv run --no-sync python get_recon.py outputs/coorg_bnb/preprocessing --device cuda:0 --image-size 252 --no-view
```

| Capture | Joint views | Padded model resolution | Exported points | Cloud |
| --- | ---: | --- | ---: | --- |
| Room | 36 | 518×518 | 1,000,000 | `outputs/room_m_block/reconstruction/point_cloud.ply` |
| Coorg | 161 | 252×252 | 1,000,000 | `outputs/coorg_bnb/reconstruction/point_cloud.ply` |

Coorg at the default 518 resolution exhausted GPU memory. Its successful run
explicitly reduces resolution, preserving all 161 views in one scene coordinate
system. Room uses the default resolution. Both exports use depth-derived points,
discard the bottom confidence quartile and cap the cloud at one million points.
Every selected view contributes points. PLY XYZ and RGB were read back with
Open3D and matched exactly to the saved NPZ; camera matrices and cloud coordinates
are finite. No interactive viewer or qualitative reconstructed-scene inspection
was run. Outputs are local generated artifacts under the repository's ignored
`outputs/` directory.

## Artifacts

| Artifact | Contents |
| --- | --- |
| `predictions.npz` | All processed arrays: both point-map variants, cameras, depth, confidence, pose iterations, model-grid RGB and optional tracks. |
| `metadata.json` | Array shapes, conventions, model identifier, precision, ordered frame identities, timing, selection/rotation records and image transforms. |
| `point_cloud.ply` | Filtered coloured point cloud readable by Open3D. |
| `point_cloud.npz` | The same cloud's XYZ, uint8 RGB, confidence, frame indices and model-pixel XY. |
| `preview.json` | View selection, cloud source, filter settings and counts. |

## Floor detection and alignment

`align_floor.py` exports a separate floor-aligned cloud from a saved reconstruction.
It preserves the original files and point count, RGB, confidence, frame indices
and model-pixel identities. Install its semantic-model and Open3D dependencies:

```sh
uv sync --extra revisits --extra reconstruction --extra viewer --extra floor --group dev
```

The implementation in `src/reconstruction/floor.py`:

1. Runs pretrained `nvidia/segformer-b2-finetuned-ade-512-512` on saved model-grid
   RGB. Pixels must be classified as **floor** or **rug**, with combined probability
   above **0.6**. Padding is excluded and a **5×5 erosion** removes uncertain borders.
2. Reads the corresponding saved world points and head confidence. It retains
   candidate floor observations above each frame's **40th confidence percentile**,
   sampling at most **4,000** points per view. The source matches `preview.json`:
   depth-derived points or direct point-head predictions, never a mixture.
3. Fits a dominant scene plane and local planes using **1,000-iteration RANSAC**.
   The distance threshold is **0.5%** of the median camera-to-floor-observation
   distance, in arbitrary scene units. Local planes require at least **200** sampled
   points, **60%** inliers, broad two-dimensional support and a normal within **25°**
   of the dominant plane. Normals face towards the observing camera.
4. In `--mode rigid` (default), applies one rigid rotation/translation to the whole
   scene, giving **Z-up and floor Z=0**. Scene dimensions and inter-view alignment
   are preserved. This levels the floor but cannot remove reconstruction drift.
5. In `--mode per-view`, additionally rotates each view's entire geometry minimally
   about its camera and translates vertically to make its estimated floor plane
   agree with the reference. Missing local planes are interpolated in capture time;
   the first/last estimate is held outside the anchor range. Local geometry is rigid,
   not individually flattened or scaled. Adjusted camera poses are exported too.

**Assumption:** per-view correction is for one genuinely flat, single-storey floor.
It does not support steps or separate storeys and is not bundle adjustment. It may
change cross-view alignment of walls/objects. Independent venues share the floor
convention, not an XY origin or a known metric scale. Semantic errors and distorted
depth can leave floor outliers; masks are detection hypotheses, not ground truth.

The runs for these captures used visually inspected masks:

```sh
uv run --no-sync python align_floor.py outputs/room_m_block/reconstruction --mode rigid --floor-masks outputs/room_m_block/floor_diagnostics/semantic_masks.npz
uv run --no-sync python align_floor.py outputs/coorg_bnb/reconstruction --mode per-view --floor-masks outputs/coorg_bnb/floor_diagnostics/semantic_masks.npz
```

Omit `--floor-masks` to run segmentation from scratch (`--device cpu` is supported
for segmentation). A cached mask archive must match the prediction grid and identify
the supported semantic model. When present, cached frame IDs and RGB hashes are
also checked; use archives exported by this command for provenance-checked reuse.
`--output-dir` overrides the default sibling `floor_aligned/` directory.

| Capture | Mode | Reliable local planes | Points |
| --- | --- | ---: | ---: |
| Room | Global rigid leveling | 7 / 36 | 1,000,000 |
| Coorg | Per-view floor constraint | 90 / 161 | 1,000,000 |

For Coorg's **reliable-view fitting samples**, median / p95 absolute floor height
improved from **0.01693 / 0.06824** to **0.000840 / 0.005185** scene units.
Over **all 736,706 confident semantic floor pixels**, including rejected/interpolated
views and outliers, median / p90 / p95 improved from **0.01708 / 0.05843 / 0.07504**
to **0.001017 / 0.009548 / 0.04554**. The longest gap between reliable floor anchors
is **27.18 s**. Room's geometry is unchanged apart from its coordinate transform.

A separate real ALIKED/LightGlue diagnostic on 36 adjacent-frame pairs (7,773
ambiguity-filtered matched observations) measured median 3D disagreement
**0.12430 → 0.12699** and p90 **0.54203 → 0.54342**. This is a floor-consistency
correction, not an improvement in general multi-view correspondence accuracy.
Headless side-view plots confirm a much tighter dominant Coorg floor band, with
remaining outliers. Interactive Open3D viewing has not been run by the agent.

### Aligned outputs and viewing

Each `outputs/<capture>/floor_aligned/` contains:

- `point_cloud.ply` / `point_cloud.npz`: corrected observations, original colors and provenance.
- `alignment.npz`: global and per-view original-world-to-aligned transforms, estimated
  floor planes, reliable-frame mask, adjusted camera poses/extrinsics and frame IDs.
- `alignment.json`: plane support, interpolation status, settings, coordinate contract,
  source hashes and residuals. `X_aligned = T[:3,:3] @ X_original + T[:3,3]`.
- `semantic_masks.npz`: floor masks, frame IDs, model name and model-RGB checksum.
- `validation.json`, `floor_comparison.png`: numeric verification and inspected headless
  comparison plots from this execution; these are diagnostic artifacts, not CLI outputs.
  Coorg also has `correspondence_validation.json` for the matched-feature diagnostic.

The original `predictions.npz` and camera arrays retain the original coordinate frame.
Use the exported transforms and adjusted cameras when combining them with an aligned
cloud. The CLI does not duplicate or rewrite dense prediction archives.

```sh
uv run --no-sync python view_point_cloud.py outputs/coorg_bnb/floor_aligned/point_cloud.ply
uv run --no-sync python view_point_cloud.py outputs/room_m_block/floor_aligned/point_cloud.ply
```

Read processed arrays without Torch or VGGT:

```python
import numpy as np

with np.load("outputs/room_m_block/reconstruction/predictions.npz", allow_pickle=False) as data:
    depth = data["depth"]
    intrinsics = data["intrinsics"]
    camera_to_world = data["camera_to_world"]
```

## Python interface

### Reusing a loaded model

Construction is lightweight. `load()` is explicit and idempotent; `predict()` also
loads lazily. Only CUDA devices are accepted, with no CPU fallback.

```python
from src import VGGTInference

model = VGGTInference(device="cuda:0", precision="auto").load()
try:
    single = model.predict("frame0.png")
    result = model.predict(["frame0.png", "frame1.png", "frame2.png"])
    depth = result.depth
    cameras = result.cameras
    cloud = result.point_cloud(source="depth")
    all_outputs = result.to_dict()
finally:
    model.unload()
```

Replace paths with existing images. Inputs may also be a single uint8 `H,W,3`
**RGB** array or a sequence of arrays/paths. Each call represents one scene, not a
batch of unrelated scenes, and establishes its own coordinate system.

`precision="auto"` chooses bfloat16 when supported by the selected device, otherwise
float16. `float32` disables the outer autocast. Camera/depth/point heads keep their
upstream precision behaviour. `unload()` releases the wrapper's model reference;
Torch may retain cached CUDA allocations until reused or the process exits.

### Consuming preprocessed frames

```python
from src import VGGTInference, reconstruct_preprocessed, save_point_cloud, show_point_cloud

model = VGGTInference(device="cuda:0")
try:
    result = reconstruct_preprocessed("outputs/room_m_block/preprocessing", model)
finally:
    model.unload()

result.save("outputs/room_m_block/reconstruction")
cloud = result.point_cloud(confidence_percentile=25, max_points=1_000_000)
save_point_cloud(cloud, "outputs/room_m_block/reconstruction/point_cloud.ply")
show_point_cloud(cloud)
```

`load_preprocessed_frames()` is the separate preparation API. It accepts a directory
or `PreprocessResult` and returns an `ImageBatch` without loading VGGT. It consumes
the existing `load_result()` handoff, including covisibility/report records, but does
not use graph edges to reorder, align or optimize predictions.

Frames follow clip-manifest order, then source PTS order. The default pool is
`geometry`. `pool="appearance"` selects that pool alone; it already includes geometry
observations. Output provenance retains source PTS, timestamp relative to clip start,
time base and the complete selected-frame rotation record.

Selection is explicit, applied in this order:

1. `frame_ids=[...]`: filter identities without reordering (Python API).
2. `stride=2` / `--stride 2`: take every second remaining frame.
3. `max_frames=8` / `--max-frames 8`: keep the first eight remaining frames.

All selected views run together. CUDA out-of-memory errors propagate; there is no
silent subsampling or independent chunking. A first-N limit previews a partial scene.

### Tracking query points

Tracking requires first-image query points. Supply `N,2` XY in **model-image pixels**,
or map upright exported-image pixels first:

```python
import numpy as np
from src import VGGTInference, load_preprocessed_frames

batch = load_preprocessed_frames("outputs/room_m_block/preprocessing")
source_xy = np.array([[100, 200], [150, 250]], dtype=np.float32)
model_xy = batch.transforms[0].transform_points(source_xy)
model = VGGTInference(device="cuda:0")
try:
    result = model.predict(batch, query_points=model_xy)
finally:
    model.unload()

tracks = result.tracks
visibility = result.track_visibility
confidence = result.track_confidence
```

Queries outside the first model image or in padding fail validation. Map tracks
back with each frame's `transform.transform_points(track_xy, inverse=True)`.

## Output contract

All result arrays live on the CPU. `S` is the view count, `H,W` the model image
dimensions, `I` the camera-refinement iteration count and `N` the query count.
Single-image results retain `S=1`.

| Attribute / array key | Shape | Meaning |
| --- | --- | --- |
| `images` | `S,H,W,3` | uint8 RGB including white padding. |
| `valid_mask` | `S,H,W` | Boolean image-content mask; excludes padding, not invalid predictions. |
| `depth` | `S,H,W` | Camera z-depth, not Euclidean ray distance. |
| `depth_confidence` | `S,H,W` | Depth-head confidence. |
| `world_points` | `S,H,W,3` | Direct point-head world XYZ. |
| `world_points_confidence` | `S,H,W` | Point-head confidence. |
| `depth_world_points` | `S,H,W,3` | Depth unprojected using decoded cameras. |
| `pose_encoding` | `S,9` | Final upstream camera encoding. |
| `pose_encoding_iterations` | `I,S,9` | All camera-refinement encodings. |
| `cameras.intrinsics` / `intrinsics` | `S,3,3` | Intrinsics in model pixels. |
| `cameras.original_intrinsics` / `original_intrinsics` | `S,3,3` | Intrinsics in upright exported-image pixels. |
| `cameras.world_to_camera` / `world_to_camera` | `S,3,4` | OpenCV extrinsics `[R|t]`. |
| `cameras.camera_to_world` / `camera_to_world` | `S,4,4` | Homogeneous inverse poses. |
| `source_to_model` (dictionary/archive) | `S,3,3` | Upright source-to-model pixel transforms. |
| `tracks` | `S,N,2` | Model-pixel XY; absent without queries. |
| `track_visibility`, `track_confidence` | `S,N` | Tracking scores; absent without queries. |

Geometry is float32; masks are boolean and RGB is uint8. Camera axes are x right,
y down, z forward. Extrinsics obey `X_camera = R @ X_world + t`. Geometry shares
one scene frame per call with **arbitrary scale, not metres**.

### Image transformations

Inputs must already be upright. The loader does not reapply display rotation or
EXIF orientation. It resizes the longest side to 518 by default, rounds the other
side to a multiple of 14 (minimum 14), then centres it in a white square. PIL bicubic
resampling follows VGGT's pad-mode preparation. `image_size` changes the padded side
length and must be a positive multiple of 14.

Each `ImageTransform` records source, resized and model `(width,height)` and
`(left,top)` offsets. Integer pixel centres obey
`x_model = sx * (x_source + 0.5) - 0.5 + left`, similarly for y. Original intrinsics
are `inverse(source_to_model) @ model_intrinsics`. Source-to-upright rotation remains
separately recorded in `FrameProvenance.selected_frame`.

### Point-cloud processing

`result.point_cloud()` defaults to depth-derived geometry. `source="point_head"`
uses direct points and that head's confidence. The method removes padding,
nonfinite coordinates/confidence, and nonpositive depth (or points behind their
camera for direct points).

By default it discards the bottom 25th confidence percentile over eligible scene
observations. `confidence_percentile=0` retains all eligible observations; optional
`min_confidence` adds an absolute cutoff. Scores are uncalibrated, not probabilities.
`max_points` deterministically subsamples flattened eligible pixels after filtering:
the CLI defaults to one million points, the Python method to no cap.

`PointCloud` contains `points`, `colors`, `confidence`, `frame_indices` and `pixel_xy`.
Overlapping observations remain: this is concatenated filtered geometry, not surface
fusion. Outlier removal, meshing, scale calibration and navigation are downstream
work. An empty cloud is inspectable in Python; Open3D export/viewing rejects it.

## Verification

Run CPU-side checks without model inference:

```sh
uv run pytest tests/reconstruction -q -ra
uv run --extra viewer pytest tests/reconstruction -q -ra
uv run ruff check src/reconstruction/*.py src/__init__.py tests/reconstruction get_recon.py
uv run python get_recon.py --help
```

Tests exercise real image preparation, analytic camera/depth reprojection, filtering,
serialization, lazy imports and a real short-video preprocessing handoff. The Open3D
test performs a real coloured PLY round trip with the viewer extra; otherwise it is
explicitly skipped. No fake model substitutes for pretrained inference.

Pretrained CUDA inference and headless PLY export/readback have now run on the RTX
A4000 machine, as recorded above. No tests were added or run for this execution and
viewer request. The standalone viewer was linted but not executed. Interactive
reconstructed-capture inspection remains pending user review. See
[Project state](STATE.md) for verification results and blockers.

Upstream references: [VGGT](https://github.com/facebookresearch/vggt) and
[Open3D point-cloud documentation](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html).
