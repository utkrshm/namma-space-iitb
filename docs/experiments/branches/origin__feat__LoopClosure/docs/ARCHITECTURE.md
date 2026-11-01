# Planned architecture

The diagram outlines the components of the proposed capture-to-navigation pipeline. Their responsibilities are explained below; project scope and required outcomes are defined in [GOALS.md](GOALS.md).

```text
                        SMARTPHONE
                            │
                            │ RGB video
                            ▼
                  ┌─────────────────────┐
                  │ 1. CAPTURE / SOP    │
                  │                     │
                  │ Camera trajectory   │
                  │ Exposure / motion   │
                  │ Coverage            │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 2. VIDEO INGESTION  │
                  │                     │
                  │ Decode video        │
                  │ Frame extraction    │
                  │ Quality filtering   │
                  │ Keyframe selection  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 3. 3D PERCEPTION    │
                  │                     │
                  │ VGGT / VGGT-Ω      │
                  │                     │
                  │ • camera poses     │
                  │ • intrinsics       │
                  │ • depth            │
                  │ • point maps       │
                  │ • tracks           │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼────────────────┐
              │              │                │
              ▼              ▼                ▼
          Camera poses      Depth         Point cloud
              │              │                │
              │              │                │
              └──────────────┼────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ 4. GEOMETRIC        │
                  │    RECONSTRUCTION    │
                  │                     │
                  │ Point fusion        │
                  │ Outlier removal     │
                  │ Surface estimation  │
                  │ Mesh generation     │
                  └──────────┬──────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Dense visual geometry      Scene geometry
                 │                       │
                 ▼                       ▼
            3D Gaussian Splatting    Mesh / TSDF
                 │                       │
                 ▼                       │
          Photorealistic scene          │
                 │                       │
                 └───────────┬───────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 5. SCENE            │
                  │    UNDERSTANDING     │
                  │                     │
                  │ Object detection   │
                  │ Instance masks     │
                  │ Semantic labels    │
                  │ Open-vocabulary    │
                  │ embeddings         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 6. 3D SEMANTIC MAP  │
                  │                     │
                  │ Objects ↔ 3D       │
                  │ geometry           │
                  │                     │
                  │ "chair_12"         │
                  │ position           │
                  │ extent             │
                  │ category           │
                  │ embedding          │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 7. POI / SCENE      │
                  │    DATABASE          │
                  │                     │
                  │ Meeting Room 4     │
                  │ Sofa               │
                  │ Reception          │
                  │ Printer            │
                  │ Conference Room    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ 8. NAVIGATION GRAPH │
                  │                     │
                  │ Walkable geometry  │
                  │ Obstacle geometry  │
                  │ NavMesh / graph    │
                  │                     │
                  │ A* / shortest path │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              WEB WALKTHROUGH    PHONE NAVIGATION
                    │                 │
                    ▼                 ▼
              3DGS renderer      Localization
                    │                 │
                    ▼                 ▼
              First-person       Destination
                roaming          guidance
```

## 1. Capture / SOP

The capture Standard Operating Procedure guides how a person records an indoor space with a smartphone. It covers camera movement, exposure, motion and spatial coverage so that the footage contains usable, overlapping observations of rooms and their connections.

## 2. Video ingestion and preprocessing

This stage decodes the video and extracts frames while preserving their source identity and timing. Quality filtering identifies unusable observations, such as severely blurred frames. Keyframe selection reduces redundant views while retaining useful coverage and connections between parts of the scene. This stage is frozen after verification on the supplied captures.

The stage is model-independent: it does not resize, reorder or group frames for a particular reconstruction model. The default input is one continuous video per venue with deliberate revisits; short patch clips may be added. The package is `src/preprocessing`, and its public entry point is `preprocess_capture(clips, output_dir, config=None, resume=True)`. Thresholds are in `configs/preprocessing.toml`.

It runs in five passes (probe plus four processing passes), each writing an artifact that later passes can resume from:

| Pass | Modules | Output |
| --- | --- | --- |
| 0. Probe | `probe.py` | `clips.json`: rotation, time base, frame-rate variation, colour/HDR, size, duration |
| 1. Score every frame (640 px greyscale) | `decode.py`, `scoring.py`, `tracking.py` | `all_frames.csv`: sharpness, exposure, timestamp gaps, motion, tracking reliability, flags |
| 2. Select | `selection.py`, `tracking.py` | `selection.json`: geometry keyframes, denser appearance frames, candidate pool |
| 3. Link | `revisits.py`, `graph.py`, `report.py` | `covisibility.json`, `report.json` |
| 4. Export (full resolution) | `storage.py`, `export.py`, `visualize.py` | frame images, `manifest.json`, `debug/` plots |

Shared record types and their JSON forms are defined in `contracts.py`; `config.py` loads the configuration.

### Output contract

```text
outputs/<capture_id>/preprocessing/
  run.json             config, input hashes, library versions, timings per pass
  clips.json           per-clip metadata
  all_frames.csv       one row per decoded frame: scores and flags
  selection.json       selection decisions, including the candidate pool
  manifest.json        selected frames: provenance, pool, keyframe link, reason, image path
  covisibility.json    nodes = geometry keyframes; sequential and revisit edges with measured weights
  report.json          low-quality intervals, tracking breaks, weak links, unlinked clips, loop links
  geometry_frames/     PNG, full resolution
  appearance_frames/   JPEG (quality configurable), full resolution
  debug/               timeline plot, contact sheets, revisit match images
```

Conventions:

- **Frame IDs:** `<clip_id>:<pts>`. The PTS is the stream's presentation timestamp in its time base; times are in seconds from the stream start.
- **Transform:** the only image transform applied is the clip's display rotation. Exported frames are upright, with the pixel origin at the top left, x pointing right and y pointing down. Analysis-resolution measurements record their scale relative to the exported frame.
- **Candidate pool:** unexported candidates are stored as IDs and can be decoded again from the unchanged source video.
- **Appearance frames:** these are linked to their geometry keyframe but have no camera poses; pose recovery happens downstream.
- **Consumers:** the reconstruction stage reads `manifest.json`, `covisibility.json` and the frame folders through `load_result(output_dir)`. Model-specific preparation happens in that stage's adapters.

Commands, Python usage, measurement semantics and current limitations are in
[Running preprocessing](PREPROCESSING.md). The implemented core handles SDR clips;
HDR is detected and requires an explicit SDR conversion before decoding. Revisit
matching is optional and reports “not run” when its dependencies are absent.

Schema 2 records per-frame edge fraction and re-blur loss in addition to sharpness.
Insufficient texture produces an informational annotation, not a quality failure.
Corner strength is screened against the whole frame before grid assignment;
reliability uses surviving track count and spatial coverage, not survival ratio.
Every accepted revisit has a debug match image and recorded timestamps/path.

Local revisit matches must also beat spatially distant same-image descriptor
alternatives on both sides (configured L2 distance ratio). Only unambiguous matches
enter geometric inlier and spread checks. `Edge.ambiguous_matches` counts discarded
raw local matches (zero for sequential edges); `revisits.rejected_repetitive` records
pairs that would pass without this guard. Pass-specific source fingerprints allow
link/export changes to reuse unchanged upstream artifacts.

## 3. 3D perception

The geometric front end estimates spatial information from the selected images. Camera poses describe camera position and orientation; intrinsics describe image projection; depth describes the distance to observed surfaces. Point maps associate image pixels with 3D positions, while tracks associate observations of the same point across views.

VGGT and VGGT-Ω are examples shown in the diagram. The available predictions, and whether point maps or tracks are produced directly or derived separately, depend on the selected method.

### Implemented VGGT interface

`src/reconstruction/` provides `VGGTInference`, with explicit `load()`, reusable
`predict()` and `unload()` methods. Each call jointly infers one image or an ordered
sequence representing one scene on a CUDA GPU with sufficient memory. Both captures
have run on the RTX A4000 machine; settings and limitations are recorded in
[Project state](STATE.md). Torch, weights and Open3D are loaded lazily.

`load_preprocessed_frames()` consumes the frozen `load_result()` handoff, defaults
to geometry frames, and orders views by clip then source PTS. Model-specific
preparation resizes and pads without cropping. Frame IDs, timestamps, selected-frame
rotation metadata, valid-image masks and pixel-centre transforms are retained.
`reconstruct_preprocessed()` passes those views jointly to VGGT; graph optimization
and independent-chunk alignment are not part of this adapter.

The CPU `ReconstructionResult` includes model-grid RGB, depth/confidence, camera
encodings and decoded matrices, direct point maps/confidence, depth-derived world
points and optional tracks/visibility/confidence. OpenCV extrinsics map world to
camera (`X_cam = R X_world + t`); inverse poses and upright-source intrinsics are
also supplied. Geometry uses arbitrary scene units, not metric scale.

The root script `get_recon.py` and `namma-reconstruct` entry point save processed
arrays/provenance and a filtered RGB point cloud under
`outputs/<capture_id>/reconstruction/`. Open3D exports and displays the cloud.
`--no-view` supports headless GPU export; `--view existing.ply` and standalone
`view_point_cloud.py PATH.ply` view saved geometry without model loading.
Array shapes, preparation and commands are documented in
[Running VGGT reconstruction](RECONSTRUCTION.md).

### Optional floor-plane postprocessing

`src/reconstruction/floor.py` and root `align_floor.py` provide a separate
`floor_aligned/` export. A pretrained semantic model identifies floor/rug pixels;
confidence-filtered world observations support robust scene/local floor planes.
Global rigid leveling establishes Z-up/floor Z=0. Optional per-view rigid
floor constraints assume one flat floor, interpolating plane estimates when the
floor is unseen. This does not perform bundle adjustment, metric calibration or
general multi-view registration. The original reconstruction remains in its original
frame; aligned clouds, per-frame transforms and adjusted cameras are explicit
separate artifacts. See [floor detection and alignment](RECONSTRUCTION.md#floor-detection-and-alignment).

### Overlapping high-resolution section alignment

`src/reconstruction/sections.py` adds an explicit alternative to whole-capture joint
inference. It reconstructs overlapping source-frame windows sequentially, records
GPU peak memory and cache provenance, and retains each window's original predictions.
`section_registration.py` estimates positive-scale, proper-rotation Sim(3) transforms
from identical source-frame/model-pixel observations using balanced RANSAC and robust
weighted refinement, with tolerance scaled by camera-to-observation distance.
Independent whole interior overlap frames validate each connection; shared endpoint
views remain separate stress diagnostics and are excluded from fitting.
`section_benchmark.py` revalidates image revisits and their spatially distributed 3D
support before they become graph constraints. The first section fixes the gauge;
robust global optimization jointly estimates section-local-to-global similarities.

`section_evaluation.py` tests multiscale robust point-to-plane ICP at the established
scale. A correction is applied only when exact held-out correspondences improve and
absolute section-support checks still pass. The optimized graph is checked again
on held-out overlaps. This is section-level registration, not per-frame bundle
adjustment: internally inconsistent views can remain within an accepted section.

The root `align_sections.py` orchestrates inference and alignment; separate
`validate_section_alignment.py` and `inspect_section_alignment.py` provide real
image-feature comparison against the original reconstruction and headless inspection
of the saved PLY. Coorg execution writes exclusively under
`outputs/coorg_bnb/section_alignment/`. Every source frame contributes observations
from one owner section, selected by distance from section boundaries. No spatial
observation fusion is performed. Original preprocessing and reconstruction artifacts
remain separate. See [section alignment](RECONSTRUCTION.md#high-resolution-section-alignment).

At this boundary, `X_global = s R X_section + t`, in arbitrary scene units.
Camera centers receive the similarity; camera orientations receive only `R`, and
camera z-depth receives `s`. Exported inverse extrinsics therefore remain rigid.
Each point retains source frame index, model-pixel identity, confidence and owner
section; source timestamps/rotations/image transforms remain in the manifest and
per-section metadata. The PLY contains XYZ/RGB; its NPZ companion carries provenance.

### Loop closure (`src/loop_closure/`)

The loop-closure stage reads a finished `section_alignment/` directory **read-only** and
writes to a separate directory (default `outputs/<capture>/loop_closure/`). It shares
one loop detection step: DINOv2 retrieval of frame pairs more than 20 s apart, verified
with ALIKED/LightGlue on the shared 518-pixel model grid. Each verified pair is
assigned a role: a method-A chunk centre, a training pair, or a held-out pair that no
method uses.

- **Method A (VGGT-Long-style loop chunks):** VGGT jointly infers the frames around both
  ends of a loop. The chunk is registered to each owner section on exact shared pixels,
  and the composed Sim(3) becomes a `loop_chunk` edge in the existing section graph.
  Held-out frames and the loop pair's own feature matches gate each edge.
- **Method B (bundle adjustment):** per-frame pose and log depth-scale increments plus
  track points minimize robust reprojection error, with a log-depth prior to VGGT depth.
  It starts from the baseline graph and, if A ran, from A as well.

Every method exports `cameras.npz` (`frame_ids`, `owners`, `camera_to_world` 4×4,
`depth_scale`, model-grid `intrinsics`) and a `point_cloud.npz|ply` in the section-export
schema. A frame's points are `camera_to_world · (depth_scale · owner depth)`. All methods
are compared on identical correspondences in `comparison.json`. See
[LOOP_CLOSURE.md](LOOP_CLOSURE.md).

## 4. Geometric reconstruction

Reconstruction combines the estimated observations into a coherent scene representation. It aligns and fuses points, removes inconsistent observations, estimates surfaces and generates geometry suitable for downstream use. The visual representation and the geometry used for spatial reasoning serve different purposes.

The implemented previews concatenate confidence-filtered observations in VGGT's joint
scene frame or the explicitly optimized section frame, excluding padded/invalid
geometry and optionally limiting the point count. They are point-cloud previews,
not fused surfaces or navigation meshes.

### Photorealistic representation: 3D Gaussian splatting

Gaussian splatting represents the scene using 3D Gaussian primitives whose appearance is fitted to captured images using estimated cameras. Rendering these primitives supports views from positions other than the original capture viewpoints. Its role is to provide the visual appearance of the digital twin.

### Scene geometry: mesh / TSDF

Explicit geometry represents surfaces and spatial boundaries for collision checking and navigation. A mesh describes surfaces with vertices and faces. A truncated signed distance field (TSDF) stores distances near observed surfaces and can support fusion and mesh extraction. This representation supplies geometric information that a photorealistic rendering alone does not establish.

## 5. Scene understanding

Scene understanding adds meaning to the reconstructed space. Object detections locate items, instance masks distinguish their visible regions, and semantic labels describe their categories. Open-vocabulary embeddings support matching user descriptions to scene content beyond a fixed list of labels.

## 6. 3D semantic map

The semantic map connects object and region descriptions to their spatial locations. It associates observations across views so that an item seen repeatedly can be represented as one scene instance, with a position, extent, category and searchable description or embedding.

## 7. POI / scene database

The scene database organises named destinations such as reception, meeting rooms and printers. It supports tagging, lookup and selection of those destinations in the viewer. Labels may be assigned by a person or assisted by scene understanding; a visual category alone does not establish a venue-specific name such as “Meeting Room 4”.

## 8. Navigation graph

Navigation represents where a visitor can walk and which areas are blocked by walls or furniture. A navigation mesh or graph connects traversable regions and destination locations. A pathfinding method, such as A*, searches this representation to produce a floor-constrained route to a selected POI.

## User-facing applications

### Web walkthrough

The web application presents the visual scene, POI interaction, search and route guidance. First-person exploration works independently of active navigation, using desktop keyboard/mouse controls and suitable mobile controls. Scene geometry supports movement constraints, while the visual representation provides the rendered environment.

### Phone navigation: optional AR

The phone-navigation branch represents the optional live-camera AR experience, distinct from mobile web exploration. Localisation connects the phone's current pose to the reconstructed scene so that destination guidance can be placed in the camera view and updated as the user moves.
