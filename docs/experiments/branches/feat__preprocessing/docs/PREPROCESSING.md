# Running preprocessing

Preprocessing turns continuous smartphone video and optional patch clips into a
model-independent image package. It does not estimate poses, metric scale or depth.
See [the architecture contract](ARCHITECTURE.md#2-video-ingestion-and-preprocessing).

## Commands

Run these commands from the repository root. Source videos remain unchanged.

```sh
uv sync
uv run namma-preprocess videos/room_m_block.mp4
uv run python preprocess_capture.py videos/coorg_bnb.mp4
uv run namma-preprocess videos/room_m_block.mp4 --no-revisits
uv run pytest -rs
uv run ruff check src tests/preprocessing preprocess_capture.py
```

The default destination is `outputs/<first_clip_stem>/preprocessing/`. For multiple
clips, supply all paths and optionally set `--output-dir`. Clip stems must be unique
after replacing punctuation with underscores.

Use `--config configs/preprocessing.toml` for settings and `--no-resume` to force
all passes. An existing pass is reused only when source hashes, applicable config,
pass-specific code/record definitions, library versions and artifact hashes match. Missing or corrupted images
invalidate export. Changing JPEG quality reuses scoring and selection.
Changing only revisit verification also reuses probe, scoring and selection. The
whole-package hash is audit metadata, not an instruction to invalidate every pass.

### Optional revisit matching

```sh
uv sync --extra revisits
uv run --extra revisits namma-preprocess videos/coorg_bnb.mp4
uv run --extra revisits pytest -ra
```

This installs DINOv2, ALIKED and LightGlue dependencies. Model weights must be
available in the Hugging Face and Torch caches, or downloadable. The implementation
extracts descriptors one image at a time, releases DINOv2 before local matching,
and uses a bounded CPU feature cache. Both full videos have run on the 4 GB RTX
3050 with `device = "auto"` selecting CUDA. All accepted pairs are exported as
match images, with paths and timestamps recorded in `covisibility.json`.

The local verifier rejects descriptor ambiguity from repeated structures before
fitting the fundamental matrix. Each cross-image descriptor distance must be less
than `self_match_ratio` times the nearest distant same-image descriptor distance,
in **both** images. Descriptors are L2-normalized; same-image alternatives within
`self_match_radius_px` are excluded. Defaults are ratio **0.8** and radius **32 px**
at the 1024-pixel verification resolution. Only surviving matches count toward
inlier and spatial-spread thresholds. This may drop true links; thresholds are not
relaxed to preserve a particular door or luggage pair.

When the extra is absent or disabled, sequential tracking still runs and
`report.json` states `revisits.status = "not run"`, with a reason. No substitute
matcher is used. Import/model/runtime failures with an installed extra propagate
instead of silently becoming successful matching runs.

## Python handoff

```python
from src import PreprocessConfig, load_result, preprocess_capture

config = PreprocessConfig.from_toml("configs/preprocessing.toml")
result = preprocess_capture(
    ["videos/room_m_block.mp4"],
    "outputs/room_m_block/preprocessing",
    config=config,
)
result = load_result(result.output_dir)
for frame in result.geometry:
    print(frame.frame_id, frame.image_path, frame.rotation)
```

Passing `config=None` uses the typed built-in defaults, which initially match the
checked-in TOML. Pass the loaded TOML explicitly to pick up subsequent edits.
The CLI always loads the TOML. Image paths in selected records are relative to
`result.output_dir`; artifact paths in `result.artifacts` are absolute. Moving the
whole output folder is supported for loading exported observations; retrieving
unexported candidates also requires the unchanged source videos at recorded paths.

## Artifacts and measurements

- `clips.json`: whitelisted source metadata, checksums and original paths. Container
  GPS tags are never copied. `start_pts` and `time_base` define relative times.
- `all_frames.csv`: every decoded frame, including flagged frames. Empty `flow_px`
  means unknown motion; it must not be interpreted as zero. Flags are JSON arrays
  inside CSV fields. Schema 2 adds `edge_fraction` and `reblur_loss`; the run's
  schema version also covers this table. Schema 1 runs are recomputed automatically.
- `selection.json`: selected frames, all candidate IDs, temporal link evidence and
  per-frame tracking measurements. The short image buffer is replayed when an
  earlier frame becomes the new reference.
- `manifest.json`: exported geometry PNGs and optional appearance JPEGs, their pool,
  keyframe association and source-to-upright rotation. Appearance frames have no
  estimated poses. The appearance pool includes all geometry observations.
- `covisibility.json`: geometry nodes, supported sequential edges, optional verified
  revisit edges, connected components and debug match coordinates. Sequential edge
  `inliers` counts forward/backward-checked KLT tracks, not fundamental-matrix inliers.
  `ambiguous_matches` counts raw local matches removed by the descriptor guard;
  it defaults to zero for sequential/older edges. This is an additive schema-2 field.
- `report.json`: low-quality intervals, unsupported links, timing info and revisit
  status. Low-texture intervals are informational and do not count as quality
  failures. Neither a flag nor a disconnected component automatically mandates recapture.
  `revisits.pair_diagnostics` includes raw/ambiguous/unambiguous counts and acceptance;
  `revisits.rejected_repetitive` lists pairs that passed the unfiltered geometric
  check but failed after ambiguity removal. Diagnostic unfiltered inliers never
  supply an accepted edge. A rejected pair has zero *accepted* `inliers_after`,
  not a claim that it had no geometrically fitting points.
- `run.json` and `debug/`: input/config/artifact hashes, versions, original pass
  durations, resume status, timelines, paginated keyframe sheets and revisit matches.

Analysis uses 640-pixel-long-side greyscale derived from range-aware RGB. Scores,
flow and tracking coordinates are at that analysis resolution. The manifest records
actual x/y scale factors; pixel centers obey `x' = sx * (x + 0.5) - 0.5` and similarly
for y. Exported frames have full upright resolution, origin at the top left, x right
and y down. Only display rotation changes the exported image geometry.

## Current limitations

Sharpness cannot conclusively separate low texture, defocus and motion blur. Frames
with insufficient edge content receive informational `low_texture`, not
`suspected_blur`. Re-blur loss measures gradient energy lost on strong edge pixels
when blurring again. Low loss can support a blur flag; ambiguous loss requires
temporal corroboration, and temporal sharpness alone cannot flag low-texture frames.
Reliable flow supports diagnosis but is never a prerequisite for a blur flag.
The labelled 49.5 s door remains ambiguous using per-frame evidence alone at 640 px;
this limitation is preserved as a strict expected-failure test. The full pipeline
still flags that observation with temporal evidence. See [review results](PREPROCESSING_REVIEW.md).

Corners must exceed a whole-frame relative and absolute eigenvalue floor before
entering the grid. Reliability requires sufficient tracks and spatial coverage;
track survival remains a diagnostic rather than a reliability or selection gate.
The temporal selector measures track coverage and displacement, not 3D parallax.

Ordinary 33 ms VFR steps are not timing warnings. Steps of about 67 ms or longer
produce info flags. Frame identity always uses source PTS, never index/fps.

HDR is identified during probing and rejected at decode with an explicit request
for a consistent SDR conversion. Automatic tone mapping, lens correction, rolling
shutter correction, dynamic-object masking and reconstruction-feedback orchestration
are not implemented. Candidate IDs and source provenance support future recovery.

For a qualitative review, inspect the timelines and sheets alongside original
footage. Plain-wall flags may be texture-related, and physically rolling the phone
can leave tilted content even when its display rotation is applied correctly.
