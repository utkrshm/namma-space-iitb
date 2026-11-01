# Multi-view refinement: execution and handoff

## Current execution status (26-10-26)

Implementation and experiments are preserved on `fix/recon-mv-refinement`. This is
a **promising direction, but explored too early**: defer further refinement until
base reconstruction is stabilized, then reassess how it should interact with that
stage. Active work is moving to `feat/reconstruction`; no evidence establishes that
branch as superior. Dense Coorg matching, constrained-pose export, joint-input M Block
refinement, matcher controls and real export audits have run. Targeted test, lint,
format and lock checks are confirmed. **Neither scene is accepted as a navigation-ready
reconstruction.**
Coorg's living wall improves, but entrance/upper-wall layering and green floor patches
remain. M Block's bed/room structure improves slightly, but holes remain. The
execution checklist below distinguishes completed verification from failed scene
quality acceptance. This document distinguishes historical outputs on the previous
machine from the current machine's regenerated baseline.

## Setup

```sh
uv sync --extra alignment --extra reconstruction --extra viewer --extra revisits --extra floor --extra refinement --group dev
```

Use a CUDA device for learned matching. The optimizer can also use `--device cpu`.
ALIKED/LightGlue and DINOv2 use the existing pretrained model dependencies. Optional
detector-free anchors use real Kornia indoor LoFTR weights downloaded from
`https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt` (44.2 MiB).

## Commands

Default separate refinement of an aligned section export (fresh output path):

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python refine_multiview.py \
  outputs/coorg_bnb/section_alignment --output-dir outputs/coorg_bnb/multiview_refinement_NEW
```

The equivalent installed entry point is `uv run --no-sync namma-refine`.
Joint inference exports, e.g. `outputs/room_m_block/reconstruction`, use the same
interface. The current-machine M Block joint-input execution is recorded below.

The current Coorg dense-anchor execution, already run, was:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python refine_multiview.py \
  outputs/coorg_bnb/section_alignment \
  --output-dir outputs/coorg_bnb/multiview_refinement/experiments/dense_links_v2 \
  --tracks outputs/coorg_bnb/multiview_refinement/tracks.npz --dense-anchors
```

The constrained-pose export reused the resulting effective track set, not the
pre-dense base tracks:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python refine_multiview.py \
  outputs/coorg_bnb/section_alignment \
  --output-dir outputs/coorg_bnb/multiview_refinement/experiments/constrained_pose \
  --tracks outputs/coorg_bnb/multiview_refinement/experiments/dense_links_v2/dense_matches/tracks_c53000b9c55f7c84.npz \
  --field-steps 0 --geometry-prior-weight 0.2
```

`--export-only` consumes saved `effective_tracks.npz` when present (or the legacy
single unambiguous base/dense tracks cache) and saved corrections; it does not run
the matcher again. The original dense run's effective augmentation is represented by
`dense_matches/tracks_c53000b9c55f7c84.npz`; do not assume that run has an
`effective_tracks.npz`. New runs save `effective_tracks.npz`. Implicit
cache reuse validates requested matching settings and rejects mismatches. An explicit
`--tracks` override is permitted and recorded in manifest provenance; it validates
source identity/split integrity but intentionally may supply different matcher
settings. Keep the requested configuration and effective track artifact together.

Optional `--flow-anchors` adds real, conservative pyramidal Lucas–Kanade tracks.
`--matching-only` stops after observation construction. `--pose-steps`,
`--field-steps`, `--grid-size`, `--geometry-prior-weight`, `--learning-rate`, and
`--max-points` are explicit controls. `--export-only` re-exports saved corrections
after checking the source fingerprint. It does not rerun optimization.

View an exported cloud:

```sh
uv run --no-sync python view_point_cloud.py outputs/coorg_bnb/multiview_refinement/point_cloud.ply
uv run --no-sync python view_point_cloud.py outputs/coorg_bnb/multiview_refinement/geometry_after.ply
```

Generate actual-PLY inspection sheets, overriding the saved absolute input path
if the outputs were transferred to a new machine:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python inspect_multiview.py \
  outputs/coorg_bnb/multiview_refinement --input-dir outputs/coorg_bnb/section_alignment
```

Audit exported arrays, source metadata, pixel/provenance consistency and the saved
PLYs. The script writes `audit_report.json` and fails on inconsistencies:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python validate_multiview.py \
  outputs/coorg_bnb/multiview_refinement/experiments/dense_links_v2 \
  --input-dir outputs/coorg_bnb/section_alignment
```

Run real LoFTR positive/negative controls and the flow held-out protection audit:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python validate_refinement_matching.py \
  --input-dir outputs/coorg_bnb/section_alignment \
  --tracks outputs/coorg_bnb/multiview_refinement/tracks.npz \
  --output-dir outputs/coorg_bnb/multiview_refinement/validation_controls \
  --negative-input-dir outputs/room_m_block/reconstruction
```

These are the actual executed control paths; M Block's negative input is its joint
`reconstruction/` export, not a section-alignment directory.

## Implemented procedure and boundaries

- `src/refinement/inputs.py`: adapt joint or section-aligned depth predictions to
  common OpenCV cameras and depth units. Apply the section similarity scale to
  camera-z depth before refinement. Hash source arrays/provenance for cached evidence.
- `matching.py`: real ALIKED/LightGlue, DINOv2 candidate retrieval, ambiguity/image
  geometry checks, conflict-free multi-frame tracks. Whole tracks are split into
  70% training / 15% validation / 15% test. Exact detector identities do not cross
  splits. Source-image hashes and matcher settings identify pair caches.
- `optimization.py`: robust shared-point and reprojection objectives, individual
  camera rotation/translation and positive depth scales, then a zero-mean smooth
  spatial log-depth field. Reference camera/scale anchors fix observed components.
  Parameter, field-smoothness and dense input-world-displacement priors restrain
  unsupported deformation. Checkpoints use validation only; test is reported separately.
- `flow.py`: optional additional image texture anchors. Forward/backward flow,
  patch appearance/uniqueness and image geometric support are required. Validation
  and test feature neighborhoods are protected; new tracks enter training only.
- `dense_matching.py`: optional LoFTR across weak/disconnected groups. Historically, the first
  0.9-confidence experiment accepted zero anchors. A real model probe found its
  confidence distribution much lower (e.g. maximum 0.890 for frames 23/24). The
  revision subsequently executed in `dense_links_v2` uses 0.5 confidence, reciprocal matches within 1.5
  matcher pixels, protected held-out neighborhoods, broad image support and robust
  epipolar checks. Candidate selection now includes compatible initial camera views
  across each component pair, rather than only nearby temporal gaps.
- `export.py`: preserve observation identity and RGB; no spatial averaging/fusion.
  Save matched-observation geometry comparisons separately from spatial sampling
  and multi-view support selection. Relative z agreement checks include occlusion
  handling; visibility agreement is not proof of true geometry on glass/reflections.
- `inspection.py`: read actual PLYs, assert PLY/NPZ XYZ/RGB equality, use common
  before/after framing and source-camera comparisons. First-camera upright display
  is not estimated gravity. Overview clipping counts are recorded.

No Coorg indices, floor-plane constraints, Manhattan assumptions or metric units
are embedded in the algorithm. Generalization is an interface/design property;
quality across other videos has not yet been demonstrated.

## Output contract

`X_world = R_c2w @ (z * inverse(K) @ [u,v,1]) + C` and
`z_new = z_initial * exp(log_scale + bilinear_field(u,v))`.
The field covers actual image content, with constant extension at its edges.
Coordinates remain arbitrary input-scene units, not metres.

| Artifact | Meaning |
| --- | --- |
| `manifest.json` | Source fingerprint, source frame/timing/rotation records, pixel transforms, content bounds, configuration. |
| `tracks.npz`, `tracks.json`, `matches/` | Original learned observation identities/splits and real match evidence. Optional texture/dense evidence is stored separately. |
| `corrections.npz`, `refinement_report.json` | Refined cameras, log-scales/fields, initial cameras, training support, gauge anchors, validation history and test/per-pair results. |
| `depths.npz`, `cameras.npz` | Refined camera-z maps, masks, intrinsics and camera/extrinsic arrays. |
| `geometry_before/after.ply` and `.npz` | Exactly the original cloud's pixels; isolates geometric changes. |
| `unfiltered_before/after.ply` and `.npz` | Same seeded spatial samples, all original confidence levels; isolates sampling from filtering. |
| `supported_before.ply`, `point_cloud.ply` and `.npz` | Original/refined multi-view-supported observations; support and contradiction counts in NPZ. |
| `export_report.json`, `inspection/` | Per-frame retention, recovered below-quartile observations, actual-PLY backed visual diagnostics. |

## Real results retained

### Historical first pass (previous machine): `outputs/coorg_bnb/multiview_refinement/`

- 161 input views; 660 candidate pairs; 252 supported pairs;
  29,409 learned tracks / 78,861 observations; 848 conflicting merges rejected.
- Training graph has eight anchored components, including three views with no
  learned observations (24, 37, 140). The separate window/ceiling groups matter:
  low residuals inside them do not establish cross-group alignment.
- Untouched test: 7,469 paired 3D observations, normalized median/p90
  **0.014908/0.055694 → 0.002722/0.012388**. Reprojection median/p90
  **5.399/23.305 → 0.827/2.272 model pixels**. Scene-distance median changes
  0.9188 → 0.8497; current-depth-normalized errors are reported too.
- Transition frames 112/113: 55 untouched test pairs, median/p90
  **0.245938/0.354392 → 0.005527/0.015513**, using the fixed input denominator.
- **Visual outcome:** improved wall/furniture consistency in some living regions,
  but severe entrance layering and displaced green floor patches. This result is
  not accepted. Original-observation clouds make the regression visible independently
  of point density/support filtering.

### Historical follow-up experiments (previous machine)

- `experiments/texture_anchors/`: first conservative texture-anchor experiment.
- `experiments/regularized_texture/`: 381 additional texture tracks / 850 observations
  with protected validation/test neighborhoods, plus dense displacement prior.
  Whole-scene held-out consistency still improves; entrance layering persists in
  actual-cloud inspection. User has been given this cloud's path too.
- `experiments/dense_anchors/`: first LoFTR attempt (0.9 cutoff), zero added
  observations. This is distinct from the executed current-machine `dense_links_v2`.

### Current-machine runs

- The earlier machine's outputs were **not transferred**. Main regenerated the
  baseline under the current defaults: 29,409 tracks / 78,861 observations, 660
  candidates / 252 supported pairs. Baseline test median/p90:
  **0.014908441925989 / 0.055694211609**.
- `experiments/dense_links_v2/` added **363 links / 726 observations**: pair counts
  23/24=104, 24/25=70, 36/37=83, 37/38=59, 140/141=47. Components decreased 8→3,
  with anchors 0, 51, 140; there remains no link to isolated views 51–57 or dining
  views 140–160. Two match sheets from five accepted pair groups were inspected: anchors on shoes/floor
  look plausible, but the repeated grille remains ambiguous and is not independently
  accuracy-validated. Actual PLY review found improved living-wall consistency but
  rejected the scene for persistent entrance/upper-wall layering and green floor
  patches.
- Baseline refinement test median/p90 changes **0.014908441925989 / 0.055694211609 →
  0.002817065372553119 / 0.01319972200159**. Dense-anchor test median/p90 changes
  **0.0149222511 / 0.0557457992 → 0.00289809773 / 0.01325093729**; its denominator
  changes with the added observations. The constrained-pose export used
  `--field-steps 0 --geometry-prior-weight 0.2`
  and exact dense tracks `tracks_c53000b9c55f7c84.npz`. Its selected state is the
  initial checkpoint, **not a newly improved reconstruction**. Test median/p90 before
  and after are unchanged at **0.0149222511 / 0.0557457992**
  in its initial selected state (no gain; this is not the fixed-original-denominator
  comparison). Dense Coorg export:
  **1,673,762 points**.
- M Block joint route completed for 36 frames. After the OpenCV remap batch fix,
  export resumed with `--export-only`; it used saved effective tracks, not a matcher
  rerun. For **2,254** test pairs, median/p90 changed **0.024690008895 /
  0.070772898169 → 0.003012746528 / 0.009096651970**. One training component is
  anchored at view 0. Export contains **1,395,236 points**. Main sees slight bed/room
  improvement, but remaining holes make it unsuitable for navigation.
- All four `audit_report.json` reports passed: regenerated baseline, dense Coorg,
  constrained-pose Coorg and M Block. Dense Coorg also passed the enhanced
  exact-metadata audit. The dense held-out protection report compared
  baseline **78,861** to augmented **79,587** observations: **726** added across five
  pairs, original arrays/pairs unchanged, all additions training-only, **0** held-out
  mask hits, nearest held-out distance **10.1782547 px** (median **46.5522 px**).
  Main inspected PLY-backed `geometry_comparison.jpg` (three angles), `supported_comparison`,
  and `camera_comparison_00.jpg` for dense Coorg; component views 51 and 140; M Block
  `geometry_comparison.jpg` and `camera_comparison_04`; and constrained-pose geometry
  (initial/identity state). Dense-match sheets `dense_matches_00.jpg` and
  `dense_matches_04.jpg` were inspected (five accepted pair groups, not five sheets).
  Sampled measured matches look plausible,
  but repetitive grille ambiguity remains and there is no independent truth validation.
- Real controls: positive accepted **400** warped matches over a **24×16 px** grid,
  residual median/p90 **0.76069 / 1.19252**; blank control **0**, cross-capture
  control **0**. Flow added **850** training-only observations and had **0** held-out
  mask hits. The minimum Euclidean distance was **9.393 px** after integer-rounded
  masks, so this is not evidence of a continuous radius ≥10 px.
- Confirmed tests: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/test_refinement.py tests/test_refinement_cache.py tests/test_refinement_controls.py tests/test_refinement_inspection.py tests/test_refinement_validation.py tests/reconstruction -q` — **88 passed in 53.87 s**.
- Targeted `uv run --no-sync ruff check` on `src/refinement`, the five test files,
  `src/__init__.py`, and the four refinement/validation scripts passed. Ruff format
  check on the 23 affected Python files passed; `uv lock --check` and
  `git diff --check` passed. Separate unscoped `uv run --no-sync ruff check`
  reported **1,943 pre-existing vendored-submodule findings** under
  `src/reconstruction/deps`; full-repository Ruff therefore does not pass.

## Execution checklist against the original acceptance goals

1. **Execute dense cross-group matching — executed, graph coverage incomplete.**
   Five pair groups were accepted; graph components fell 8→3, but views 51–57 and
   dining 140–160 remain without a connecting link. Repetitive grille ambiguity
   remains a quality risk.
2. **Inspect accepted matches and actual geometry — inspection completed; quality
   goal failed.** Main inspected the dense PLY geometry, supported comparison, camera
   and component views; M Block geometry/camera views; constrained-pose geometry; and
   the two inspected dense-match sheets from five accepted pair groups. Living wall improves, but entrance/upper-wall
   layers and green floor patches persist. Sampled match plausibility is not independent
   truth validation; no whole-scene acceptance.
3. **Evaluate constrained correction and the second capture — executed; goal failed.**
   Coorg selected the initial checkpoint and is not a newly improved solution. M Block
   improves some room structure slightly but has holes and is not navigation-ready.
4. **Audit output integrity and held-out protections — passed for all four outputs.**
   Dense held-out audit confirms 726 added
   training-only observations, unchanged original arrays/pairs, zero mask hits and
   nearest held-out distance 10.1782547 px (median 46.5522 px).
5. **Run real matcher controls — executed with limits.** Positive/blank/cross-capture
   outcomes and flow additions are recorded above. Integer-rounded mask distance
   does not establish a continuous 10 px exclusion radius.
6. **Finalize verification — execution complete; quality acceptance failed.** Targeted
   tests and scoped lint/format/lock checks passed; visual inspection, dense held-out
   audit, and all four export audits completed.
   Entrance/upper-wall layering and floor patches remain a blocking quality failure;
   improved held-out metrics do not change that verdict.

## Deferred follow-up, after base reconstruction stabilizes

1. Investigate why entrance/upper-wall geometry remains layered despite cross-group
   evidence. Do not accept false repeated-pattern links just to connect the graph.
2. Add justified cross-view constraints or constrain unsupported correction modes
   where evidence supports it. Do not treat independent component gauges or
   filtering-away surfaces as successful global reconstruction.
3. Reassess scene quality against actual source views; do not advance to surface fusion
   or navigation until entrance geometry and coverage are acceptable.

## Historical checks (previous machine; not the current combined test run)

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/test_refinement.py tests/reconstruction -q
uv run --no-sync ruff check src/refinement tests/test_refinement.py src/__init__.py refine_multiview.py inspect_multiview.py
uv run --no-sync ruff format --check src/refinement tests/test_refinement.py src/__init__.py refine_multiview.py inspect_multiview.py
uv lock --check
git diff --check
```

Historical tests: **57 passed**. Current-machine confirmed checks are recorded above.
The historical suite included real analytic multi-camera optimization recovery,
whole-track splitting/conflicts, border/field interpolation, similarity-invariant
metrics, camera-z contracts and occlusion semantics. Existing reconstruction tests
also pass.

## Moving to another machine

Git/chat exports do not include ignored generated artifacts or cached weights.
Transfer the repository branch/commit and the relevant `videos/` and `outputs/`
directories separately. To resume the next experiment without rerunning VGGT,
retain at least Coorg's preprocessing frame exports, the complete `section_alignment/`
predictions/metadata/graph/cloud, and the first refinement's `tracks.npz` and
`tracks.json`. Keep the full first refinement directory for comparison with the
user-inspected regression. Absolute source-image paths in old metadata can fall
back to the standard sibling preprocessing directory; leave original metadata
unchanged if reusing its fingerprint-checked tracks.
