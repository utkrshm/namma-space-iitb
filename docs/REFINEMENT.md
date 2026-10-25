# Multi-view refinement: execution and handoff

## Status at session stop (26-10-26)

Implementation is on `fix/recon-mv-refinement`. **The first Coorg result is not
accepted as a good whole-scene reconstruction.** Real feature consistency improves
substantially, but the user and agent both inspected severe distinct entrance
layers. The wall behind the person improves. Subsequent regularization and sparse
texture anchors do not resolve the entrance. The latest detector-free matching
revision is implemented but has **not** been run or qualitatively verified.

The user requested an immediate session handoff. No reconstruction/inspection
process remains running. Original preprocessing, VGGT predictions, section graphs
and first-pass refinement exports are preserved.

## Setup

```sh
uv sync --extra alignment --extra reconstruction --extra viewer --extra revisits --extra floor --extra refinement --group dev
```

Use a CUDA device for learned matching. The optimizer can also use `--device cpu`.
ALIKED/LightGlue and DINOv2 use the existing pretrained model dependencies. Optional
detector-free anchors use real Kornia indoor LoFTR weights downloaded from
`https://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt` (44.2 MiB).

## Commands

Default separate refinement of an aligned section export:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python refine_multiview.py \
  outputs/coorg_bnb/section_alignment --output-dir outputs/coorg_bnb/multiview_refinement_NEW
```

The equivalent installed entry point is `uv run --no-sync namma-refine`.
Joint inference exports, e.g. `outputs/room_m_block/reconstruction`, use the same
interface. That second-capture execution is still pending.

**Next experiment to run**, preserving the result already inspected by the user:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python refine_multiview.py \
  outputs/coorg_bnb/section_alignment \
  --output-dir outputs/coorg_bnb/multiview_refinement/experiments/dense_links_v2 \
  --tracks outputs/coorg_bnb/multiview_refinement/tracks.npz --dense-anchors
```

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
- `dense_matching.py`: optional LoFTR across weak/disconnected groups. The first
  0.9-confidence experiment accepted zero anchors. A real model probe found its
  confidence distribution much lower (e.g. maximum 0.890 for frames 23/24). The
  **unrun latest revision** uses 0.5 confidence, reciprocal matches within 1.5
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

### First pass: `outputs/coorg_bnb/multiview_refinement/`

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

### Follow-up experiments

- `experiments/texture_anchors/`: first conservative texture-anchor experiment.
- `experiments/regularized_texture/`: 381 additional texture tracks / 850 observations
  with protected validation/test neighborhoods, plus dense displacement prior.
  Whole-scene held-out consistency still improves; entrance layering persists in
  actual-cloud inspection. User has been given this cloud's path too.
- `experiments/dense_anchors/`: completed first LoFTR attempt (0.9 cutoff), zero
  added observations, eight groups remain. **Do not confuse these outputs with the
  latest 0.5/reciprocal revision, which has not been executed.**

## Remaining plan, in order

1. Run `dense_links_v2` above and inspect accepted matches themselves, especially
   repeated window structures; do not accept false repeated-pattern links just to
   connect the graph. Check which entrance groups actually gain measured links.
2. Inspect actual same-observation geometry and the support-selected PLY from
   multiple angles. The entrance must improve while preserving the living-wall
   gain. Add per-group/surface diagnostics; pooled feature residuals alone are
   demonstrably insufficient. Expand inspection selection to include well-supported
   worst pairs (the current automatic selector can prioritize tiny one-match pairs).
3. If measured links remain weak, add justified cross-view geometric constraints
   or constrain unsupported correction modes. Do not treat independent component
   gauges or filtering-away surfaces as successful global reconstruction.
4. Audit geometry/support/camera-z and full source-pixel provenance on the selected
   final real export. Improve cache-setting validation before relying on implicit
   reuse after changing matcher parameters/split seed; explicit `--tracks` currently
   checks source identity and split integrity, not requested matcher settings.
5. Exercise the joint-input route on the supplied M Block capture, inspect it, and
   run real LoFTR-specific positive/negative controls and flow-protection checks.
6. Record the accepted configuration/results or remaining blocker in `STATE.md`.

## Checks completed before handoff

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync pytest tests/test_refinement.py tests/reconstruction -q
uv run --no-sync ruff check src/refinement tests/test_refinement.py src/__init__.py refine_multiview.py inspect_multiview.py
uv run --no-sync ruff format --check src/refinement tests/test_refinement.py src/__init__.py refine_multiview.py inspect_multiview.py
uv lock --check
git diff --check
```

Tests: **57 passed**. They include real analytic multi-camera optimization recovery,
whole-track splitting/conflicts, border/field interpolation, similarity-invariant
metrics, camera-z contracts and occlusion semantics. Existing reconstruction tests
also pass. Real Coorg matching/refinement/export/inspection ran. The latest dense
matcher revision, its dedicated real controls and second-capture execution remain pending.

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
