# Experiment documentation

This submission keeps the current implementation in `src/`. Earlier approaches are
retained as documentation, not as additional runnable pipelines or dependency groups.

| Experiment | Record | Outcome |
| --- | --- | --- |
| VGGT inference, floor and section alignment | [Reconstruction](../docs/RECONSTRUCTION.md) | Superseded by calibrated SfM for this submission |
| Loop chunks, bundle adjustment and square-wall constraints | [Loop closure](../docs/LOOP_CLOSURE.md) | Partial consistency gains; unresolved scene distortion |
| Multi-view camera/depth refinement | [Refinement](../docs/REFINEMENT.md) | Deferred; improved metrics did not meet the scene-quality goal |
| Dense cameras, loop gates and densification | [SfM](../docs/SFM.md) | Lessons incorporated into the current camera solve |
| Splat training, block merging, exports and tours | [Splatting](../docs/SPLATTING.md) | Results informed the current trainer and renderer |

The [experiment ledger](../docs/STATE.md) records successes, failures and deferred
work. [Decision records](../docs/PLANS.md) retain the rationale. Additional historical
documents and their original source revision identifiers are under
[`docs/experiments/`](../docs/experiments/README.md).

Commands in historical guides refer to the original branches. They are not setup
instructions for this submission; use the root [README](../README.md) to run it.
