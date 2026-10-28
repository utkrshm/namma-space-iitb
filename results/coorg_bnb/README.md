# Coorg BnB — photoreal Gaussian splat (feat/dense-sfm-splat)

- `scene_pruned.ply` (491 MB, not in git: GitHub limits files to 100 MB; locally at `outputs/coorg_bnb/splat_runs/blocks/merged_finetune/web/`) — 2.08M Gaussians, full view-dependent colour (SH degree 3), standard 3DGS PLY.
  Open it in SuperSplat (https://superspl.at/editor) or with the repository viewer:
  `uv run --no-sync python view_gaussian_splat.py scene_pruned.ply --cameras cameras.json`.
- `scene.splat` — the same scene in the compact 32-byte web format (view-independent colour) for browser viewers.
- `coorg_flythrough.mp4` — 1387-frame smoothed fly-through along the capture path, rendered from the splat.
- `heldout/`, `heldout_overview.jpg` — frames held out from training: real video | previous VGGT splat | new splat.
- `metrics.json`, `web_export.json` — held-out metrics. On the 20 held-out frames: 23.9 dB raw / 27.0 dB
  colour-corrected, SSIM 0.91 (previous VGGT splat: 17.07 dB / SSIM 0.59, at the 518-pixel grid where the new splat scores 24.6 dB).
- `sfm/` — the full-resolution camera solution report and top-down plot (927 frames, 1.31 px).

Scene units are arbitrary (SfM gauge); cameras are OpenCV world-to-camera.
How it was made: docs/SFM.md and docs/SPLATTING.md on the branch.
