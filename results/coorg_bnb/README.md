# Coorg BnB — photoreal Gaussian splat (branch `feat/dense-sfm-splat`)

The final model is `outputs/coorg_bnb/splat_runs/opencv/finetune_2`. It was made in four
steps:

1. **Blocks:** three time blocks (entrance, living room, dining) were trained on one
   full-resolution SfM solution, merged and jointly fine-tuned on 8 frames per second
   with a monocular depth prior.
2. **Re-solve (29 Oct):** the cameras were solved again with COLMAP's OPENCV lens model.
   The earlier one-coefficient lens model sank the entrance floor about 0.3 camera heights
   below the hall.
3. **Transfer:** the splat was moved onto the new cameras (`transfer_splat.py`).
4. **Fine-tune:** two passes, 12k + 8k steps.

How it was made: [docs/SFM.md](../../docs/SFM.md) and [docs/SPLATTING.md](../../docs/SPLATTING.md).

## Files

| File | What it is |
| --- | --- |
| `scene_ply_parts/` | **Full-quality splat**: 491k Gaussians, view-dependent colour (SH degree 3), standard 3DGS PLY (116 MB), split into parts below GitHub's 100 MB limit. Reassemble below. |
| `scene.splat` | The same Gaussians in the 32-byte web format (view-independent colour, 15 MB) for browser viewers. |
| `coorg_tour.mp4` | 3:24 tour of the whole model (H.264, 1920×1080), in five parts: (1) a top-down build-up as the camera walks; (2) a 360° orbit with the ceiling cut away; (3) a view from above the entrance, living hall and dining room; (4) **0:53–1:57: standing inside each room and looking around** at eye level (entrance, living hall from two spots, dining room); (5) an eye-level walkthrough beside the plan, with the current position marked. Made with `render_tour.py`. |
| `look_around_frames.jpg` | Six frames from part 4 of the tour. |
| `coorg_flythrough.mp4` | 2599-frame smoothed fly-through along the capture path, rendered entirely from the splat. |
| `flythrough_frames.jpg` | Seven frames sampled from the fly-through. |
| `heldout_overview.jpg`, `heldout/` | Frames held out from training: real video, the previous final model and this one. |
| `metrics.json`, `web_export.json` | Held-out metrics of the trained model and of the pruned export. |
| `cameras.json` | All 1,733 cameras (OpenCV world-to-camera, refined training poses), usable with `render_gaussian_splat.py`. |
| `sfm/` | OPENCV-lens camera solution report, top-down plot and the PnP densification report (`densify.json`). |

Reassemble and check the PLY, then view it in SuperSplat (https://superspl.at/editor) or with the repository viewer:

```sh
cd results/coorg_bnb/scene_ply_parts
cat scene.ply.part-* > scene.ply && sha256sum -c scene.ply.sha256
uv run --no-sync python ../../../view_gaussian_splat.py scene.ply --cameras ../cameras.json
```

## Quality on the 20 held-out frames

These are the same frames the previous VGGT splat was scored on:

| Model | PSNR, raw / colour-corrected | SSIM | PSNR at 518 px | Off-path floaters* |
| --- | --- | --- | --- | --- |
| Previous VGGT splat (161 views at 294×518) | – | 0.59 | 17.07 dB | – |
| Blocks, merged and fine-tuned (927 frames) | 24.63 / 27.56 dB | 0.91 | 25.5 dB | 2.65% |
| + 8 frames/s + depth prior (previous final) | 24.71 / 27.04 dB | 0.91 | 25.6 dB | 2.33% |
| **This model (OPENCV-lens cameras)** | **25.39 / 27.37 dB** | **0.90** | **26.3 dB** | **2.48%** |

\* Share of pixels rendered closer than 0.25 units in views shifted 0.3 units sideways off the capture path.

The table below uses per-view colour-corrected PSNR at half resolution:

| | Previous final | This model |
| --- | ---: | ---: |
| Mean | 27.3 dB | 27.7 dB |
| Worst view | 20.3 dB | 21.2 dB |
| Five weakest views | 23.4 dB | 24.7 dB |

- **Gained:** 12 of 20 views are better than the previous final model, with the largest gains at the weakest
  views: 50 s door +2.4 dB, 159 s +2.2 dB, 181 s +2.0 dB and 200 s +2.5 dB.
- **Lost:** four views that were already strong are 1–1.5 dB lower (59, 69, 99 and 125 s).
- **Geometry:** the floor is flat. Earlier, the entrance floor sat 0.3 camera heights low, and the tour's plan view
  showed a black hole there. The remaining step is about 0.07 camera heights.
- **Pruning:** keeping Gaussians with opacity ≥ 0.02 leaves the held-out PSNR unchanged (27.68 dB). Dropping
  view-dependent colour costs about 3.8 dB, which is what the `.splat` loses.

The look-around turns only through directions the phone filmed from where it stands, and
tilts down the way the phone did. A level 360° turn from the middle of a room shows walls
nobody filmed from there, which come out smeared, and dark gaps under the ceiling.
Training changes (opaque-surface and flatness terms, regrowing the Gaussians) were tried
on 30 Oct and did not fix those views, so the model is unchanged; see
[docs/SPLATTING.md](../../docs/SPLATTING.md).

Remaining limits come from the capture. The 199–201 s frames are motion-blurred, the 72–80 s ceiling has no texture,
and no loop returns from the dining room. Scene units are arbitrary (SfM gauge).
