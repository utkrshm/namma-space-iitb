# Coorg BnB — photoreal Gaussian splat (branch `feat/dense-sfm-splat`)

Final model: three time blocks (entrance, living room, dining) trained on one full-resolution SfM solution,
merged by visibility-weighted ownership and jointly fine-tuned. It was then fine-tuned again on 1,745 frames
(8 per second, the extra frames registered by PnP) with a monocular depth prior
(`outputs/coorg_bnb/splat_runs/blocks/merged_dense8_depth`).
How it was made: [docs/SFM.md](../../docs/SFM.md) and [docs/SPLATTING.md](../../docs/SPLATTING.md).

## Files

| File | What it is |
| --- | --- |
| `scene_ply_parts/` | **Full-quality splat**: 885k Gaussians, view-dependent colour (SH degree 3), standard 3DGS PLY, split into parts below GitHub's 100 MB limit. Reassemble below. |
| `scene.splat` | The same Gaussians in the 32-byte web format (view-independent colour, 27 MB) for browser viewers. |
| `coorg_flythrough.mp4` | 1387-frame smoothed fly-through along the capture path, rendered entirely from the splat. |
| `flythrough_frames.jpg` | Seven frames sampled from the fly-through. |
| `heldout_overview.jpg`, `heldout/` | Frames held out from training: real video, the previous VGGT splat and the new splat. |
| `metrics.json`, `web_export.json` | Held-out metrics of the trained model and of the pruned export. |
| `cameras.json` | All 1,745 cameras (OpenCV world-to-camera, refined training poses), usable with `render_gaussian_splat.py`. |
| `sfm/` | Full-resolution camera solution report, top-down plot and the PnP densification report (`densify.json`). |

Reassemble and check the PLY, then view it in SuperSplat (https://superspl.at/editor) or with the repository viewer:

```sh
cd results/coorg_bnb/scene_ply_parts
cat scene.ply.part-* > scene.ply && sha256sum -c scene.ply.sha256
uv run --no-sync python ../../../view_gaussian_splat.py scene.ply --cameras ../cameras.json
```

## Quality on the 20 held-out frames

These are the same frames the previous VGGT splat was scored on:

| Model | PSNR, raw / colour-corrected | SSIM | PSNR at 518 px | Worst view (raw) | Off-path floaters* |
| --- | --- | --- | --- | --- | --- |
| Previous VGGT splat (161 views at 294×518) | – | 0.59 | 17.07 dB | 11.5 dB | – |
| Best single full-house model | 19.05 / 21.6 dB | 0.80 | 19.5 dB | – | – |
| Blocks, merged and fine-tuned (927 frames) | 24.63 / 27.56 dB | 0.91 | 25.5 dB | 13.7 dB | 2.65% |
| **This model (+ 8 frames/s + depth prior)** | **24.71 / 27.04 dB** | **0.91** | **25.6 dB** | **19.5 dB** | **2.40%** |

\* Share of pixels rendered closer than 0.25 units in views shifted 0.3 units sideways off the capture path.

19 of the 20 held-out views beat the VGGT splat.

- **Gained:** the weakest views improved most. The 200 s shelf went from 13.7 to 19.5 dB raw, and the five weakest
  views average 20.0 dB, up from 18.6. There are fewer floaters off the capture path, and the blank ceiling at 72–80 s
  renders with straighter beams.
- **Lost:** colour-corrected PSNR dropped 0.5 dB. It comes mostly from the person on the sofa at 125 s, who moves
  between passes, and from small exposure and haze differences at 181 s.
- **Pruning:** keeping Gaussians with opacity ≥ 0.02 leaves the held-out PSNR unchanged (27.26 dB). Dropping
  view-dependent colour costs about 3.7 dB, which is what the `.splat` loses.

Remaining limits come from the capture. The 199–201 s frames are motion-blurred, the 72–80 s ceiling has no texture,
and no loop returns from the dining room. Scene units are arbitrary (SfM gauge).
