# Coorg BnB — photoreal Gaussian splat (branch `feat/dense-sfm-splat`)

Final model: three time blocks (entrance, living room, dining) trained on one full-resolution SfM solution, merged
by visibility-weighted ownership and jointly fine-tuned (`outputs/coorg_bnb/splat_runs/blocks/merged_finetune2`).
How it was made: [docs/SFM.md](../../docs/SFM.md) and [docs/SPLATTING.md](../../docs/SPLATTING.md).

## Files

| File | What it is |
| --- | --- |
| `scene_ply_parts/` | **Full-quality splat**: 1.44M Gaussians, view-dependent colour (SH degree 3), standard 3DGS PLY, split into parts below GitHub's 100 MB limit. Reassemble below. |
| `scene.splat` | The same Gaussians in the 32-byte web format (view-independent colour, 44 MB) for browser viewers. |
| `coorg_flythrough.mp4` | 1387-frame smoothed fly-through along the capture path, rendered entirely from the splat. |
| `flythrough_frames.jpg` | Seven frames sampled from the fly-through. |
| `heldout_overview.jpg`, `heldout/` | Frames held out from training: real video, the previous VGGT splat and the new splat. |
| `metrics.json`, `web_export.json` | Held-out metrics of the trained model and of the pruned export. |
| `cameras.json` | All 927 cameras (OpenCV world-to-camera, refined training poses), usable with `render_gaussian_splat.py`. |
| `sfm/` | Full-resolution camera solution report and top-down plot. |

Reassemble and check the PLY, then view it in SuperSplat (https://superspl.at/editor) or with the repository viewer:

```sh
cd results/coorg_bnb/scene_ply_parts
cat scene.ply.part-* > scene.ply && sha256sum -c scene.ply.sha256
uv run --no-sync python ../../../view_gaussian_splat.py scene.ply --cameras ../cameras.json
```

## Quality on the 20 held-out frames

These are the same frames the previous VGGT splat was scored on:

| Model | PSNR, raw / colour-corrected | SSIM | PSNR at 518 px |
| --- | --- | --- | --- |
| Previous VGGT splat (161 views at 294×518) | – | 0.59 | 17.07 dB |
| Best single full-house model | 19.05 / 21.6 dB | 0.80 | 19.5 dB |
| **This model** | **24.6 / 27.6 dB** | **0.91** | **25.5 dB** |

Pruning to opacity ≥ 0.02 leaves the held-out PSNR unchanged (27.73 dB). Dropping view-dependent colour costs
about 3.7 dB, which is what the `.splat` loses.

18 of the 20 held-out views beat the VGGT splat. The two that don't:

- **200 s shelf:** the training frames around 199–201 s are motion-blurred (sharpness 91–183, against more than 220
  elsewhere), so the splat learned a soft shelf.
- **12.8 s plain wall:** visually near-identical; the gap is mostly exposure.

The blank-ceiling pan at 72–80 s is also soft. Scene units are arbitrary (SfM gauge).
