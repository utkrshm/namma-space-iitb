# House example

Explore this scene as **Apartment 2** in the
[live viewer](https://gausswalk.pages.dev/#apartment-2).

This published house reconstruction comes from the earlier dense-camera/block-training
experiments. The original capture identifiers remain in the metadata for provenance.
The implementation used for new captures is the current `src/` pipeline.

| Artifact | Contents |
| --- | --- |
| `scene.splat` | Compact browser-viewer export, with view-independent colour |
| `scene_ply_parts/` | Full-SH PLY split below GitHub's per-file limit; 491,267 Gaussians |
| `coorg_tour.mp4` | 3:24 rendered tour |
| `coorg_flythrough.mp4` | Smoothed camera-path fly-through |
| `cameras.json` | Saved camera poses |
| `metrics.json`, `web_export.json` | Held-out scores and export measurements |
| `heldout/`, `heldout_overview.jpg` | Real-image/render comparisons |
| `sfm/` | Camera-solve and densification diagnostics |

## Open the full-colour model

From the repository root, reassemble into the ignored output directory and verify it:

```sh
mkdir -p outputs/house-example
cat example/house/scene_ply_parts/scene.ply.part-* > outputs/house-example/scene.ply
(cd outputs/house-example && sha256sum -c ../../example/house/scene_ply_parts/scene.ply.sha256)
```

Open the resulting PLY in a compatible Gaussian-splat viewer. The `.splat` export
is smaller but drops view-dependent colour; it is not a full-quality replacement
for the PLY. The web demo uses a compressed export that retains those colour bands.

The saved final-model measurements are 25.39 dB raw / 27.37 dB colour-corrected PSNR
and SSIM 0.90 on the historical 20-view evaluation. Scene units are arbitrary.
Motion blur, reflections, textureless regions and unobserved directions remain
limitations. See [SfM](../../docs/SFM.md) and [splatting](../../docs/SPLATTING.md)
for the experiment history.
