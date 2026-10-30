# Benchmarks

Protocol (`spacegen bench`):
- every 8th image (sorted by name) is held out;
- indoor Mip-NeRF 360 scenes use `images_2`; Deep Blending uses the images as distributed
  in the 3DGS `tandt_db` archive;
- scores are PSNR, SSIM and LPIPS (VGG) on held-out views, with no colour correction;
- **cameras come from our own SfM**, not the dataset's COLMAP model, so these numbers
  measure the whole pipeline.

The published 3DGS numbers (Kerbl et al. 2023, 30k iterations) use the dataset's COLMAP
cameras.

## Results

Filled in as runs finish; see `results.md` in the benchmark output folder.

## Diagnostic runs on playroom (30 Oct 2026)

All held-out numbers are at 6k steps unless noted.

| Run | Cameras | Held-out PSNR | LPIPS | Finding |
| --- | --- | ---: | ---: | --- |
| Pipeline default, first version | ours | 21.9 | 0.52 | Low; train views also low (22.7), so it is not overfitting. |
| Same, official COLMAP cameras | dataset | 24.0 | 0.43 | Our SfM costs about 2 dB here. |
| + full resolution from step 0 | dataset | 24.5 | 0.42 | Coarse-to-fine costs about 0.5 dB at 6k steps. |
| + no per-image colour or pose | dataset | 24.7 | 0.42 | Corrections are neutral on this consistent-exposure set. |
| Depth-seeded start | ours | 11.9 | 0.63 | Seeds in free space grew into blobs in front of cameras. Seeding is now off by default. |
| After the fix (outlier cameras dropped, robust radius) | ours | 24.0 | 0.39 | Equal to the official cameras: the SfM gap is closed. |
| 30k steps, default (anti-aliased) | dataset | **30.24** | 0.305 | SSIM 0.909. Published 3DGS: 30.04 / 0.906 / 0.241. |
| 30k steps, classic rasterization, init scale 0.1 | dataset | 29.97 | 0.304 | Anti-aliasing stays the default. |
| Root cause of the SfM gap | ours | – | – | Seven "registered" images had no triangulated points, one of them 870 units away. Taken as the maximum camera distance, the scene radius scaled every learning rate about 100×. Fixed: such images are dropped, and the radius is a 95th percentile. |
