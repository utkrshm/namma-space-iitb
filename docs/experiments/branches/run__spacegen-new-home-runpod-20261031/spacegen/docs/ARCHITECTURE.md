# spacegen architecture

`spacegen.pipeline.run` calls the stages in order. A stage whose output exists is
skipped, so runs resume.

## Conventions

These are also in `spacegen/contracts.py`.

- **World:** the SfM frame, in arbitrary units. `scene_frame.json` adds a rotation to a
  gravity-aligned frame (z up, floor at z = 0) and the length unit *camera height*.
- **Camera:** OpenCV axes (x right, y down, z forward). `world_to_camera` is 4×4.
  Intrinsics refer to the stated width × height, with integer pixel centres.
- **COLMAP:** pixel centres at (0.5, 0.5); `sfm.solve.load_solution` converts them.
- **gsplat:** samples at (u + ½, v + ½); `splat.model.render` converts at the boundary.

## Stage contracts

| Stage | Module | Input | Output |
| --- | --- | --- | --- |
| ingest | `ingest.py` | video file or image folder | `frames.json`: `name`, `index`, `position`, `timestamp_s`, `sharpness`, `width`, `height`. Images are upright, long side ≤ `ingest.max_long_side`. |
| sfm | `sfm/` | `frames/` | `undistorted/{images,sparse}`: COLMAP PINHOLE model of the largest reconstruction. `report.json` covers pairs, loop check, registration and reprojection error. `load_solution` returns `Solution` (cameras in capture order, image paths, points, colours, per-point views, frame records). |
| depth | `depth.py` | undistorted images | `depth/<stem>.npy`: float16 relative inverse depth at 1/`depth.downscale` resolution. |
| splat | `splat_stage.py`, `splat/`, `partition.py` | `Solution`, depth | `scene.ply` (3DGS layout, SH degree 3), `cameras.json` (refined train poses; `split` = train/val), `metrics.json` (held-out PSNR/SSIM/LPIPS and `psnr_cc`), `checkpoint.pt` (including per-image `corrections`). |

## Held-out split

Every 8th chosen frame (`frames.json` `position % 8 == 0`) is held out and never trained
on. For image folders this is the standard 3DGS split. Unregistered frames do not shift
it.

## Large captures

When there are more training images than `partition.max_images_per_block`, cameras are
clustered by position (k-means). Each block trains on its own cameras plus cameras that
see at least `partition.overlap` of its points. Gaussians go to the block whose own
cameras see them best (Σ 1/depth²). The merged model is fine-tuned on every image at
full resolution.
