# What goes wrong in indoor video captures, and what spacegen does about it

Sources:
- the Coorg BnB capture: about 30 experiments, 26–30 Oct 2026, recorded in the parent
  repo's `docs/PLANS.md` and `docs/SPLATTING.md`;
- the benchmark runs in [BENCHMARKS.md](BENCHMARKS.md);
- published work on indoor Gaussian splatting.

Each row is a failure, its cause, and the stage that handles it.

## Capture and frames

| Problem | Cause | What spacegen does |
| --- | --- | --- |
| Motion blur | Fast turns and indoor exposure times. Coorg's 199–201 s frames are unusable. | `ingest` picks the sharpest frame in each piece of motion and skips frames below 0.6× the local median sharpness when it has a choice. |
| Too few or too many frames | A fixed frame rate oversamples slow walking and undersamples fast turns. | Frames are spaced by optical-flow motion (`motion_step`), not time, with `max_gap_s` and `max_frames` caps. |
| Rotated or vertical video | Phone rotation metadata. | Display rotation is applied at decode. |
| Auto-exposure and white balance | Each frame has a different gain. Bright windows push the room darker. | A per-image affine colour map is trained with the splat and kept zero-mean, so the scene's own colours stay at the average exposure. Held-out views are also scored after a colour fit (`psnr_cc`). |
| Lens distortion | Phone lenses. One radial coefficient bent Coorg's entrance floor 0.3 camera heights. | COLMAP's `OPENCV` model (k1, k2, p1, p2) by default. |

## Camera solve

| Problem | Cause | What spacegen does |
| --- | --- | --- |
| False loops | Repeated grilles, tiles and identical doors match between different places. On Coorg, 10–20% of retrieval pairs were wrong by 100–180°, and the global mapper folded the house. | Retrieval pairs need 60 epipolar inliers, and a single homography may explain at most 70% of them (repeats are mostly planar). For video, a solve with only sequential pairs is made first, and a loop is kept only if its relative rotation agrees within 10°. Loops that join separate sequential models need twice the inliers. |
| Drift without loops | Long walks. | Retrieval pairs (DINOv2 global descriptors) close true loops. Per-image pose refinement during splatting takes up the remaining error. |
| "Registered" cameras with no points | GLOMAP can register an image that observes no triangulated points; its pose is then arbitrary. On playroom one camera sat 870 units away in an 8-unit room. | `load_solution` leaves out images with fewer than 30 points. |
| Wrong focal guess | COLMAP starts at 1.2 × the long side (2,304 px for 1080×1920 portrait video) and stores it as a prior. The phone lens was 1,050 px. Verification then called 29% of sequential pairs "uncalibrated", and relative poses came from a focal 2× too long. GLOMAP still reported 1.28 px, but it scattered the first 28 cameras and dropped the second half of the walk four units below the first. | The focal is estimated from the verified fundamental matrices (E = KᵀFK must have two equal singular values). If it differs from the guess by more than 15%, the cameras are rewritten and the pairs verified again. The estimate is within 1% of the dataset calibration on playroom. |
| Reprojection error hides a broken path | A solve can be self-consistent and still put stretches of a walk in the wrong place. | For video, `continuity` checks camera positions along capture order. Cameras far from all their temporal neighbours are "isolated" and are left out of training. Steps more than 12× the median are "jumps", which count against a mapper attempt. |
| Blank walls break tracking | No features. | Retrieval links the frames across the gap. If the global mapper registers less than 90% of frames, the incremental mapper is tried as well and the better result is kept. |

## Splat training

| Problem | Cause | What spacegen does |
| --- | --- | --- |
| One outlier camera wrecks training | The scene radius scales the position learning rate, MCMC noise and the scale clamp. Taken as the *maximum* camera distance, a single bad camera made it about 100× too large; playroom held-out PSNR fell to 22 dB, with fog. | The radius is the 95th-percentile camera distance from the median centre. |
| Fog, everything soft | MCMC position noise is lr × covariance in world units. gsplat assumes a scene radius of about 1, and Coorg's 11 made the noise about 120× too strong. | The noise lr is divided by radius². |
| Near-camera floaters, out of memory | Gaussians in front of a camera grow over thousands of tiles. | Screen-size pruning: a Gaussian wider than 15% of the image is handed to MCMC relocation. |
| Monocular-depth seed points | Each image's depth map is fitted to SfM points separately, so some seeds land in free space; they grew into blobs in front of other cameras (playroom: 11.9 dB against 21.9 dB at 6k steps). | Off by default (`splat.depth_seed_points = 0`). A safe version would keep only seeds consistent across several views. |
| Huge or needle Gaussians | Unbounded scales. | Scales are clamped to 10% of the scene extent, with an MCMC scale regularizer. |
| Blank walls and ceilings have no shape; glossy floors | Photometric loss is flat on textureless surfaces. | Monocular depth (Depth Anything V2 Small) with a patch-wise Pearson loss that fades to zero by step 15k. A constant weight cost 0.4 dB on Coorg; early shape followed by photometric detail does not. |
| Reflections (mirrors, glass, polished floors) | A mirror image is consistent virtual geometry *behind* the mirror. View-dependent colour needs high-order SH. | SH degree 3; on Coorg, degree 0 lost 3.7 dB. The depth prior is faded out, so it does not fight the virtual geometry. |
| Shimmer when zooming out | Aliasing of small Gaussians. | Anti-aliased rasterization (Mip-Splatting's 2D filter); tour frames are rendered at 1.5× and downsampled. |
| One budget for a whole house | Above about 1.6M Gaussians, Coorg's whole-house training was unstable, and a single model scored 21.6 dB against 27.6 dB for blocks. | Automatic spatial blocks (k-means on cameras, `max_images_per_block`), border cameras shared between blocks, visibility-weighted merge, joint fine-tune. |
| A room revisited at another time | Time-range blocks put one room into several blocks. | Blocks are spatial, and merge ownership is by observation strength (Σ 1/depth²), not nearest camera. |

## Free viewpoint, the hard part

| Problem | Cause | What spacegen does |
| --- | --- | --- |
| Smears and black gaps when turning in a room's middle | Nobody filmed those directions from there; the phone was tilted 20–35° down. 88% of Coorg's Gaussians were invisible in every training view, and removing them changed nothing. The smears are *visible* Gaussians fitted from other directions. | Each Gaussian records which of 64 directions the capture saw it from (`support`). Probe views at each area's standing point find unsupported or unrendered pixels; SDXL inpaints them; the generated pixels are distilled into the splat; the new Gaussians are flagged `filled`. |
| Fill that contradicts itself between views | Each inpainted view invents its own content. | Probes run in sequence. Seeded Gaussians count as supported from their probe's direction, so overlapping probes reuse earlier fills instead of inventing new ones. |
| Generated content passing as measured | — | `filled` flags in the PLY, a generated share per probe in `fill.json`, held-out PSNR re-scored after filling, and an honest report. |
| Training-only fixes | Opaque-surface and flatness terms, and MCMC regrowth (Coorg, 30 Oct). | Kept out of the defaults: they held the held-out score, or lost 2–3 dB, without fixing unseen directions. |

## Things to film better (no algorithm replaces these)

- In each room's middle, turn slowly through a full 360° at eye level, then once more
  tilted up toward the ceiling.
- Move slowly through doorways. Each room change should take several seconds.
- Return to the start, so the walk has at least one loop.
- Keep the light steady: avoid walking from a dark corridor straight into a sunlit room.
