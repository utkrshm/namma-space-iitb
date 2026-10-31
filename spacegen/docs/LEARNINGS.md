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
| Blank walls and ceilings bumpy, with blobs in front of them; they smear as soon as the viewer moves | Photometric loss is flat on textureless surfaces. The old prior, a patch-wise Pearson correlation, ignores scale and offset, so a blob uniformly in front of a wall patch scored as well as the wall; it also faded to zero at step 15k. On vyas-room, held-out normals on blank walls were off by 38° on average. | Monocular depth (Depth Anything V2 Small) is fitted per image to the SfM points it sees (`depth/alignment.json`, median error 3.7% on vyas-room). Training compares *alpha-weighted* inverse depth with it on every pixel (texture-weighted), plus normals of rendered vs prior depth on textureless pixels. Both decay over the whole run. vyas-room at 10k steps: blank-wall normal error 37.5° → 17.9°, held-out depth error 6.8% → 3.5%, held-out PSNR 23.06 → 22.86 dB (colour-corrected 28.06 → 28.22, SSIM 0.876 → 0.877). Weights 0.1/0.05 kept PSNR exactly but left 20.7°. |
| Reflections and glass (mirrors, windows, glass table tops, polished floors) | A mirror image is consistent virtual geometry *behind* the mirror. Glass shows what is behind it and a reflection at once. View-dependent colour needs high-order SH. Blown-out windows are clipped by the sensor. | SH degree 3; on Coorg, degree 0 lost 3.7 dB. The metric depth prior puts window panes and grilles on the window plane instead of scattering them (vyas-room). Reflections in glass are then painted onto the surface: stable, no floaters, but no parallax. Clipped pixels only penalize a darker render (`splat.saturation_level`). |
| Shimmer when zooming out | Aliasing of small Gaussians. | Anti-aliased rasterization (Mip-Splatting's 2D filter); tour frames are rendered at 1.5× and downsampled. |
| One budget for a whole house | Above about 1.6M Gaussians, Coorg's whole-house training was unstable, and a single model scored 21.6 dB against 27.6 dB for blocks. | Automatic spatial blocks (k-means on cameras, `max_images_per_block`), border cameras shared between blocks, visibility-weighted merge, joint fine-tune. |
| A room revisited at another time | Time-range blocks put one room into several blocks. | Blocks are spatial, and merge ownership is by observation strength (Σ 1/depth²), not nearest camera. |

## Free viewpoint, the hard part

| Problem | Cause | What spacegen does |
| --- | --- | --- |
| Smears and black gaps when turning in a room's middle | Nobody filmed those directions from there. Support is per *direction*, not distance, so a wall filmed from 3 m away counted as supported from 0.5 m. A filled model's tour turned a full 360°: Area 2 of vyas-room had 37% unmeasured pixels, reported in `tour.json` and published anyway. The standing point (nearest the area's centre) had a 10° filmed arc. | Look-arounds stand where the capture turned most (`filmed_standing_points`), sweep only the yaw range filmed by cameras within 0.4 camera heights, at their tilt, and every 5° view must pass the checks *before* rendering (`_checked_arc`); otherwise the area is skipped. Gaps are still filled and flagged `filled` (see the fill stage), but never shown as a full turn. |
| Fill that contradicts itself between views | Each inpainted view invents its own content. | Probes run in sequence. Seeded Gaussians count as supported from their probe's direction, so overlapping probes reuse earlier fills instead of inventing new ones. |
| Generated content passing as measured | — | `filled` flags in the PLY, a generated share per probe in `fill.json`, held-out PSNR re-scored after filling, and an honest report. |
| Training-only fixes | Opaque-surface and flatness terms, and MCMC regrowth (Coorg, 30 Oct). | Kept out of the defaults: they held the held-out score, or lost 2–3 dB, without fixing unseen directions. |

## Process mistakes that cost a day

| Mistake | What caught it | Guard now |
| --- | --- | --- |
| The new depth prior reused the loop variable `decay`, which the position learning rate is multiplied by every step. Gaussians froze early; every prior variant lost ~2 dB regardless of weight, which looked like a prior problem. | Four ablations moving nothing; training-view PSNR dropping too. | `training.jsonl` logs `means_lr`, and a GPU test asserts it ends at `means_lr_final × radius` with the priors on. |
| Holes defined as alpha < 0.9. On a black training background, dark objects sit at alpha 0.6–0.9 and look right, so good views failed. | Marking the flagged pixels on the render. | `tour.HOLE_ALPHA = 0.5`. |
| Quality checks computed but not enforced. | The user's review of the tour. | Look-arounds are planned from checks; failing parts are cut, not published. |

## Things to film better (no algorithm replaces these)

- In each room's middle, turn slowly through a full 360° at eye level, then once more
  tilted up toward the ceiling.
- Move slowly through doorways. Each room change should take several seconds.
- Return to the start, so the walk has at least one loop.
- Keep the light steady: avoid walking from a dark corridor straight into a sunlit room.
