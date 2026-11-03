# Project plans

Record substantive decisions and their implementation steps here before execution. Timestamps use `DD-MM-YY HH:MM:SS IST`, with 24-hour time in Indian Standard Time (`Asia/Kolkata`, UTC+05:30). Latest entries get recoreded first, stale entries come later.

## Entry format

```text
## DD-MM-YY HH:MM:SS IST — Title
- Status: proposed / adopted / superseded
- Decision:
- Rationale and trade-offs:
- Affected components:
- Implementation steps:
- Completion criteria:
```

## 03-11-26 23:55:00 IST — spacegen: plane prior for walls, floors and ceilings (experiment)

- **Status:** implemented behind `splat.planar_prior` (off); A/B queued
  (`outputs/spacegen/ab_queue.sh`, after the GPU tests and the m block retrain).
- **Decision:** per training frame, segment `wall`, `floor` and `ceiling` (Mask2Former,
  ADE20K, the segmenter already used for windows and mirrors), fit up to two planes per
  large region to the frame's depth target (RANSAC in camera space, inliers within 1.5% of
  their distance), and use the plane depth there as trusted depth: full depth weight and
  a normal target on every plane pixel (`spacegen.planar`, `depth/planar/<stem>.npz`). A
  region whose planes do not explain 60% of its depth keeps its depth.
- **Rationale and trade-offs:** blank walls still carry floaters that smear off the
  filmed path (m block, Coorg), and monocular depth bends on them. PlanarGS, AtlasGS and
  3D Gaussian Flats report that plane constraints fix low-texture indoor geometry where
  flattened Gaussians alone do not. Risks: a wrong segmentation flattening furniture
  (guarded by the inlier share), and plane pixels outvoting texture on posters (wanted).
- **Affected components:** new `spacegen/planar.py`; `splat/train.py` (`Priors` returns
  a 4th element `planar`, `prior_losses`, `drop_mirror_depths`); `splat_stage.py`;
  `config.py`; new `scripts/experiment.py` (10k-step A/B runs and per-frame comparison).
- **Completion criteria:** on vyas_room_v4 and room_m_block_v4 at 10k steps, held-out
  PSNR within 0.1 dB of the default or better, fewer frames losing > 1 dB, and visibly
  flatter walls off the path; only then switch it on by default.

## 03-11-26 23:25:00 IST — spacegen: fixes from the code review of `spacegen/src`

- **Status:** adopted; CPU tests pass. GPU tests are written but not run: the GPU has been
  down since 03-11-26 ~18:50 IST (`outputs/automation/GPU_DOWN.md`).
- **Decision:** fix what the review (`/code-review high`) found:
  1. Area camera indices in `scene_frame.json` count all cameras of `splat/cameras.json`.
     Fill counted training cameras only; tour counted all, and demo training only, so
     standing points came from the wrong cameras, increasingly so later in a capture. One
     helper (`scene_frame.run_frame`) now makes and reads the file, and stores
     `camera_count`, so a file counted otherwise is remade.
  2. Frame selection on videos longer than `max_frames × max_gap_s` (15 min) looped
     forever: the time gap now widens so that time cuts alone fit in `max_frames`.
  3. Mirrors: only points and depths *seen through* a mirror (the line of sight from a
     camera in front crosses the mirror rectangle) count as reflections. Before,
     everything behind the plane did, which in a house deletes the real room behind a
     bathroom mirror's wall. Seed points are dropped when most of their views see them
     through the plane.
  4. Reflector planes count distinct frames (a window split by mullions in one frame
     counted as 3); the reflector cache is reused only for the same solution size and
     glass setting; mirrors and windows are still searched when no floor is found.
  5. Training resumes mid-run: checkpoints every `checkpoint_every` steps also hold the
     optimizer states, view order and random state, keyed by the run's configuration and
     split. Before, an oomd kill at step 29,000 restarted from 0.
  6. Smaller fixes: gap seeds run on the pipeline's device; fill no longer divides
     gsplat's expected depth by alpha a second time (it inflated fill depth by up to 11%);
     support weights use `autograd.grad` instead of `backward()`, which left gradients on
     the model's parameters; images are loaded only for the cameras a run uses.
- **Rationale and trade-offs:** each was a silent wrong result or a lost run. Mid-run
  checkpoints are 3× larger (Adam moments); the final checkpoint keeps its size.
- **Affected components:** `scene_frame.py`, `fill.py`, `tour.py`, `demo.py`,
  `ingest.py`, `splat/reflect.py`, `splat/train.py`, `splat/init.py`, `splat_stage.py`,
  `reflectors.py`, `support.py`; `spacegen/docs/ARCHITECTURE.md`.
- **Completion criteria:** CPU tests for 1–4 pass; the GPU resume test
  (`test_training_killed_mid_run_resumes_from_its_checkpoint`) and the existing GPU tests
  pass once the GPU is back.

## 02-11-26 00:50:00 IST — spacegen: find the floor when the video looks straight down

- **Status:** adopted.
- **Decision:** `estimate_frame` searches for the floor plane around two guesses of up:
  the cameras' mean image-up (−y, as before) and the direction most perpendicular to all
  camera x axes (people rarely roll the phone). It keeps the plane with more inliers. A
  floor is accepted at 3% of the low points (was 5%), still with at least 200 inliers.
- **Rationale and trade-offs:** `room_m_block` films the floor and beds from above for
  its second half. A camera looking down has a horizontal image-up, so the mean image-up
  leaned 68° from vertical and the floor (outside the 30° search window) was not found:
  the tour orbit and walkthrough came out tilted, and the fill and reflector stages lost
  their floor. The x-axis guess was 22° off, and the plane found from it was within 5° of
  the up of the upright early frames. The beds hide most of the floor, so it holds only
  5% of the low points; vyas_room, new_home and Coorg hold 26–93%, and both guesses give
  the same plane there (within 0.2°). Keeping the mean image-up as a second guess covers
  captures that switch between portrait and landscape, where x axes are not horizontal.
- **Affected components:** `spacegen/src/spacegen/scene_frame.py`; runs re-do audit,
  fill, tour and demo (the splat is unchanged, but its reflector search also needs the floor).
- **Completion criteria:** a synthetic test with half the cameras looking straight down
  recovers up within 1°; `room_m_block_v4` finds its floor and its tour is level.

## 01-11-26 14:45:00 IST — spacegen: why `new_home_v2` scores below `new_home` (v1)

- **Status:** tested, 01-11-26 18:10 IST. 2 blocks of 524 (`new_home_v3`, finished from
  its final checkpoint after a systemd-oomd kill; the reflector f0 is from before the
  joint fine-tune) score 22.97 / 26.79 / 0.876 / 0.412, against 22.50 / 26.55 for 3 blocks
  and 23.30 / 26.93 for v1: the block count explains about 60% of the loss. Per frame
  (mean over the 124 held-out frames: v1 23.73, 3 blocks 22.86, 2 blocks 23.37) frames
  00280, 00096 and 00160 still lose about 6 dB, while the 10k-step single model renders
  00280 at 23.4 dB. So splitting into blocks costs quality on the frames near block
  borders. The joint fine-tune of the 2.59M-Gaussian merged model trained in 6.4 GB of
  GPU memory, so `new home` fits one model. Next: train captures up to a few thousand
  frames as a single model (`max_images_per_block` well above the frame count, a larger
  `max_gaussians`), keeping blocks for larger captures. The systemd-oomd kills (07:06
  and 17:52) led to training resuming from a final checkpoint, which now also stores
  the reflectors' f0.
- **Finding:** the full `new_home_v2` build (v4 settings, 30k steps) scores 22.50 / 26.55
  dB held-out (PSNR / colour-corrected) against 23.30 / 26.93 for v1, on the same
  cameras and 124 held-out frames. v1 predates two groups of settings: G1, the 3f7310e
  priors (depth prior 0.3 → 0.03 instead of 0.1, normal prior, saturation); and G2, the
  31-10-26 changes (MVS and gap seeds, needle penalty, pruning guard, reflections, hard
  views). 10k-step single-model runs on the v2 cameras (`outputs/spacegen/new_home_v2/exp`),
  held-out PSNR / colour-corrected / SSIM / LPIPS:
  - all v4 settings: 22.28 / 26.52 / 0.870 / 0.410;
  - G1 off: 22.53 / 26.81 / 0.872 / 0.406;
  - G2 off: 21.91 / 25.96 / 0.861 / 0.430;
  - both off (v1 settings): 21.95 / 26.17 / 0.864 / 0.428;
  - no metric near depth: 22.07 / 26.54 / 0.869 / 0.412;
  - with the re-sliced extra frames: 22.46 / 26.61 / 0.871 / 0.419.
  So G2 helps `new home` (+0.37 dB), G1 costs it about 0.25 dB (it was added for the
  blank walls of `vyas-room`), and metric near depth and the extra frames are neutral to
  slightly positive. None of the settings explains the 0.8 dB. Per frame, the 10k run
  with v4 settings beats the full v2 build on the frames that regressed (00280: 23.4 vs
  18.2 dB; 00160: 27.1 vs 24.2), so the loss comes from how the full build trained.
  The clearest difference: the extra frames raised the training set from 868 to 1048
  images, so the partition (`max_images_per_block = 450`) made 3 blocks of about 350
  instead of v1's 2 blocks of about 434.
- **Test:** rebuild the splat with `partition.max_images_per_block = 600` (2 blocks of
  524) into `outputs/spacegen/new_home_v3`, otherwise the v4 settings.
- **Completion criteria:** if 2 blocks recover v1's held-out score, raise the block
  size default (GPU memory permitting) or make the partition count blocks before adding
  the extra frames; otherwise look at the merge and the joint fine-tune.

## 31-10-26 20:34:02 IST — spacegen: near doors (gap seeds, guarded screen-size pruning) and planar reflections (floor, windows, mirrors)

- **Status:** Adopted with changes, 01-11-26 00:05 IST (branch `feat/spacegen`): gap
  seeds, guarded pruning, metric near depth (below), and reflections for mirrors and
  windows only (`floor_reflector = false`). 10k-step `vyas-room` results (held-out PSNR /
  colour-corrected PSNR / SSIM / LPIPS; training time). The times of `fix1`, `door2` and
  `mirrors` are inflated: a `new home` splat was training on the same GPU (it had not
  stopped). Timed alone, one render took 0.063 s plain, 0.15 s with the floor reflector
  and 0.21 s with all five planes.
  - needle baseline: 23.32 / 28.91 / 0.888 / 0.298; 934 s;
  - everything, floor included (`fix1`): 22.89 / 27.74 / 0.879 / 0.322; 3404 s. Training
    views also fell (29.85 → 28.82 colour-corrected), so the fit itself was worse. Shoes
    and chair legs on the floor smeared. Gaussians under the floor fell from 318k to 119k.
    The floor is in nearly every frame, so its reflection pass ran on almost every step;
  - door fixes, no reflections (`norefl`): 23.44 / 29.07 / 0.889 / 0.292;
  - plus metric near depth (`door2`): 23.64 / 29.04 / 0.891 / 0.293; 1756 s;
  - plus mirror and window reflections (`mirrors`): 23.31 / 29.13 / 0.889 / 0.292;
    2402 s. The mirror views gained 0.9–1.8 dB (00312, 00320), and the person filming
    now shows in the mirror. 00128 lost 2 dB.
  - The door leaf was still mostly missing: a dark haze in 00038, nothing in 00040–00041.
    Trained on frames 00038–00063 alone (3k steps), the door formed: it is seen in 3 of
    344 training frames, so it got 1% of the steps. Hence *hard views*
    (`hard_view_share = 0.3`): 30% of the steps train a view drawn ∝ its last L1².
  - plus hard views (`hard`, the adopted setting): 23.14 / 29.04 / 0.889 / 0.296; 1193 s
    alone on the GPU (needle: 934 s). The door is back, wood grain included, though
    still patchy at 10k steps: held-out 00040 +3.7 dB, training 00041 +8 dB.
- **Metric near depth (added 31-10-26 22:40 IST).** The gap seeds alone did not bring the
  door back: each image's affine depth fit only holds between its SfM points, all far
  away, and put the door at 5.3, 1.0, 1.0 and 1.7 units in 00038–00041, so the
  neighbour check rejected it and the depth prior pulled it away. Depth Anything V2
  Metric Indoor (Base) said 0.84, 0.67, 0.77 and 0.65 m. Each image's metric depth gets a
  scale from its SfM points (`depth.align_metric_depth`, ``depth/metric/``), and pixels
  nearer than the image's nearest SfM points take it (`depth.near_inverse_depth`) in the
  depth prior, the gap seeds and the reflector fit. The door then sat at 0.97, 0.97 and
  1.01 units.
- **Original proposal.** The user asked to remake all three captures with door and
  glass-reflection fixes, and to upload the new splats and videos to Drive.
- **Decision:**
  - **Doors.** On `vyas-room` the door leaf next to the lens (frames 00038–00041) is
    missing even in its own *training* frames. The fitted depth shows it, but nothing
    seeds it: no SfM points (no texture), no MVS depth (the sweep range starts at
    0.7 × the nearest SfM depth), and the monocular seeds keep to the SfM depth range.
    MCMC only adds Gaussians next to existing ones. Two changes:
    - *gap seeds* (`mvs.gap_seeds`, 100k points): fitted monocular depth where MVS
      measured nothing, away from depth edges, kept only where 2 of the 4 neighbouring
      frames' fitted depths agree (2 px, 3%);
    - screen-size pruning (`max_screen_fraction`) no longer removes a large Gaussian whose
      centre lies on the depth prior's surface (within 10%). A near door needs large
      splats; a floater in front of the camera is still removed.
  - **Planar reflections** (`splat/reflect.py`, `reflectors.py`), after MirrorGaussian
    (ECCV 2024) and TR-Gaussians (2025). The image is T + Σ_k S_k · F_k(θ) · R_k:
    - T is the ordinary render;
    - S_k is each Gaussian's learned strength, gated to Gaussians within 0.03 camera
      heights of plane k and inside its extent;
    - F_k is Schlick Fresnel with a learned f0 per plane;
    - R_k is the scene mirrored across plane k, rendered at half resolution.
    - Depth comes from T, so the priors keep the surface on its plane.
    - The planes are the floor (from `scene_frame`) plus windows, glass doors and mirrors
      from an ADE20K segmenter (Mask2Former). Each of these is a vertical plane fitted to
      the depth *around* the region, merged across ≥ 3 frames, with an extent.
    - Floors and mirrors are *opaque*: SfM, MVS and gap seeds behind them, and depth
      targets behind them, are dropped as reflections. Windows are not opaque, since what
      is seen through them is real.
- **Rationale and trade-offs:**
  - The needle model of `vyas-room` had 318k of 1.5M Gaussians (21%) under the tiled
    floor, a mirrored room imitating the reflections; there were more behind both mirrors.
    That geometry fights the depth prior, and it is why reflections looked painted on.
  - Costs:
    - one extra half-resolution render per visible reflector per step;
    - a 0.9 GB segmenter download;
    - standard 3DGS viewers ignore the `reflect_<k>` properties and show the scene
      without reflections. Our renders (tour and demo videos) include them.
  - Not handled: curved or non-planar glass, and moving people in mirrors.
- **Affected components:** `mvs.py`, `reflectors.py`, `splat/reflect.py`,
  `splat/model.py` (optional `reflect` parameter, `<ply stem>.reflectors.json`),
  `splat/train.py`, `splat_stage.py`, `partition.py`, `config.py`, tests.
- **Implementation steps:**
  1. Code, a GPU test (a mirror floor matches the explicitly mirrored scene) and a PLY
     round trip.
  2. A 10k-step `vyas-room` comparison against the needle run: held-out metrics, audit,
     renders of the door, the mirrors and the floor.
  3. Full runs of `vyas-room`, `new home` and `coorg_bnb`; tour and demo videos; upload to
     Drive under new names.
- **Completion criteria:**
  - held-out PSNR no worse;
  - the door present in its training frames;
  - Gaussians under the floor mostly gone;
  - reflections in the mirrors and on the floor moving with the viewpoint in off-path
    renders.

## 31-10-26 18:32:39 IST — spacegen: depth prior in the joint fine-tune of large captures; needle penalty

- **Status:** Needle penalty adopted 31-10-26 19:58 IST (branch `feat/spacegen`). On
  `vyas-room` at 10k steps (MVS on in both runs), it changed held-out PSNR from 23.38 to
  23.32, colour-corrected PSNR from 28.91 to 28.91, SSIM from 0.889 to 0.888 and LPIPS
  from 0.301 to 0.298. Weak audit segments fell from 14 to 12, and probe-view
  unrendered/unmeasured shares were within ±0.005. Splats with axis ratio above 4 fell
  from 31% to 3.2%, and the 99th-percentile ratio from 19.7 to 4.3. The block
  fine-tune fix waits on the `new home` re-run.
- **Decision:**
  - The joint fine-tune after merging spatial blocks (`partition.train_in_blocks`) keeps
    the depth prior at `depth_prior_weight_final`. It used to set the weight to 0, so on
    `new home` (and on any capture above `max_images_per_block`) the final 10k steps over
    all frames ran without depth.
  - Add a needle penalty: `needle_weight` · mean(max(0, log(largest axis / second
    largest) − log(`needle_ratio`))), with weight 0.01 and ratio 4. Splats stretched
    along one axis streak in off-path views. Flat discs on walls are unaffected, unlike
    PhysGaussian's max/min ratio, which would fight surfaces.
- **Rationale and trade-offs:** Depth is what holds surfaces in place between
  viewpoints. Both changes keep the model grounded in the video, with no generated
  content, and they cost nothing at render time. The penalty could blur thin real
  structures (cables, grille bars); check held-out views and off-path probe views.
- **Affected components:** `spacegen/src/spacegen/partition.py`, `splat/train.py`,
  `config.py`, tests.
- **Implementation steps:**
  1. Code and a unit test.
  2. A 10k-step `vyas-room` comparison: held-out views plus audit probes.
  3. Re-run `new home` with the full pipeline (MVS, re-slicing, blocks with the prior).
  4. Zip the results and upload them to Drive.
- **Completion criteria:** held-out PSNR no worse, fewer unrendered or unmeasured pixels
  in probe views, and `new home` off-path renders visibly steadier.

## 31-10-26 17:44:06 IST — spacegen: multi-view stereo, a self-audit from many viewpoints, and denser frames where the audit finds gaps

- **Status:** Adopted 31-10-26 19:58 IST (branch `feat/spacegen`). On the full `vyas-room` run (30k
  steps), held-out PSNR went from 21.91 to 22.30, colour-corrected PSNR from 27.82 to
  28.06, SSIM from 0.864 to 0.867 and LPIPS from 0.311 to 0.307. Weak audit segments fell
  from 17 to 9. Re-slicing added 50 frames, all of which registered (median 1641
  inliers). The near door edge and the bright window are still weak.
- **Decision:**
  - **MVS stage** (`spacegen.mvs`, after `depth`). This is our own GPU plane sweep in
    PyTorch, because the installed pycolmap has no CUDA and COLMAP's patch-match needs it.
    - For each image, take up to 6 source views with the most shared SfM points and a
      useful triangulation angle.
    - Sweep inverse-depth planes over the image's SfM depth range. Cost is window NCC,
      averaged over the best half of the sources so occluded views do not vote.
    - Refine the winner to sub-plane precision.
    - Keep only depths confirmed by at least 2 source views' depth maps: reprojection
      within 1 px and depth within 1%.
    - Output: per image, metric depth on the depth-map grid (0 = none) and `mvs/mvs.json`.
    - Uses: on textured pixels, the depth prior trains against MVS depth instead of the
      fitted monocular depth. Fused MVS points join the seeds of the initial model.
  - **Self-audit** (`spacegen.audit`). Render a model from:
    - the held-out views (PSNR, SSIM, error map);
    - probe views around sampled training cameras (yaw ±30°, pitch ±15°, steps of 0.25
      camera heights), counting unrendered and unsupported pixels.

    Each problem is pinned to the video time of the nearest training cameras. Output:
    `audit.json` (per-view scores, weak time segments with the reason) and a contact sheet
    of the worst views.
  - **Re-slice stage** (`spacegen.reslice`, video captures only, before `splat`).
    1. Train a short preview splat (5k steps) and audit it.
    2. For the weak segments, decode the video again and choose extra frames at a
       third of the motion step.
    3. Undistort them with the solved lens: an exact remap built from COLMAP's camera
       model.
    4. Register each by PnP. ALIKED + LightGlue matches to the two nearest registered
       frames, lifted to 3D with the MVS depth, or else with the preview's rendered depth.
    5. Accept a frame only with enough inliers and a pose between its neighbours.
    6. Compute depth and MVS for the accepted frames too. They join training only, never
       the held-out set.
  - **Final audit stage** after `splat`, for the report. The new stage order is ingest, sfm,
    depth, mvs, reslice, splat, audit, fill, tour, demo.
- **Rationale and trade-offs:**
  - Monocular depth is a learned guess fitted with two numbers per image. MVS measures
    depth from parallax, so it is right on textured surfaces (furniture, floor patterns,
    window frames), where the fitted monocular depth bends.
  - The audit answers "what is missing where" from the model itself, rather than only
    from the training views, which the model can memorise.
  - Re-slicing spends frames only where the reconstruction is weak, instead of halving the
    motion step for the whole video, which doubles SfM time (Coorg took 3 h).
  - Extra frames sit between training frames, so they are closer to some held-out frames
    than the regular training frames are. Held-out metrics with and without re-slicing are
    not strictly comparable. Report both, plus the probe checks, which have no held-out
    leak. Keep extra frames at least half a motion step away from any held-out frame.
- **Affected components:**
  - new `spacegen/src/spacegen/{mvs,audit,reslice}.py`;
  - `pipeline.py` (stages);
  - `splat/train.py` (`Priors` uses MVS);
  - `splat/init.py` (MVS seeds);
  - `splat_stage.py` (extra frames are train-only);
  - `config.py`;
  - tests, ARCHITECTURE, STATE.
- **Implementation steps:**
  1. MVS module, with a GPU test on a synthetic textured scene with known depth.
  2. Use MVS in training and seeding.
  3. Audit module and test.
  4. Re-slice module and test (PnP recovers a known pose).
  5. Pipeline wiring.
  6. Run on `vyas-room`: compare MVS depth with the fitted monocular depth, inspect audit
     sheets, and compare probe checks and held-out metrics against the current model.
- **Completion criteria:**
  - Tests pass.
  - On `vyas-room`, MVS covers most textured pixels with error below the monocular fit.
  - The audit flags the regions seen as poor in renders.
  - The re-sliced run is no worse on held-out views and better on probe checks.

## 31-10-26 13:04:01 IST — spacegen: metric depth and normal priors for blank walls and glass; tours that only show what was filmed

- **Status:** Adopted 31-10-26 15:20 (branch `feat/spacegen`). On `vyas-room` at 10k steps,
  against the old prior:
  - blank-wall normal error 37.5° → 17.9°;
  - held-out depth error 6.8% → 3.5%;
  - held-out PSNR 23.06 → 22.86 dB (colour-corrected 28.06 → 28.22), SSIM 0.876 → 0.877.

  Two corrections came out of validation:
  - The depth prior compares *alpha-weighted* inverse depth on every pixel. With
    normalized depth on opaque pixels only, floaters could turn transparent and leave dark
    smudges. These were seen in runs that also had the frozen learning rate below, so that
    fix is not measured on its own.
  - Holes are alpha < 0.5, not < 0.9, because dark objects legitimately sit at 0.6–0.9.

  A loop-variable clash (`decay`) froze the position learning rate in the first four
  runs. It is now logged (`means_lr`) and tested.
- **Decision:**
  - Fit each image's monocular inverse depth to the SfM points that image sees (inverse
    depth = a · mono + b, trimmed refit) in the depth stage, and write the fits next to
    the depth maps.
  - Train against that *metric* inverse depth (L1, relative to the image's median) for
    the whole run, decaying exponentially instead of fading to zero. Pearson stays as the
    fallback for images without a usable fit.
  - Add a normal prior: normals of the rendered depth should match normals of the fitted
    prior depth, weighted towards textureless pixels, where the photometric loss cannot
    place a surface.
  - Do not let saturated pixels (blown-out windows) pull renders darker than the clipped
    capture.
  - Tours and demos show only what was filmed from near each standing point: look-arounds
    sweep the yaw range of nearby capture cameras at their tilt, never a full 360° turn
    over generated content. A section whose checks fail is shortened or dropped instead
    of published.
- **Rationale and trade-offs:**
  - Rendered depth and normals of the `vyas-room` splat show blank walls and the floor as
    bumpy surfaces with blobs in front of them. They look right from the filmed views and
    smear as soon as the viewer moves.
  - A Pearson correlation per 4 × 4 patch cannot see this. It ignores scale and offset,
    so a blob uniformly in front of a whole patch scores as well as the wall itself. It
    also fades to zero at step 15k, while MCMC keeps moving Gaussians until step 25k.
  - The tour's first sections were 360° turns on the filled model. In `vyas-room`, Area 2
    had 37% unmeasured pixels, which were reported in `tour.json` and published anyway.
    These are the "first rooms" that looked poor.
  - Cost: a metric prior inherits the monocular network's errors. Mirrors and glass are
    painted onto their surface instead of showing parallax. The held-out score must not
    drop by more than 0.3 dB.
- **Affected components:** `spacegen.depth`, `spacegen.splat.train`, `spacegen.splat.init`,
  `spacegen.tour`, `spacegen.config`, `spacegen/docs/LEARNINGS.md`, `spacegen/docs/ARCHITECTURE.md`.
- **Implementation steps:**
  1. Run a 10k-step baseline and a 10k-step version with the priors on `vyas-room`.
  2. Compare held-out PSNR, SSIM and LPIPS; rendered depth and normals; and off-path
     views.
  3. Gate tour sections with their checks.
  4. Run the full pipeline again on `vyas-room` and Coorg.
- **Completion criteria:**
  - Walls and floors are flat in rendered normals.
  - The held-out score holds.
  - No tour section exceeds 5% unmeasured or 1% unrendered pixels.

## 31-10-26 00:25:16 IST — spacegen: estimate the focal length before mapping; check video solves for continuity

- **Status:** Adopted (branch `feat/spacegen`).
- **Decision:**
  - Before mapping, estimate each camera's focal from the verified fundamental matrices
    (Mendonça–Cipolla: E = KᵀFK should have two equal singular values). If COLMAP's guess
    (1.2 × the long side) is more than 15% off, replace it and verify the pairs again.
  - For video, measure the solve's continuity along capture order. Isolated cameras (far
    from all temporal neighbours) and jumps (a step more than 12× the median) count
    against a mapper attempt, and isolated cameras are left out of training.
- **Rationale and trade-offs:** The 991-frame portrait phone video `new home.mp4`
  registered every frame at 1.28 px with GLOMAP. Even so:
  - its first 28 cameras were scattered hundreds of units away;
  - the second half of the walk sat four units below the first, in a single-floor
    apartment;
  - the splat trained on it was fog (12.7 dB held-out).

  The database held the 2,304 px guess as a *prior* focal, but the lens is 1,050 px.
  Verification therefore labelled 29% of the sequential pairs "uncalibrated", and every
  relative pose was decomposed with a focal more than 2× too long. The incremental mapper
  (2.7 h) did not fix it either. The estimator gives 1,051 px here, 1,046 px on playroom
  (dataset: 1,040 px) and 780 px on a distorted action cam (bundle adjustment: about
  1,005 px; the guess was 2,304 px). It costs a second verification pass, minutes on
  CPU.
- **Affected components:** `spacegen/src/spacegen/sfm/{mapping,solve}.py`,
  `SfmConfig.estimate_focal`, `spacegen/docs/LEARNINGS.md`, a CPU test.
- **Completion criteria:** `new home.mp4` solves with no isolated cameras and no height
  steps, and its splat gives held-out PSNR comparable to the benchmark scenes.

## 30-10-26 16:31:31 IST — `spacegen/`: a general video-to-3D-space pipeline

- **Status:** Adopted (branch `feat/spacegen`).
- **Decision:** Build a standalone project in `spacegen/`, with its own `pyproject.toml`,
  `src/spacegen/`, tests, configs and docs. One command, `spacegen run <video or image
  folder> --out <dir>`, runs every stage and resumes. It ends in a splat, a report and a
  tour mp4. Proven parts of `src/` are copied and cleaned up, not imported, so the folder
  can become its own project.

  These Coorg lessons are built in:
  - ALIKED + LightGlue matching, with a planar gate on retrieval pairs;
  - loops checked by rotation against a sequential solve;
  - the OPENCV lens model;
  - MCMC noise normalized by scene radius, plus screen-size pruning;
  - per-image colour and pose correction;
  - a depth prior;
  - visibility-weighted block merge with a joint fine-tune;
  - supersampled renders;
  - a look-around that turns only through supported directions.

  New in spacegen:
  - automatic blocks by camera position, sized to GPU memory, instead of typed-in times;
  - automatic room finding and floor alignment;
  - per-Gaussian *view-direction support*, which records from which directions the
    capture saw each Gaussian;
  - a **fill-and-mark** stage. Holes and unsupported pixels in probe views at room
    centres are inpainted by a diffusion model and distilled into the splat. Filled
    Gaussians are flagged in the PLY, and the report gives the share of filled pixels.
- **Rationale and trade-offs:**
  - **Typed-in times:** Coorg needed time ranges, room names and a hand-run chain of
    about ten steps. A general pipeline must decide these from the data.
  - **Why fill and mark:** smears and black gaps come from directions the camera never
    saw (see the 30-10-26 10:30 entry). Training cannot recover them. The user chose to
    fill them with generated content and mark it, rather than show only filmed views.
    Filled content is plausible, not measured. It must never be reported as measured.
  - **Scoring:** benchmarks give honest numbers. Mip-NeRF 360 indoor scenes and Deep
    Blending use the standard every-8th-image held-out split and are compared with the
    published 3DGS scores. YouTube walkthroughs test real phone video and are kept local.
  - **Cost:** a standalone environment costs some disk for its own venv; uv hard-links
    from its cache.
- **Affected components:** new `spacegen/` only. Nothing in `src/` changes. This entry,
  `docs/STATE.md` and `docs/ARCHITECTURE.md` point to `spacegen/docs/`.
- **Implementation steps:**
  1. Scaffold, contracts and ingest: video or images to frames, with sharpness and
     motion-adaptive selection.
  2. SfM.
  3. Monocular depth.
  4. Single-block trainer with LPIPS/PSNR/SSIM evaluation, benchmarked on Deep Blending
     and Mip-NeRF 360 indoor.
  5. Partition and merge, tested on Coorg.
  6. Floor, rooms and tour.
  7. Support and fill-and-mark.
  8. YouTube walkthroughs.
  9. Fix what the checks find and repeat.
- **Completion criteria:**
  - Held-out PSNR within about 0.5 dB of published 3DGS, or better, on the benchmark
    scenes.
  - Coorg at least as good as `opencv/finetune_2` without hand settings.
  - Every tour frame has no unrendered (alpha < 0.5) pixels.
  - Filled share is reported.
  - The mp4s are inspected visually.

## 30-10-26 10:30:00 IST — Free-viewpoint quality: solid surfaces and an in-room look-around

- **Status:** Partly adopted (30-10-26 11:10 IST). The look-around, supersampled tour
  renders and `prune_invisible.py` are adopted. The training terms remain options; the
  final model is unchanged.
- **Outcome:**
  - *Training terms:* the fine-tune (`configs/coorg_solid/solid.toml`) held the held-out
    score: 25.46 / 27.22 dB against 25.39 / 27.37. It was better on only 7 of 20 views,
    and in-room views were nearly unchanged.
  - *Regrowth:* MCMC regrowth from the visible Gaussians lost 2–3 dB with or without
    the new terms. Relocating all dead Gaussians at once gave NaN.
  - *Why:* this capture's unfilmed directions cannot be recovered by training.
  - *The fix:* the look-around now turns only through directions filmed from where it
    stands, and follows the phone's tilt, as the problem statement below required.
- **Problem:** Eye-level renders along the capture path match the video, but views from
  the middle of a room, turned away from where the phone pointed, show smeared walls and
  furniture and dark holes near the ceiling. Two findings:
  - 88% of the 4.04M Gaussians never contribute even 1 px in any training view, but
    removing them leaves these views unchanged.
  - Rendering without view-dependent colour also leaves them unchanged.

  So the smears are visible Gaussians that sit in the wrong place: semi-transparent,
  thick layers that only reproduce the filmed angles.
- **Decision:** Add three optional training terms and compare fine-tunes of
  `opencv/finetune_2` on held-out views and on fixed in-room look-around views:
  1. A random background colour per step (`training.random_background`), as in
     Instant-NGP and Nerfstudio. Any see-through region then shows a different colour each
     step, so training must make surfaces opaque instead of fog over black.
  2. A flatness term (`training.flatten_regularization`): the mean ratio of each
     Gaussian's smallest to largest scale. Thin, disc-like Gaussians lie on surfaces, as
     in 2DGS, Gaussian Surfels and PGSR.
  3. A stronger monocular depth prior (0.05 → 0.2), since free roaming now matters more
     than the last few tenths of a dB near the path.

  The tour gains an in-room look-around before the walkthrough. For each room it stands
  at the eye-level point whose surroundings were filmed from the widest range of
  directions, then turns through the headings that were actually filmed from nearby.
- **Completion criteria:** held-out PSNR within about 0.5 dB of the current final model;
  visibly cleaner look-around views; the new tour checked frame by frame.

## 29-10-26 17:20:00 IST — Flat-floor check, corrected cameras for the entrance, and a robust tour video

- **Status:** Adopted. The tour's floor handling is in place. The cameras are now solved
  with the OPENCV lens model (amended 29-10-26 21:50 IST; see below).
- **Amendment (21:50 IST):**
  - *Approaches that failed:*
    - The incremental mapper split into three models.
    - `floor_bundle_adjust.py` flattened the floor only at 2–3 px reprojection error, and
      plain bundle adjustment returned to the step.
    - A piecewise shrink of the entrance, followed by bundle adjustment, also returned to
      the step.
    - A separate focal length for 0–96 s found the same focal and left the step.
  - *The approach that worked:* the same GLOMAP pass with `--camera-model OPENCV`. The
    floor spread falls from 0.144 to 0.059 camera heights. The camera height above the
    floor at 24–60 s and 96–110 s falls to 1.05× and 1.01× the hall value. Reprojection
    improves from 1.31 to 1.27 px. The fitted lens has k1 0.082 and k2 −0.216, so the
    single radial term was the cause.
  - *Adopted pipeline:*
    1. `sfm_opencv`.
    2. `densify_sfm.py --holdout-every 8 --holdout-exclusion-s 0.13`, giving
       `sfm_opencv_dense8`.
    3. `transfer_splat.py`: point-based local similarities move the final splat.
    4. A 12k-step fine-tune with the depth prior
       (`configs/coorg_opencv/finetune.toml`).
  - `floor_bundle_adjust.py` stays as a diagnostic: it measures how much reprojection
    error flattening a floor costs.
- **Decision:**
  - The tour's cut-away measures the floor locally on a plan grid instead of using one
    floor height.
  - Re-solve the cameras with COLMAP's incremental mapper, on the same verified matches as
    `sfm_loops2`.
  - If its floor is flat, move the trained splat onto the new cameras
    (`src/splatting/transfer.py`) and fine-tune, instead of retraining from scratch.
- **Rationale and trade-offs:**
  - *Symptom:* the user saw a black hole in the tour's plan at the entrance (1:09), where
    the eye-level view shows the mat and crocs.
  - *Measured cause:* the rendered floor of the bay-window foyer and the mat corridor
    (24–72 s) lies about 0.4 units (≈0.3 camera heights) below the rest of the house. The
    global floor cut removed it.
  - *Why it is a solve error:* the video shows one flat floor. The main door's bottom
    sits at hall level, and the same mat runs on continuously. Two door heights agree to
    5% (2.13 vs 2.02 units).
  - *Size of the error:* camera height above the local sparse floor is 1.2–1.36× the
    hall value in GLOMAP pass 3, against 2.4× in pass 2 and 3× sequential-only. VGGT shows
    1.09–1.22×.
  - *Why the incremental mapper:* it triangulates and bundle-adjusts as it registers each
    frame, so scale passes through 3D points rather than only through translation
    directions. That is the usual cure for sag along a weakly looped stretch.
  - *Why transfer instead of retraining:* retraining the three blocks would take about 6
    hours. Transfer uses the shared keypoints of both solves: blended per-camera
    similarities plus point residuals, then a fine-tune.
- **Affected components:** `src/splatting/tour.py` (`local_floor`, reflection and floater
  filters), new `src/splatting/transfer.py` and `transfer_splat.py`, and the
  `outputs/coorg_bnb/sfm_inc` run.
- **Completion criteria:**
  - The sparse floor-height ratio is ≈1 in every time window.
  - Held-out PSNR is not worse than `merged_dense8_depth`.
  - The tour plan shows the entrance floor.

## 27-10-26 15:28:36 IST — Dense full-resolution SfM cameras and quality-focused splat training

- **Status:** Adopted (on `feat/dense-sfm-splat`, which merges `feat/LoopClosure` and the
  committed `origin/feat/splatting`).
- **Decision:** Add `src/sfm/`. It selects ~4 sharp frames per second from the source
  video (always including the 161 geometry frames) and exports them at full 1080×1920
  resolution. It matches them with ALIKED/LightGlue on sequential pairs and on verified
  DINOv2 retrieval pairs, then solves one shared camera with pycolmap (global mapper,
  incremental mapper as an alternative) and undistorts to a pinhole COLMAP model.
  Add a COLMAP-input trainer to `src/splatting` (new modules only). It uses gsplat MCMC
  densification, per-image camera refinement, per-image colour (exposure/white-balance)
  compensation and progressive resolution. It writes the existing PLY/checkpoint
  layout, so `view_gaussian_splat.py` and `render_gaussian_splat.py` keep working.
- **Rationale and trade-offs:** The existing Coorg splat (161 VGGT views at 294×518) scores
  30.6 dB on training views but 17.0 dB / SSIM 0.59 on held-out views, and its held-out
  renders are smeared and offset. That gap points to too few views (0.75 per second) and
  VGGT cameras that are only accurate to about one 518-grid pixel (≈4 px at full
  resolution), not to the trainer. Dense video frames give the overlap that SfM and 3DGS
  need. Learned features on dense sequential pairs avoid the failure of the earlier
  SIFT/exhaustive COLMAP test on 36 sparse room views. VGGT cameras stay the independent
  check: SfM cameras on the shared geometry frames are compared with them after a Sim(3).
  Phone video has auto-exposure, so per-image colour compensation stops exposure changes
  from turning into floaters. Camera refinement absorbs remaining sub-pixel pose error.
  The new trainer does not edit utkarsh's in-progress `src/splatting` files; it adds
  modules beside them.
- **Affected components:** new `src/sfm/`; new `src/splatting/sfm_*` modules; root scripts
  `build_sfm.py` and `train_sfm_splat.py`; `sfm` extra (`pycolmap`); tests; SFM/SPLATTING/
  ARCHITECTURE/STATE docs.
- **Implementation steps:**
  1. Select and export frames, extract features, match pairs, map, undistort.
  2. Write a report and inspection plots, and compare with VGGT cameras on shared frames.
  3. Train with held-out views that are exactly utkarsh's 20 validation frames, for a
     like-for-like comparison.
  4. Inspect renders visually.
  5. Iterate on the dominant error, e.g. a depth prior for plain walls, or more frames.
- **Amendments (28-10-26):** Retrieval loops are checked against a sequential-only
  trajectory (rotation), then against the loop-closed model (epipolar fit). The whole
  house is trained as three time blocks, merged by visibility-weighted ownership and
  jointly fine-tuned, because a single model was capacity-limited and became unstable
  with larger budgets. MCMC noise is scale-normalized; screen-size pruning was added.
- **Completed `28-10-26 21:13:39 IST`:** 927/927 frames registered. The final splat scores
  23.9 dB raw / 27.0 dB colour-corrected, SSIM 0.91, on the shared held-out frames
  (VGGT splat: 17.07 dB / SSIM 0.59 at 518). Details are in STATE.md, SFM.md and
  SPLATTING.md.
- **Completion criteria:** Most frames are registered in one model with sub-pixel reprojection
  error. Held-out PSNR/SSIM on the shared validation frames is clearly above 17.0 dB /
  0.59. Held-out renders and novel views are inspected visually. Tests on real data pass.

## 26-10-26 19:11:38 IST — Loop closure for global camera drift (two compared methods)

- **Status:** Adopted for implementation on `feat/LoopClosure` (based on `fix/recon-section-alignment`).
- **Evidence motivating the change:** Camera-relative and top-down slices of the actual
  `section_alignment/point_cloud.ply` (Coorg) colored by source time show that the late
  dining sections (≈180–215 s) are rotated by roughly 30° relative to the living room and
  pass through the TV wall. Plain walls appear twice (≈170 s vs ≈200 s copies), and the
  wood pillar appears in three to four places. The four preprocessing revisits connect only
  entrance/luggage views, so nothing constrains the end of the capture. The video does
  look back from the dining area into the living room (≈207–211 s), and from the living
  room into the dining area (≈88–113 s).
- **Decision:** Add a separate `src/loop_closure/` stage that reads the cached section
  predictions **read-only** and writes only to a new output directory. It shares one
  loop-detection step and offers two methods for comparison:
  - **A — VGGT-Long-style loop chunks.** Retrieve far-apart frame pairs with DINOv2 and
    verify them with ALIKED/LightGlue. Jointly infer a new VGGT "loop chunk" built from
    the frames around both ends. Register the chunk to each owner section using exact
    shared pixels (the same contract as adjacent overlaps), then derive a section-to-section
    Sim(3) loop edge. Add it to the existing global section graph. Held-out whole frames
    and cross-side feature agreement inside the chunk gate every loop edge. This follows
    the algorithm of the vendored `deps/VGGT-Long` (chunk → loop chunk → Sim(3) graph),
    but stays inside our VGGT wrapper, preprocessing contract and validation gates. It
    avoids VGGT-Long's separate image-directory pipeline and its extra dependencies
    (faiss, numba, pytorch-lightning, SALAD weights).
  - **B — Bundle adjustment.** Start from a section-graph export (baseline or A).
    Refine one camera pose and one depth scale per frame, plus 3D track points, using
    ALIKED/LightGlue tracks over sequential neighbours and verified loop pairs.
    Reprojection residuals are robust (Huber), and a weak log-depth prior ties each track
    to its VGGT depth. This prior fixes scale and keeps dense depth exportable. Intrinsics
    stay at the owner-section VGGT values. Some loop pairs are held out of the solve
    entirely for evaluation.
- **Rationale and trade-offs:** One Sim(3) per section cannot remove drift *inside* a
  section, so A alone may leave bends; B can bend every frame but depends on sparse
  feature coverage (weak on blank walls). Comparing both on identical held-out evidence
  shows which actually closes the loop. Neither method fuses observations, edits source
  inputs, or overwrites `section_alignment/`.
- **Affected components:** new `src/loop_closure/` package (contracts, detection, loop
  chunks, graph export, bundle adjustment, evaluation, CLI), a keyword-only `export_dir` /
  `strict_post_graph` option on `run_section_alignment` (defaults unchanged), root
  script `close_loops.py`, `[project.scripts] namma-loop-closure`, `tests/loop_closure/`,
  `docs/LOOP_CLOSURE.md`, `docs/ARCHITECTURE.md`, `docs/STATE.md`, `README.md`.
- **Completion criteria:** Real Coorg run of both methods. The same held-out loop-pair
  feature correspondences are evaluated for baseline, A and B (normalized 3D disagreement),
  along with adjacent-pair consistency and top-down/camera-slice renders of the actual
  exported clouds. The adopted method is recorded with its failures; no gate is loosened
  to make a result pass.
- **Amendments during execution (`26-10-26 20:25:16 IST`):** (1) the retrieval threshold
  dropped to 0.40/top-8, because at 0.70 the repeated entrance grilles monopolised
  candidates. (2) A non-planar override admits repetition-flagged pairs whose epipolar
  inliers are not explained by one or two homographies; this was measured on real true
  and false pairs before adoption. (3) Bundle adjustment uses explicit parameter scales
  plus a 30 px initial gate; with Jacobian-based scaling the cameras did not move.
  (4) An optional square-walls arm (M) was added to test the ≈30° dining yaw. It had no
  effect because image evidence supports the 162 s turn.
- **Completed:** B (from baseline) is adopted as the camera solution. A and M are retained
  as compared, documented arms. Results and limitations are in
  [LOOP_CLOSURE.md](LOOP_CLOSURE.md) and [STATE.md](STATE.md).

## 25-10-26 18:53:36 IST — Failed COLMAP reconstruction experiment on the M Block room

- **Status:** Failed experiment; further work in this direction discontinued.
- **Plan tried:** Evaluate COLMAP-based reconstruction as an alternative to VGGT, using real CPU SIFT extraction, exhaustive feature matching and incremental sparse Structure-from-Motion through `pycolmap`. Implementation and tests were carried out on the separate `feat/colmap-reconstruction` branch; this entry records the outcome on the current `fix/recon-section-alignment` branch.
- **Evaluation:** Tested on `videos/room_m_block.mp4` (the M Block room), using its existing 36 geometry keyframes with shared per-clip `SIMPLE_RADIAL` intrinsics, a 1600-pixel maximum feature-extraction image size, an 8192-feature setting and four CPU threads. The completed run took 259.761 seconds and registered 27/36 distinct images across three overlapping, independently framed submodels. Their image/point counts were 2/479, 21/7,329 and 26/11,970; mean reprojection errors were approximately 0.345, 0.992 and 0.696 pixels respectively. Submodel coverage overlaps and must not be summed.
- **Qualitative findings:** Personally inspected multiple orthogonal and perspective views of every submodel, compared them with chronological source frames, and rendered the largest exported PLY directly in Open3D. The largest model recovered recognizable bedding colors, bed-frame edges and furniture fragments, but most geometry remained thin, disconnected patches. Blank walls and broad floor regions had large holes, some structures were stretched or poorly aligned, and camera positions clustered tightly relative to the reconstructed scene extent. The smaller 21-image model was less coherent; the two-image model was an almost planar patch rather than a room. Nine keyframes remained unregistered, and the low reprojection errors did not establish convincing room shape or metric accuracy.
- **Decision and rationale:** The constructed point cloud was not convincing enough to warrant further work in this direction. Discontinue this COLMAP-based sparse reconstruction approach for the project based on the incomplete, geometrically unreliable M Block room result. This experiment did not evaluate dense stereo or meshing.
- **Evidence retained:** Native sparse models, colored `model_0.ply` / `model_1.ply` / `model_2.ply`, `report.json` and inspection screenshots remain under `outputs/room_m_block/colmap_geometry/`. The separate COLMAP branch retains its implementation and evaluation documentation.

## 25-10-26 14:12:41 IST — Depth-scaled fitting and independent interior handoff validation

- **Status:** Adopted after real overlap investigations; supersedes the tail configuration and endpoint-based primary validation in the preceding execution entry.
- **Evidence:** Retrying only the last window, including an earlier 24-view bridge, did not repair distortion already present at the end of original section 4. Replacing the last two original sections with four 24-view windows gives strong adjacent support. A separate probe showed the generic cloud-extent RANSAC threshold was too tight on wall patches: one overlap's target extent was 0.166 while camera distance was 0.514. Whole-section endpoints also contain per-view pose/depth outliers and are not the views chosen at the owner handoff. The failed original connection remained bad even after excluding endpoints; it was not rescued by relabeling validation.
- **Decision:** Keep four initial 36-view windows (`0,25,50,75`) and reconstruct four 24-view tail windows (`100,113,126,137`) at 518 resolution. Use `--tail-section-size 24 --tail-start-section 4`; preserve all superseded artifacts. Scale section-fitting tolerance by target median camera-to-observation distance (`0.03 × distance`), consistent with depth-dependent errors, rather than visible patch extent.
- **Validation contract:** For sufficiently large overlaps, hold out three whole interior frames near overlap quartiles (positions 2/5/8 of 11) and reserve the first/last shared views as separate endpoint stress observations. Neither primary validation nor stress observations enter fitting. For five shared frames, use three training/two validation frames without a separate stress split. Preserve the existing primary absolute limits and minimum three-frame/two-cell training consensus. Save all stress points/errors, including after optimization; do not remove observations or disguise within-section outliers as registration success.
- **Probe outcome:** All seven candidate boundaries pass independently held-out interior checks. Normalized median/p90 pairs are approximately `0.023/0.067`, `0.045/0.098`, `0.033/0.079`, `0.013/0.029`, `0.050/0.093`, `0.011/0.020`, `0.008/0.014`; training consensus spans 4–8 frames. Full independent feature benchmarking, actual-PLY inspection and final state recording follow the joint graph solve.
- **Completed `25-10-26 14:23:56 IST`:** Exported 999,971 RGB observations from all 161 frames with a connected eight-node/13-edge graph. All seven post-graph primary checks pass; all seven ICP candidates were rejected. On 6,937 identical eligible image-feature observations, normalized median/p90 disagreement improves from `0.23158/0.81667` to `0.01435/0.06162` (93.8% median reduction). Actual-PLY inspection finds markedly improved recognizable structure, with residual wall distortion, endpoint stress and imperfect loop agreement explicitly retained. Full results, limitations and viewing commands are recorded in [STATE.md](STATE.md) and [RECONSTRUCTION.md](RECONSTRUCTION.md#high-resolution-section-alignment). Observation fusion remains deferred.

## 25-10-26 13:45:47 IST — Repair unsupported dining-section overlap

- **Status:** Adopted during real execution of the section-alignment experiment.
- **Evidence:** All six original 36-view sections ran successfully at 518 resolution, peaking at approximately 9.96 GiB allocated / 12.21 GiB reserved. The last overlap had consensus in only one training frame; all three held-out frame medians exceeded 0.10 of camera-to-scene distance. Its ICP candidate reduced the median only from 0.218 to 0.207 and left p90 near 2.96, so that connection is unsupported. Replacing only the last window with two 24-view windows did not establish sufficient shared-frame consensus.
- **Decision:** Preserve failed section artifacts and retain the first five 36-view windows. Add earlier context to the smaller tail windows: zero-based starts 114, 127 and 137, each with 24 frames at unchanged 518 resolution. The active candidate graph therefore has eight nodes and full 161-frame coverage. Expose this explicitly as `--tail-section-size 24 --tail-context-frames 11`, never as an automatic resolution or frame-count fallback.
- **Gate clarification:** Section alignment must have consensus across at least three training frames and two spatial cells, a pooled held-out normalized median at most 0.10, and at least two held-out frames with median at most 0.10. Report upper tails and failing individual views without deleting their observations. Boundary 2–3's large p90 is concentrated in one inconsistent held-out camera/depth view; two independent held-out frames agree closely. A p90 above 0.35 is a diagnostic warning rather than a replacement for frame-level support. ICP must pass these absolute gates in addition to relative improvement. Recheck the same gates after global optimization.
- **Additional verification:** Reverified image revisits must retain broad image and 3D support after depth/confidence filtering and within both training consensus and held-out subsets. Independent benchmarking uses the same source features eligible at both resolutions, one characteristic camera-to-point distance per method, and separately reported owner-boundary pairs. Original clouds and all source observations remain unchanged; no observation fusion is performed.

## 25-10-26 12:41:11 IST — Coorg high-resolution section alignment

- **Status:** Adopted at the user's request; work on the already-created `fix/recon-section-alignment` branch.
- **Motivation:** Inspection of the initial Coorg BnB 161-view, 252-square reconstruction and source frames found ghosted/curved surfaces, detached dining-area geometry, and inconsistent camera/depth estimates. Floor leveling did not repair registration. The immediate benchmark is a demonstrable improvement over that vanilla reconstruction.
- **Scope:** Process only Coorg BnB's existing 161 geometry frames, preserving the complete image field of view, source identities, timestamps and frozen preprocessing. All new generated artifacts, including section PLYs, intermediate transforms, diagnostics and final PLYs, belong under `outputs/coorg_bnb/section_alignment/`. Multi-view observation filtering/fusion, surface reconstruction and further floor correction are deferred until the user requests them.
- **Section inference:** Six chronological 36-frame sections start at zero-based positions 0, 25, 50, 75, 100 and 125, giving 11 shared frames at every adjacent boundary and complete coverage. Infer at padded 518×518 (portrait content approximately 294×518), bfloat16, sequentially on the 16 GB A4000. Record elapsed time and actual CUDA peak allocation/reservation. Persist resumable section predictions and provenance; never silently lower resolution or drop frames on OOM.
- **Registration contract:** A section-to-global similarity acts as `X_global = s * R @ X_section + t`, with proper rotation, positive scale, and arbitrary scene units. The first section fixes the global gauge. Transform camera centers with scale and camera orientations with rotation; retain valid inverse extrinsics and per-observation provenance. Shared source-frame/pixel identities supply direct 3D correspondences between independent section predictions.
- **Pairwise optimization:** Balance samples over shared frames and spatial cells, excluding padding/nonfinite/nonpositive depths and the lowest-confidence observations. Initialize positive-scale Sim(3) with RANSAC/weighted Umeyama, refine using a robust loss, and evaluate on whole held-out shared frames. Record support, normalized median/p90 errors, scale and camera agreement. Reject collinear/insufficient support rather than inventing a connecting edge.
- **Global optimization:** Construct a connected graph of adjacent section measurements and geometrically verified cross-section revisit measurements. Reverify existing conservative preprocessing revisit candidates using actual image features and reconstructed 3D support; never force a false loop. Jointly optimize robust relative Sim(3) constraints with section 0 fixed. A chain alone is reported honestly; additional independent overlap measurements may provide redundant constraints, but do not masquerade as nonlocal loop closure.
- **ICP experiment:** After scale-aware initialization, evaluate robust, multiscale point-to-plane ICP on overlapping geometry. ICP estimates a rigid correction at the established scale. Keep a candidate only if independent held-out shared-frame correspondences improve without unacceptable upper-tail/camera regression; nearest-neighbor fitness alone cannot establish success. Save rejected experiment metrics as well as accepted decisions.
- **Validation:** Use real pretrained inference and real feature correspondences. Compare the old 252 reconstruction and the new scene on identical source observations with source-to-model pixel transforms, confidence eligibility, a common similarity gauge and normalized errors. Inspect RGB, section-colored, top/side and boundary plots of the actual saved PLYs; check PLY/NPZ equality, all 161 source IDs, finite cameras, proper rotations, positive scales and graph connectivity. Report remaining internal section distortion rather than treating section registration as bundle adjustment.
- **Implementation structure:** Reusable section inference/export, numerical registration/graph optimization and evaluation/ICP modules under `src/reconstruction/`, public exports, and a thin runnable root script. Independent code-writing agents own disjoint modules; a separate reviewer examines correctness and tests. Dependencies remain in `pyproject.toml`, managed with `uv`.
- **Commit sequence:** (1) Section inference and robust Sim(3) registration plus this adopted plan; (2) graph/ICP integration and any corrections required by real Coorg execution; (3) final verification improvements and consistent results/state/architecture/usage documentation. Target three commits total; initial implementation must remain within four code commits/five including documentation. Generated captures/artifacts remain ignored, not committed.
- **Completion criteria:** A real 518-resolution, full-161-frame Coorg export; checked and saved global section transforms/graph; an evidence-based ICP decision; quantitative and visual improvement over the initial cloud, or an explicit unresolved blocker; final reproduction/viewing commands and final `STATE.md` update.

## 21-10-26 15:58:08 IST — Floor-constrained point-cloud postprocessing

- **Status:** Adopted following user review of uneven Coorg floors.
- **Decision:** Preserve original VGGT artifacts and export separate `floor_aligned/` clouds. Use pretrained ADE20K SegFormer floor/rug masks, confidence-filtered 3D observations and robust plane fitting. Give Room one global rigid leveling transform. For the single-storey Coorg capture, also align reliable per-view floor planes by minimal rotations about each camera and vertical translations; interpolate plane estimates in source time where the floor is unseen. Export all transforms, adjusted camera poses, mask provenance and before/after residuals.
- **Rationale and trade-offs:** Diagnostics show Coorg's local floor normals vary by up to 18 degrees; global leveling alone cannot address this. Transform whole observations rigidly rather than projecting floor pixels onto a fabricated flat surface. Preserve per-view shape, scale and provenance, but explicitly treat this as a single-floor prior, not bundle adjustment or a correction of all cross-view drift. Interpolation across floor-free spans is an estimate, and genuine steps/multiple storeys require a different model. Both scenes can share Z-up/floor Z=0 without claiming a common XY origin or metric scale.
- **Affected components:** `src/reconstruction/floor.py`, public exports, root `align_floor.py`, optional floor dependencies and reconstruction/architecture/state documentation; generated diagnostics and corrected artifacts under `outputs/`.
- **Implementation steps:** Inspect semantic masks and per-view plane evidence; implement robust floor estimation and transform serialization; generate Room rigid-level and Coorg floor-constrained clouds; verify residual reduction, proper rotations, rigid distance preservation and exact cloud/camera provenance. Retain the unexecuted interactive viewer and add no tests per the user's earlier instruction.
- **Completion criteria:** Separate usable colored clouds, a documented coordinate contract, explicit inferred/reliable floor-frame status, measurable floor-residual improvement for Coorg, and original reconstructions retained.

## 21-10-26 12:38:58 IST — Execute capture reconstruction and provide a standalone viewer

- **Status:** Adopted following user request.
- **Decision:** Run each full capture's preprocessing separately, pull `feat/reconstruction`, and export joint VGGT reconstructions under `outputs/<capture>/reconstruction/`. Source Linux PyTorch and torchvision from the official CUDA 12.6 index, retaining the declared package versions. Add root `view_point_cloud.py` using the existing Open3D visualization module; do not execute the viewer or add tests.
- **Rationale and trade-offs:** This GPU machine has an RTX A4000 (16 GB) and driver 535.309.01. The default CUDA 13 PyTorch wheel cannot initialize CUDA; CUDA 12.x minor-version compatibility permits using a CUDA 12.6 build without changing the system driver. Full geometry pools remain the initial inference target; any memory-limited run must explicitly record reduced resolution or frame selection rather than silently merging independent chunks.
- **Affected components:** `pyproject.toml`, `uv.lock`, `view_point_cloud.py`, reconstruction usage/state documentation and generated artifacts under `outputs/`.
- **Implementation steps:** Install reconstruction/viewer extras with compatible CUDA wheels; verify real CUDA initialization; run both capture reconstructions headlessly; inspect saved array/cloud metadata; document commands and results; add a thin standalone viewer CLI accepting a point-cloud path.
- **Completion criteria:** Both captures have real exported point clouds and recorded inference settings, or an explicit hardware/runtime blocker. The standalone viewer is documented and linted but is not executed; interactive quality review is left to the user.

## 21-10-26 11:48:44 IST — CUDA VGGT inference and Open3D reconstruction preview

- **Status:** Adopted following user clarification.
- **Decision:** Add a reusable vanilla VGGT wrapper under `src/reconstruction/`, an adapter consuming the frozen preprocessing `load_result()` contract, and root script `get_recon.py`. Run all selected views jointly on a larger CUDA GPU. Use Open3D for coloured point-cloud export and interactive viewing. Do not run model inference on the development laptop.
- **Rationale and trade-offs:** Joint inference gives one scene coordinate system; independently inferred chunks cannot be concatenated without alignment. Default to full-field-of-view, patch-aligned resize and padding, with explicit pixel transforms and valid masks. Preserve frame identity, timestamps and source rotation. Depth and geometry remain in arbitrary VGGT scene units, not metres. The preview concatenates filtered observations; it does not claim surface fusion, a mesh, or a navigation-ready digital twin.
- **Affected components:** New reconstruction modules and tests, stage public exports, optional `reconstruction`/`viewer` dependencies, `get_recon.py`, architecture/state and reconstruction usage documentation. The preprocessing implementation and its artifacts are frozen.
- **Implementation steps:** Define CPU NumPy output contracts; implement lazy model loading, CUDA precision selection, camera decoding and depth unprojection; adapt ordered geometry/appearance records; save arrays/provenance plus PLY; provide Open3D viewing of new or existing PLY files.
- **Dependency boundary:** Load the checked-in VGGT submodule lazily and declare its inference runtime dependencies explicitly. Avoid installing its upstream demo stack or its conflicting `numpy<2`/GUI OpenCV requirements into the frozen preprocessing environment. Keep Open3D separately installable for CPU-only viewing of an exported cloud.
- **Completion criteria:** Focused image-transform, geometry, filtering, preprocessing-handoff and serialization tests pass without a fake model; lint passes; CLI help and import checks run locally. Pretrained CUDA inference and interactive reconstructed-scene inspection are explicitly recorded as not run on this laptop.

## 27-10-26 02:30:17 IST — Geometry-initialized Gaussian splatting

- **Status:** Adopted; code authoring only, runtime verification deferred to the development server.
- **Decision:** Implement `src/splatting` on `feat/splatting`, consuming a versioned JSON manifest of RGB images, decoded VGGT camera-head extrinsics/intrinsics, refined camera-Z depth and refined world-space point maps. Use PyTorch and gsplat 1.5.3 for geometry-initialized, RGB/depth-supervised 3D Gaussian optimization. Keep cameras fixed. Provide `src/rendering`, offline rendering, and a Python-launched Viser viewer using the same full spherical-harmonic renderer.
- **Rationale and trade-offs:** The upstream files are on an inaccessible development server, so define the handoff explicitly rather than guessing its filenames. Preserve source identities, image transforms, world coordinates and arbitrary reconstruction scale. Initialization uses training views only, confidence/validity filtering, reprojection and multiview depth checks, bounded sampling, voxel fusion and nearest-neighbor scales. A temporal holdout measures novel-view appearance without feeding its geometry into initialization. CUDA execution belongs exclusively to the development server; no third-party compute is involved.
- **Affected components:** `src/splatting/`, `src/rendering/`, root training/rendering/viewing scripts, `configs/gaussian_splatting.toml`, optional dependencies, tests, and splatting/architecture/state documentation.
- **Implementation steps:** Define strict contracts/configuration; implement memory-bounded geometry loading and initialization; add RGB/SSIM and decaying relative depth losses, capped adaptive refinement, resumable checkpoints and standard SH PLY; share differentiable rendering with the viewer; write real-implementation tests and server commands.
- **Completion criteria:** Code and tests are written, statically reviewed, and documented. Dependency resolution, lint, tests, CUDA compilation, actual VGGT integration, reconstruction quality and visual inspection remain explicitly unverified until run on the development server. Do not execute project code, install dependencies, train or render on the authoring machine.

## 21-10-26 02:39:40 IST — Reject descriptor-ambiguous repeated structures

- **Status:** Adopted following round-three user review.
- **Decision:** Before geometric verification, compare each cross-image ALIKED match against distant same-image descriptor alternatives in both images. Only matches whose cross-image descriptor distance beats both self alternatives by a configured ratio may contribute inliers or spatial spread. Record removed-match counts on accepted edges and rejected-repetitive pair diagnostics in the report. Do not use a homography-fraction veto or time-specific exceptions.
- **Rationale and trade-offs:** Repeated window motifs can produce shifted correspondences that satisfy fundamental-matrix and coverage checks. Prefer missed loop links over ambiguous ones; do not relax the guard to retain doors or luggage. Keep the geometric matcher shared between the standalone pair API and pipeline.
- **Affected components:** Revisit verification/config/edge diagnostics, real fixtures/tests and pass-specific cache provenance. Preserve ingest, scoring and selection behavior, including user commit `4a6a0a3`.
- **Implementation steps:** Add self-alternative filtering and real door/window controls; add a real spliced-input tracking-break control; rerun link/export only with validated upstream cache reuse; record all prior accepted pairs' before/after counts and images.
- **Cache detail:** Invalidation now uses pass-specific module/record fingerprints plus config, schema, library versions, inputs and artifact checksums. The round-two global source hash was verified against the unchanged source before editing; per-pass provenance was recorded then for a validated legacy-key migration. Old runs without that provenance conservatively recompute. Optional edge diagnostics remain backward compatible with cached sequential edges.
- **Completion criteria:** Door positive accepts >= configured inliers, at most one of the two reviewed different-window pairs accepts, cross-building and temporal controls still pass, splice has one break without a cross-boundary edge. Record any lost true links rather than tuning them back in.

## 21-10-26 00:51:44 IST — Texture-aware tracking and blur evidence

- **Status:** Adopted following user review.
- **Decision:** Apply a whole-frame relative and absolute minimum-eigenvalue floor before distributing corners into cells. Only supported cells enter the overlap denominator. Tracking reliability depends on track count and spatial coverage; survival remains diagnostic. Separately, require sufficient edge content for a per-frame blur decision, use edge re-blur evidence rather than a cross-scene sharpness reference as the deciding signal, and report low texture as informational rather than a selection failure.
- **Rationale and trade-offs:** Cell-local corner thresholds admit compression noise in blank walls, causing false breaks. Sharpness relative to a textured neighboring scene confounds texture with blur. Real-frame regression fixtures will exercise both failures; ambiguous blur labels will be reported rather than fitted with time-specific rules.
- **Affected components:** Tracking, scoring/config/contracts, quality reporting, fixtures/tests and documentation. Existing PTS, orientation and color conversion remain unchanged.
- **Implementation steps:** Fix tracking first; then evaluate texture/blur on the eight user-labelled Coorg observations; finally run real optional revisits, export every accepted pair, and rerun both captures.
- **Implementation detail:** The corner floor is `max(1e-5, 0.01 * frame_max_eigenvalue)`. Continuity needs eight forward/backward-consistent tracks over at least three of sixteen cells, initially and currently; survival is never a gate. Low reliability is recorded on triggered selections rather than independently forcing minimum-interval keyframes. Schema 2 adds edge fraction and re-blur loss to CSV scores. At 640 px, the 49.5 s motion-blurred door cannot be reliably classified from re-blur evidence alone; retain an explicit expected-failure test and use corroborating temporal evidence, rather than fitting a narrow threshold to eight observations.
- **Completion criteria:** Compare components, tracking-break/weak-link counts, interval keyframe counts and reason histograms against the baseline below. Run real revisit, cross-building negative, spread and temporal-exclusion tests; report all accepted pairs and image paths.
- **Baseline:** Room: 1 component, 0 breaks/weak links, 49 geometry, 15 in 8.8–14 s; reasons `clip_start=1, overlap=43, flagged_fallback=5, max_interval=5`. Coorg: 20 components, 19 breaks/0 weak links, 193 geometry, 20 in 72–79 s; reasons `clip_start=1, max_interval=46, tracking_weak=30, flagged_fallback=32, overlap=127`. Reasons may co-occur.

## 20-10-26 23:15:43 IST — Model-independent preprocessing pipeline for continuous smartphone video

- **Status:** Adopted.
- **Decision:** Implement preprocessing as the package `src/preprocessing`. It turns one or more smartphone clips into a model-independent package: full-resolution selected frames, a manifest with provenance and a candidate pool, a covisibility graph and a gap report. The default capture is one long continuous video with deliberate revisits; short patch clips are allowed. Model-specific resizing, reference-frame choice and memory-bounded grouping belong to the reconstruction stage, not preprocessing. Thresholds live in `configs/preprocessing.toml`. The per-frame score table is CSV, not Parquet.
- **Rationale and trade-offs:**
  - The reconstruction model is not decided (VGGT, FastVGGT, VGGT-Ω, VGGT-Long or others), so a model-specific handoff would couple the stages.
  - Continuous capture links rooms through sequential tracking. It cannot redo a single section, so patch clips and revisit detection recover weak connections.
  - Scoring every frame at low resolution is cheap on CPU and fits the 4 GB development laptop. Exporting only the selected frames avoids writing tens of thousands of full-resolution images.
  - Five passes (probe plus four processing passes), each writing a resumable artifact, cost extra decoding but allow on-site re-runs after a threshold change without starting over.
  - Measurements are flags, not verdicts. Blur, exposure, timestamp gaps and weak links are reported for review and do not cause automatic recapture.
- **Affected components:** `src/preprocessing/`, `src/__init__.py`, `configs/preprocessing.toml`, `preprocess_capture.py` (repository root), `tests/preprocessing/`, `pyproject.toml`, `docs/ARCHITECTURE.md` (section 2).
- **Implementation steps:**
  1. **Contracts and config.** `contracts.py` defines frozen dataclasses with JSON round trips and a `schema_version`. `config.py` defines `PreprocessConfig`, loaded from `configs/preprocessing.toml`.
  2. **Probe (pass 0).** `probe.py` reads rotation, time base, frame-rate variation, colour transfer/HDR, size and duration with PyAV and writes `clips.json`.
  3. **Score (pass 1).** `decode.py` yields frames with presentation timestamps at a 640 px long side in greyscale, with rotation applied. `scoring.py` computes:
     - tiled Laplacian sharpness: p75 of the variances on a 4×4 grid;
     - relative sharpness against both a short-window and a long-window reference;
     - dark and clipped fractions, luminance mean and spread, and exposure jumps;
     - timestamp gaps (`Δt` relative to the median) and cut detection from histogram correlation;
     - consecutive-frame KLT motion and survival from `tracking.py`. Unreliable tracking is recorded as unknown motion, not low motion.

     It flags low-quality intervals: severe blur can be flagged without high measured motion, and high motion is supporting evidence. Writes `all_frames.csv`.
  4. **Select (pass 2).** `selection.py` tracks KLT features from the last geometry keyframe:
     - Overlap = surviving cells ÷ cells occupied at the keyframe; initial coverage is recorded separately.
     - A keyframe is triggered by overlap below τ, or by a maximum time interval. Robust track displacement is recorded as a viewpoint-change proxy; homography residual is optional and labelled as such, not "parallax".
     - The keyframe is the sharpest unflagged frame in a short buffer. When an earlier buffered frame is chosen, tracking restarts there and replays the buffer.
     - A denser appearance set is selected with a higher τ, each frame linked to its geometry keyframe.
     - All candidates are kept as IDs in the candidate pool.

     Writes `selection.json`.
  5. **Link (pass 3).** `revisits.py` retrieves non-neighbour candidates with a global descriptor (DINOv2-based), matches local features (ALIKED or DISK with LightGlue; not SuperPoint, whose weights are non-commercial) and verifies with RANSAC on the fundamental matrix. Accepted links need spatially distributed inliers, and the temporal exclusion window is configurable. `graph.py` builds sequential and revisit edges with measured weights; consecutive keyframes do not get an edge without measured tracking support. `report.py` lists low-quality intervals, tracking breaks, weak links, unlinked clips and the loop links found. Revisits are an optional dependency extra; when they are not installed, the report states "revisits: not run".
  6. **Export (pass 4).** One sequential full-resolution decode. `storage.py` resolves output paths and source-frame retrieval. `export.py` writes `geometry_frames/` (PNG), `appearance_frames/` (JPEG q95, configurable), `manifest.json`, `covisibility.json` and `report.json`. `visualize.py` writes `debug/` plots: a sharpness/overlap timeline with selections and flags, keyframe contact sheets and revisit match images.
  7. **Public API.** `preprocess_capture(clips, output_dir, config=None, resume=True) -> PreprocessResult` and `load_result(output_dir)` are re-exported from `src/preprocessing/__init__.py` and `src/__init__.py`. The root script is `preprocess_capture.py`; the `[project.scripts]` entry point is `namma-preprocess`.
- **Completion criteria:**
  - Each pass writes its artifact and resumes when its config is unchanged.
  - The full pipeline runs on the clips in `videos/` and produces every output listed above.
  - Tests use real implementations on a short real clip cut from a capture.
  - The `debug/` outputs of a real capture have been inspected visually and the findings recorded in `docs/STATE.md`.

## 20-10-26 13:18:20 IST — Establish documentation scaffolding

- **Status:** Adopted.
- **Decision:** Keep the README to two lines, explain the architecture's components, and maintain separate state and plan documents. Defer detailed module contracts.
- **Rationale and trade-offs:** Provide a small, usable documentation structure for preprocessing work without prematurely specifying implementation interfaces.
- **Affected components:** `README.md`, `docs/ARCHITECTURE.md`, `docs/STATE.md`, `docs/PLANS.md`, and the README source notes in `docs/GOALS.md`.
- **Implementation steps:** Add the summary and documentation links; explain the existing diagram; initialise the state document with the requested sections and IST timestamps; keep affected references consistent.
- **Completion criteria:** The README has exactly two lines; documentation links resolve; the diagram's components are explained; state tracking includes Latest Decision and Current Progress instead of Latest Verification; timestamps follow the declared format.
