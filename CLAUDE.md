@AGENTS.md

## Self-improvement (reflection)

After a mistake, a wasted run or a correction from the user, and at the end of a long
session, run `/reflection` (`.claude/commands/reflection.md`): review what went wrong,
generalize it, and add or sharpen a rule under "Lessons" below.

Rules for writing lessons:
- One line each, an absolute directive first, then the reason after a colon.
- General enough to apply again; no one-off facts (those go in `docs/STATE.md`).
- Merge with an existing lesson instead of adding a near-duplicate; delete lessons that
  turned out wrong. Keep the list under 25 lines.

## Lessons

- Before ablating settings to explain a regression, diff the two runs' `config.json`, frame
  counts and block counts: new_home lost 0.8 dB to 3 blocks instead of 2, not to a setting.
- Compare models only on the same cameras and held-out frames; otherwise say they are not comparable.
- Base time estimates on the run's real structure (blocks × steps + fine-tune + fill + videos), read from its log.
- Exit code 137 on this machine is usually systemd-oomd (memory pressure, not RAM): check
  `journalctl --user | grep oomd` first.
- Make every stage longer than a few minutes resumable, and save every trained tensor in
  checkpoints: a kill while saving must not cost the whole run.
- Kill a pipeline by its python PID (or process group) and confirm with `pgrep`: killing the
  wrapper shell left a stale training running for hours with old code.
- Render under `torch.no_grad()` whenever no gradient is needed: the reflection pass keeps
  its graph and ran Coorg out of GPU memory.
- Check `nvidia-smi` before ad-hoc GPU renders while a pipeline runs, and `source
  spacegen/scripts/env.sh` first: without it gsplat has no CUDA.
- Check `df -h` before launching runs that write checkpoints; the disk sits near 97%.
- Run `bash -n` on every unattended queue script and save `status=$?` right after the
  command: `echo "$(date) exit $?"` logs date's status, and a `>>>` typo killed a script's
  last lines unseen.
