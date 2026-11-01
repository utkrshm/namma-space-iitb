# Experiment documentation provenance

The archive records the branches reviewed during consolidation. The submission's
active code is in the root `src/` package; this directory contains documentation only.

- [manifest.json](manifest.json) identifies the original 19 branch references and
  original STATE source revisions collected during consolidation.
- [branches/](branches/) retains their experiment narratives and decision records.
  Ref names use `__` in place of `/`.
- [state-history/](state-history/) preserves earlier experiment records that were
  superseded at branch tips.
- [../STATE.md](../STATE.md) is the current experiment ledger and collated history.

Documents are curated for submission scope; source hashes in the manifest identify
the original Git documents rather than checksums of the edited archive. Historical
commands describe their original branches and are not instructions for the current
submission. Use the root [README](../../README.md) for current setup and commands.

[experiments/README.md](../../experiments/README.md) indexes the main experiment guides.
