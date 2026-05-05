# Handoff — Turbo-only simplification

PR: https://github.com/happyFish/scope-stream_diffusion_v1/pull/2 (`sd-multi-model` → `main`).

## What landed

1. **Merge of `main` into `sd-multi-model`** (commit `d8d2fbd`) — brings the
   full TRT subsystem (refittable engines, node-id-keyed adapter cache, fp16
   VAE TRT path) under the multi-model dropdown.
2. **`feat(schema)` (`29f1ea7`)** — `model_id_or_path` enum trimmed to
   `stabilityai/sd-turbo` + `stabilityai/sdxl-turbo`. `num_inference_steps`
   and `use_suggested_num_inference_steps` fields removed.
3. **`refactor(pipeline)` (`1015f4b`)** — drops `self.sd_turbo`,
   `_attach_lcm_lora`, `_predict_x0_serial`, the `use_serial` branch in
   `__call__`, and the dead `denoising_steps_num > 1` branches. Step count
   is hardcoded to 1 in `__call__`.

Net: +9 / -188 LoC across `pipeline.py` and `schema.py`.

## Verified locally

- `python -m py_compile` clean on all .py files.
- Module imports cleanly from this worktree (`PYTHONPATH=src`) under Scope's
  venv (`~/Projects/daydreamlive-scope/.venv`).
- Schema reflects the trim: `Literal['stabilityai/sd-turbo',
  'stabilityai/sdxl-turbo']`, no `num_inference_steps` field.

## Not verified — please run before merging

Scope was not running on `localhost:8000` during this session, so the
hot-reload smoke test from the original handoff was skipped. To validate:

```bash
curl -X POST http://localhost:8000/api/v1/plugins/scope_streamdiffusion/reload \
  -H 'Content-Type: application/json' -d '{"force":true}'
```

Then in the UI:

- Render at `acceleration_mode=none` with `model_id_or_path=stabilityai/sd-turbo`.
- Swap to `stabilityai/sdxl-turbo` via the dropdown — confirm hot-swap path
  and SDXL fp16-fix VAE install.
- Render at `acceleration_mode=trt` for both variants.
- Run a moth dev session: scenes trigger, oscillators drive params,
  ControlNet (depth) and mask compositing still work.

## Out of scope (explicit)

- LoRA hot-swap (separate spec at `~/Projects/moth/docs/specs/lora-support.md`,
  Phase 4 of `streamdiffusion-trt.md`).
- Hyper-SD / Lightning step-LoRA fusion at load — the better path to
  1-step inference on arbitrary SD 1.5 / SDXL checkpoints; future PR.
- SD 3 / 3.5 — MMDiT, not UNet, incompatible with the current TRT path.
- Moth-side UI changes — none needed; the dropdown is schema-driven.

## Notes for next agent

- Editable install lives in the **main worktree path**
  (`~/Projects/moth-scope/plugins/scope-stream_dffusion_v1`), not this
  worktree. Once this PR merges into `main`, pulling main in the main
  worktree will pick the changes up automatically; until then, force-
  importing from `src/` is the only way to verify the worktree's code in
  Python.
- The 2-commit split (schema then pipeline) is intentional. The handoff
  asked for 2-3 commits; bundling pipeline.py into one was cleaner than
  trying to interleave the LCM LoRA / serial / dead-branch removals.
