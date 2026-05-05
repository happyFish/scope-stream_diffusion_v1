# Scheduler refactor — design notes

> Draft. Starting point for the actual implementation, not a finished spec.

## Why

`_set_timesteps` and `_unet_step` are coupled to LCMScheduler's internals.
Three calls leak through:

1. `scheduler.set_timesteps(num_steps, device, strength=...)` — the
   `strength` kwarg is LCM-only (img2img stop-step calculation).
2. `scheduler.get_scalings_for_boundary_condition_discrete(t)` →
   `(c_skip, c_out)`. The consistency-function coefficients. Used by
   `_unet_step` as `denoised = c_out * eps + c_skip * x_t`.
3. `scheduler.alphas_cumprod[idx]` direct read → `alpha_prod_t_sqrt`,
   `beta_prod_t_sqrt`. Used by the rolling-buffer renoise step (dead at
   N=1) and the RCFG `cfg_type="self"` branch.

This blocks adding Hyper-SDXL-1step (TCDScheduler + eta=1 + custom
timesteps) and SDXL-Lightning-1step (EulerDiscreteScheduler + custom
sigmas) as `MODEL_PRESETS` entries.

## Two inference paradigms

- **Consistency-model** (LCM, and TCD when used in consistency mode):
  UNet outputs noise; consistency function maps to x0 in one shot.
  Current code path.
- **Diffusion** (Euler, DDIM): UNet outputs noise; `scheduler.step()`
  integrates one timestep at a time. Lightning lives here. Hyper-SD-1step
  also lives here in practice — it relies on `eta=1.0` stochasticity that
  lives inside `scheduler.step()`, not in a c_skip/c_out formula.

The natural cut is **LCM stays on the consistency path; everything else
goes through `scheduler.step()`.**

## At N=1, the rolling buffer is dead

We hardcoded `num_inference_steps = 1`. The `denoising_steps_num > 1`
branches in `_predict_x0_batch` and `_prepare_runtime_state` were already
stripped in the Turbo-only PR. The remaining alpha/beta usage:

- The RCFG `cfg_type="self"` branch reads `alpha_prod_t_sqrt` and
  `beta_prod_t_sqrt`. At N=1, these are length-1 tensors and the `[1:]`
  slices degenerate (concatenated with `ones_like` + `randn_like`
  fallbacks). Not actually used to produce visible signal.
- `c_skip` / `c_out` ARE load-bearing — the consistency function
  `denoised = c_out * eps + c_skip * x_t` is the only thing turning UNet
  noise predictions into x0.

So the only path that *must* be made paradigm-aware is `_unet_step`.

## Proposed shape

### `MODEL_PRESETS` extension

```python
MODEL_PRESETS["hyper-sdxl-1step"] = {
    "base": "stabilityai/stable-diffusion-xl-base-1.0",
    "lora": ("ByteDance/Hyper-SD", "Hyper-SDXL-1step-lora.safetensors"),
    "scheduler_class": TCDScheduler,
    "scheduler_kwargs": {},                 # defaults are fine; eta passed at step time
    "scheduler_step_kwargs": {"eta": 1.0},  # threaded to scheduler.step()
    "timesteps_override": [800],
    "inference_kind": "diffusion",          # default "consistency"
}

MODEL_PRESETS["lightning-sdxl-1step"] = {
    "base": "stabilityai/stable-diffusion-xl-base-1.0",
    "unet_swap": ("ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet.safetensors"),
    "scheduler_class": EulerDiscreteScheduler,
    "scheduler_kwargs": {"timestep_spacing": "trailing"},
    "scheduler_step_kwargs": {},
    "timesteps_override": None,            # Lightning's recommended timestep is set via the scheduler config
    "inference_kind": "diffusion",
}
```

The existing 3 presets (`sd-turbo`, `sdxl-turbo`, `dmd2-sdxl-1step`) get
defaults: `scheduler_class=LCMScheduler`, `inference_kind="consistency"`,
no overrides. Their behavior stays identical.

### `_swap_model` change

```python
preset = MODEL_PRESETS.get(self.model_id, {})
scheduler_class = preset.get("scheduler_class", LCMScheduler)
scheduler_kwargs = preset.get("scheduler_kwargs", {})
self.scheduler = scheduler_class.from_config(self.pipe.scheduler.config, **scheduler_kwargs)
self._inference_kind = preset.get("inference_kind", "consistency")
self._scheduler_step_kwargs = preset.get("scheduler_step_kwargs", {})
self._timesteps_override = preset.get("timesteps_override")
```

### `_set_timesteps` change

Skip the LCM-specific computations when `_inference_kind == "diffusion"`.
Honor `_timesteps_override` when present.

```python
def _set_timesteps(self, num_inference_steps, strength):
    if self._timesteps_override is not None:
        # Skip scheduler.set_timesteps; pin the override directly
        timesteps = torch.tensor(self._timesteps_override, device=self.device)
    elif self._inference_kind == "consistency":
        self.scheduler.set_timesteps(num_inference_steps, self.device, strength=strength)
        timesteps = self.scheduler.timesteps.to(self.device)
    else:  # "diffusion"
        # Standard schedulers don't take `strength`; img2img re-noising
        # happens in _encode_image, not here.
        self.scheduler.set_timesteps(num_inference_steps, self.device)
        timesteps = self.scheduler.timesteps.to(self.device)

    self.timesteps = timesteps

    # ... sub_timesteps from t_list ... (unchanged)

    if self._inference_kind == "consistency":
        # Compute c_skip / c_out and alpha/beta as before
        ...
    else:
        # Diffusion path doesn't use these. Set to None as a tripwire so
        # any code that still reads them on a diffusion model errors loudly
        # rather than silently producing zeros.
        self.c_skip = None
        self.c_out = None
        self.alpha_prod_t_sqrt = None
        self.beta_prod_t_sqrt = None
```

### `_unet_step` change

Branch on kind, keeping the consistency path as the default:

```python
if self._inference_kind == "consistency":
    denoised = self.c_out * model_pred + self.c_skip * x_t
else:  # "diffusion"
    denoised = self.scheduler.step(
        model_pred, t, x_t, return_dict=False,
        **self._scheduler_step_kwargs,
    )[0]
```

### RCFG `cfg_type="self"` branch

Reads alpha/beta. At N=1 these are length-1 tensors and the slicing math
degenerates. For diffusion-kind, alpha/beta are None — that branch
crashes if entered. Two options:

1. **Forbid `cfg_type="self"` on diffusion-kind models.** Simplest.
   Diffusion models with N=1 typically don't need RCFG anyway (the
   distillation already handles noise/CFG implicitly).
2. **Compute alpha/beta from the scheduler's sigmas** for diffusion
   models (`alpha = 1 / sqrt(1 + sigma²)`, `beta = sigma * alpha` for
   v-prediction). Then RCFG works on both kinds.

Recommend (1) for the first PR, (2) as follow-up if a use case emerges.

## Risks to validate before writing code

1. **TCDScheduler's `eta` plumbing.** Need to look at diffusers' actual
   TCDScheduler.step signature to confirm `eta` is a step-time kwarg
   and not a constructor-time field. Source check before coding.
2. **Lightning's specific timestep array.** Lightning-1step expects a
   specific sigma chosen by the scheduler config (`timestep_spacing="trailing"`)
   plus `num_inference_steps=1`. Verify against the official BD usage
   example and pin via scheduler_kwargs.
3. **TCDScheduler may also expose `get_scalings_for_boundary_condition_discrete`.**
   In diffusers ≥ 0.27 this exists for TCD too. If so, Hyper-SD could
   stay on the consistency path and just thread `eta` through a separate
   route. Worth checking — keeps the diffusion-kind path optional rather
   than mandatory.
4. **`_predict_x0_batch` alpha/beta usage at N=1.** The `[1:]` slices
   produce length-0 tensors; downstream `torch.concat([empty, ones_like(slice[0:1])])`
   needs to behave the same on alpha=None for diffusion-kind. Tripwire
   means `cfg_type="self"` crashes on diffusion-kind — accept and gate.
5. **`negative_prompt_scale` in consistency-mode `_unet_step`.** The
   embedding-space subtraction uses `c_out` — needs to be inside the
   consistency branch. Diffusion-kind handles negative prompts via
   regular CFG (guidance_scale > 1 + negative_prompt_embeds).

## Estimated scope

- N=1-only refactor: **~50 LoC** across `_set_timesteps`, `_unet_step`,
  `_swap_model`, plus `MODEL_PRESETS` schema extension.
- New preset entries (Hyper-SDXL-1step, Lightning-SDXL-1step): **~20 LoC**
  more.
- Diffusion-kind multi-step support (rolling buffer in sigma space):
  **out of scope.** ~200 LoC of math validation. Defer until concrete
  reason exists.

## Sequencing

1. Land the refactor on a new branch off `main`. LCM-only behavior
   unchanged, all existing presets keep working. PR back to main.
2. After the sd-multi-model PR (Turbo-only + DMD2) merges, rebase any
   open work on top.
3. Follow-up PR: add Hyper-SDXL-1step and Lightning-SDXL-1step preset
   entries on top of the refactor. New entries only — no further code
   changes needed.

## Open questions for user

- Are Hyper-SDXL-1step and Lightning-SDXL-1step both wanted, or is one
  enough for the experiment loop?
- Hyper has a 1-step UNet variant (full UNet swap) in addition to the
  LoRA. Either works; LoRA is smaller (200 MB vs 5 GB). Prefer LoRA?
- For Hyper specifically: stay on consistency path (if TCDScheduler
  exposes the boundary function) vs. move to diffusion path. The risk
  check above covers this.
