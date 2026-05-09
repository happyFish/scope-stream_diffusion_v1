# Plan: SDXL ControlNet Support

## Context
Current state: `_ensure_trt_unet` has `if want_control: if self.sdxl: raise NotImplementedError(...)`. SD1.5 ControlNet (eager + TRT) works. SDXL ControlNet works in eager mode through diffusers but the TRT path is unimplemented.

Test target model: `diffusers/controlnet-canny-sdxl-1.0` (paired with `stabilityai/stable-diffusion-xl-base-1.0` or SDXL-Turbo). The DMD2 1-step UNet is a swap — SDXL ControlNet against DMD2 is a stretch goal; verify with the base SDXL UNet first.

## Eager Path (verify first, may already work)
1. Confirm `ControlNetHandler.update()` correctly produces residuals for SDXL-shape inputs. SDXL UNet expects `added_cond_kwargs={"text_embeds": ..., "time_ids": ...}` — the ControlNet model also needs these. Read `diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl` for the canonical wiring.
2. In `_unet_step` (post-extraction: `InferenceCore.unet_step`), when `self.sdxl and self.controlnet`, call ControlNet with the SDXL aug-conditioning, then pass the residuals into `self.unet` along with `added_cond_kwargs`.
3. If this produces correct output, eager SDXL ControlNet is done. Move to TRT.

## TRT Path

### Step 1: New ONNX export wrapper
File: `src/scope_streamdiffusion/_trt/models.py`. Add `UNetSDXLWithControlInputs` modeled on the existing `UNetWithControlInputs` (SD1.5) and `UNetSDXL` (SDXL no-control).

Inputs (in order — must match adapter feed order):
- `sample` (B, 4, H/8, W/8)
- `timestep` (scalar or (B,))
- `encoder_hidden_states` (B, 77, 2048)
- `text_embeds` (B, 1280)  ← SDXL aug
- `time_ids` (B, 6)         ← SDXL aug
- `input_control_00` … `input_control_{N-1}` (down residuals)
- `input_control_middle` (mid residual)

Output: `latent` (same shape as `sample`).

Forward should call `self.unet(sample, timestep, encoder_hidden_states, added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}, down_block_additional_residuals=[...], mid_block_additional_residual=...)`.

### Step 2: New builder
File: `src/scope_streamdiffusion/trt_engines.py`. Add `build_unet_sdxl_with_control_engine(...)` modeled on `build_unet_with_control_engine` + `build_unet_sdxl_engine`. Use the same dynamic-shape ranges as the SDXL UNet build (512–1024).

**Known constraints:**
- ONNX export of SDXL UNet runs ~5 GB. Use `external_data` format. The existing `build_unet_sdxl_engine` already does this — copy its handling.
- ControlNet residuals are full-resolution feature maps; this multiplies the export size by ~10–20%. Expect 6 GB ONNX.
- Static shape recommended for first cut. Generalize to dynamic only after a static build runs.
- Compile time: 5–15 minutes on a 4090. Cache aggressively.

### Step 3: Standalone SDXL ControlNet engine
The SD1.5 path uses a separate `ControlNetEngine` (`src/scope_streamdiffusion/_trt/engine.py`) that produces residuals consumed by `UNet2DConditionModelWithControlEngine`. Mirror this for SDXL:
- Add ONNX wrapper for SDXL ControlNet to `_trt/models.py` (it has the same SDXL aug-conditioning inputs as the UNet wrapper).
- Add builder `build_controlnet_sdxl_engine` in `trt_engines.py`.
- The existing `ControlNetEngine` class in `_trt/engine.py` has hard-coded `block_out_channels=(320, 640, 1280, 1280)` for SD1.5. SDXL ControlNet uses `(320, 640, 1280)` (one fewer block) and produces 9 down residuals + 1 mid (versus 12+1 for SD1.5). Add `ControlNetSDXLEngine` or parameterize `ControlNetEngine` by passing `chans` and `spec` at construction.

### Step 4: New runtime adapter
File: `src/scope_streamdiffusion/trt_engines.py`. Add `TRTUNetSDXLWithControlAdapter` that exposes the diffusers UNet `__call__` signature and dispatches to `UNet2DConditionModelSDXLWithControlEngine` (also new in `_trt/engine.py`) plus the SDXL ControlNet engine.

### Step 5: Wire into TRTLifecycle
In `_ensure_trt_unet`, replace the `raise NotImplementedError` with the SDXL+control branch. Path resolution and cache key must include both UNet and ControlNet model IDs.

### Step 6: TAESD
SDXL TAESD (`madebyollin/taesdxl`) already works via the existing `_ensure_trt_taesd` path. No change needed.

## Testing
1. Eager SDXL + Canny ControlNet on a webcam frame. Output should track edges.
2. TRT SDXL + Canny ControlNet, same prompt. Confirm visual parity within fp16 tolerance.
3. Live-toggle ControlNet on/off mid-stream on SDXL-Turbo.
4. Swap SD-Turbo (SD1.5) ↔ SDXL-Turbo with ControlNet attached. Confirm correct adapter is selected each time.

## Out of Scope (defer)
- Multi-ControlNet on SDXL (do single first).
- DMD2 + ControlNet (the 1-step UNet swap may not respect ControlNet residuals correctly — needs separate investigation).
