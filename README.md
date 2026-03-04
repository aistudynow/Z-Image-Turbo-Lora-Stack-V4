# ZImageTurboLoraStackV4 (No Nunchaku)

Standalone ComfyUI custom node that applies LoRAs to `MODEL` only, with key translation for Z-Image style naming.

## What it does

- Accepts `MODEL` + `CLIP` input and returns `MODEL` + `CLIP`.
- Applies LoRA only to `MODEL` (`CLIP` is passed through unchanged).
- Supports up to 10 LoRA slots (`lora_count`, `enabled_i`, `lora_name_i`, `strength_i`).
- Skips processing when strength is `0`.
- Extracts the core transformer (`model.model.diffusion_model`) for mapping checks.
- Rewrites `to_qkv -> qkv` automatically.
- Attempts fusion remapping for:
  - `to_q/to_k/to_v -> qkv`
  - `w1/w3 -> w13`
- Falls back to ComfyUI default loader if custom translation fails.

## Install

1. Copy this folder into ComfyUI custom nodes:
   - `ComfyUI/custom_nodes/ZImageTurboLoraStackV4`
2. Restart ComfyUI.
3. Add node: `ZImage Turbo LoRA Stack V4 (No Nunchaku)`.

## Files

- `__init__.py`
- `zimage_turbo_lora_stack_v4.py`
