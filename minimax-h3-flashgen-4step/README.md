---
title: MiniMax-H3 4-Step FlashGen LoRA
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
short_description: 4-step text-to-video+audio with MiniMax-H3 FlashGen LoRA
models:
  - MiniMaxAI/MiniMax-H3
  - Beidouqixing/minimax-h3-4step-lora-flashgen
---

# MiniMax-H3 4-Step FlashGen LoRA

A demo of the [FlashGen 4-step distillation LoRA](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen)
applied to [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), MiniMax's 33B omni-modal model that generates
video with native stereo audio.

The FlashGen LoRA uses distribution-matching distillation (DMD2 / VSD, data-free) to reduce inference from ~20 steps
to just **4 steps** — approximately a 5x speedup. The distilled `base_schedule` `[1.0, 0.7, 0.4, 0.15, 0.0]` replaces
the *uniform* `linspace(1, 0, num_inference_steps)` grid `MiniMaxH3Scheduler.set_timesteps` would otherwise build, and
H3's per-modality exponential sigma shift still applies on top of it — a single base schedule can serve both
modalities precisely because it is pre-shift. So the video scheduler runs
σ `1 → 0.9655 → 0.8889 → 0.6792 → 0` (`shift = 12.0`) and the audio scheduler σ `1 → 0.875 → 0.6667 → 0.3462 → 0`
(`shift = 3.0`). Each shifted grid is injected through `MiniMaxH3Scheduler.set_timesteps(sigmas=...)`, which takes it
verbatim, and *pinned* onto its scheduler, because the pipeline's own denoise step calls
`set_timesteps(num_inference_steps, ...)` again — and for this scheduler `num_inference_steps` counts sigma grid
points (terminal `0.0` included), so it runs `len(sigmas) - 1` model evaluations. Pinning is the runtime equivalent of
`merge_lora_ckpt.py` writing `_minimax_h3.base_schedule` into `model_index.json`. Each generation reports the number
of Euler steps the scheduler actually took, read back off `scheduler.step_index`, rather than a hardcoded 4.

## How it works

| Piece | What runs here |
|---|---|
| Transformer | `MiniMaxH3Transformer3DModel` — the 33B / 50-block H3 DiT, bf16, with the FlashGen LoRA folded in |
| VAEs | `AutoencoderKLMiniMaxH3` (video) + `AutoencoderKLMiniMaxH3Audio` (audio), float32 |
| Schedulers | `MiniMaxH3Scheduler` with the 4-step distilled `base_schedule` `[1.0, 0.7, 0.4, 0.15, 0.0]` under each modality's own shift (video `12.0`, audio `3.0`) |
| Conditioning | Delegated to [`multimodalart/qwen3vl-conditioner`](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner) over the gradio API (the 62 GB Qwen3-VL text encoder does not fit alongside the transformer on a single ZeroGPU worker) |

## How the LoRA is applied

The LoRA repo ships its own merge script,
[`merge_lora_ckpt.py`](https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen/blob/main/merge_lora_ckpt.py),
and this Space follows it rather than a default diffusers LoRA load. The adapter is **merged** into the base weights
at startup — `W' = W + scale * (lora_B @ lora_A)`, accumulated in fp32, at `scale = lora_alpha / rank = 1.0`
(FlashGen's alpha equals its rank of 64) — instead of being attached as a runtime PEFT adapter, and all 259 LoRA
targets are required to resolve or startup fails, exactly as the reference script does.

The reference script merges into the *original* MiniMax-H3 partition, whose keys are the LoRA's own
(`blocks.N.attn.qkv_proj.weight`, `mlp.fc1/fc2`, `attn.out_proj`, `adaln_proj.linear`). Merging into the diffusers
port instead requires replaying the same layout transforms
[`convert_minimax_h3_to_diffusers.py`](https://github.com/huggingface/diffusers/blob/main/scripts/convert_minimax_h3_to_diffusers.py)
applied to the base weights, because a delta is only valid in the layout of the weight it is added to:

| Original LoRA target | diffusers parameter | Transform |
|---|---|---|
| `blocks.N.…` / `token_refiner.blocks.N.…` | `transformer_blocks.N.…` / `token_refiner.refiner_blocks.N.…` | rename |
| `attn.out_proj` | `attn.to_out.0` | rename |
| `mlp.fc2` | `ff.net.2` | rename |
| `final_layer.adaln_proj.linear` | `norm_out.linear` | rename |
| `adaln_proj.linear` | `adaln_proj.linear` | identity |
| `mlp.fc1` | `ff.net.0.proj` | fused halves swapped, `[gate; value]` → `[value; gate]` for diffusers' `SwiGLU` |
| `attn.qkv_proj` | `attn.to_q` / `to_k` / `to_v` | **de-interleave the per-head rows** (`[head0: q k v, head1: q k v, …]` → `[q_all; k_all; v_all]`), *then* split thirds |

That last row is the one that is easy to get wrong: the checkpoint's fused QKV rows are per-head interleaved, so
splitting them into contiguous thirds — the naive reading — scatters every head's q/k/v across all three projections
and turns the adapter into noise on all 52 attention blocks. With the de-interleave in place, the merged weights
reproduce `merge_lora_ckpt.py`-then-convert to within bf16 rounding.

## License

MiniMax-H3 is covered by the **MiniMax H3 Community License Agreement**. The FlashGen LoRA is Apache-2.0.
Read both before using any output of this Space.