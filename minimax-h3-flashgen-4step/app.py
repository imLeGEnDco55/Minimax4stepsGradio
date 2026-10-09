"""MiniMax-H3 4-Step FlashGen LoRA — text (and optional keyframes) to video + audio in 4 steps.

The FlashGen LoRA (``Beidouqixing/minimax-h3-4step-lora-flashgen``) is a data-free DMD2 distillation
adapter (rank 64, alpha == rank) that reduces MiniMax-H3's T2VA inference from ~20 steps to **4**. It
targets the ``transformer/`` partition in the original checkpoint's naming (``blocks.N.attn.qkv_proj``,
``mlp.fc1/fc2``, ``attn.out_proj``, ``adaln_proj.linear``). Following ``merge_lora_ckpt.py`` — the merge
script the LoRA repo ships — it is *merged* into the base weights at startup rather than attached as a
runtime PEFT adapter: ``W' = W + scale * (lora_B @ lora_A)`` at ``scale = 1.0``, with every target
required to resolve. See the LoRA section below for the layout transforms the diffusers port needs.

The distilled schedule ``[1.0, 0.7, 0.4, 0.15, 0.0]`` is a ``base_schedule``: it replaces the *uniform*
``linspace(1, 0, num_inference_steps)`` grid that ``MiniMaxH3Scheduler.set_timesteps`` builds, and H3's
per-modality exponential sigma shift (``shift = 12.0`` video, ``3.0`` audio — the release's
``_minimax_h3.sigma_shift_scales``) still applies on top of it. A single ``base_schedule`` serves two
modalities precisely because it is pre-shift. It is injected through
``MiniMaxH3Scheduler.set_timesteps(sigmas=...)``, which takes a sigma grid verbatim, so each scheduler is
handed ``BASE_SCHEDULE`` already pushed through *its own* shift. Note that ``num_inference_steps`` counts
*sigma grid points* for this scheduler (the terminal ``0.0`` included), so a 4-step schedule is 5 points
and the pipeline has to be asked for 5, not 4 — and because the pipeline re-runs ``set_timesteps`` itself
inside the denoise step, the distilled grid is pinned onto both schedulers rather than merely set before
the call.

H3's 62 GiB Qwen3-VL text encoder does not fit alongside the 66 GiB transformer on a single ZeroGPU
worker, so conditioning is delegated to ``multimodalart/qwen3vl-conditioner`` over the gradio API —
the same split every MiniMax-H3 LoRA Space uses.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from functools import cache

import spaces
import gradio as gr
import torch

MODEL_REPO = os.environ.get("H3_MODEL_REPO", "MiniMaxAI/MiniMax-H3")
LORA_REPO = os.environ.get("H3_LORA_REPO", "Beidouqixing/minimax-h3-4step-lora-flashgen")
LORA_FILE = os.environ.get("H3_LORA_FILE", "minimax_h3_4step_lora_flashgen_v1.0_768p_bf16.safetensors")
CONDITIONER_SPACE = os.environ.get("H3_CONDITIONER", "multimodalart/qwen3vl-conditioner")
# ``pack`` places the transformer at startup (66 GB on disk via the spaces pack); VAEs move on first GPU call.
PLACEMENT = os.environ.get("H3_PLACEMENT", "pack").lower()
ATTENTION = os.environ.get("H3_ATTENTION", "_native_cudnn").lower()
GPU_SIZE = os.environ.get("H3_GPU_SIZE", "xlarge")

# The distilled schedule from the LoRA README: 5 *base* sigma points = 4 Euler steps. This is the grid
# before H3's per-modality shift, i.e. the replacement for `linspace(1, 0, num_inference_steps)` — see
# `shifted_schedule` below.
BASE_SCHEDULE = [1.0, 0.7, 0.4, 0.15, 0.0]
NUM_STEPS = len(BASE_SCHEDULE) - 1  # 4 model evaluations
# `MiniMaxH3Scheduler.set_timesteps(num_inference_steps=...)` counts *sigma grid points*, terminal 0.0
# included, and runs `len(sigmas) - 1` model evaluations (`timesteps = 1 - sigmas[:-1]`). So the number the
# pipeline has to be given for NUM_STEPS Euler steps is NUM_STEPS + 1 == len(BASE_SCHEDULE); passing 4 there
# yields a 4-point grid and only 3 steps.
NUM_SIGMA_POINTS = len(BASE_SCHEDULE)  # 5

FPS, FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 24, 17, 5
MIN_UI_DURATION, MAX_UI_DURATION = 2, 14

CANVASES = {
    # 16:9
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    # 9:16
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    # 1:1
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    # 4:3 / 3:4
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    # 21:9
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"

OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "h3-flashgen-out")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PIPE = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None


def snap_frames(seconds: float) -> int:
    """The frame count MiniMax-H3's video VAE can decode: the next ``17 * n + 5`` at 24 fps."""
    frames = max(1, round(float(seconds) * FPS))
    while frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        frames += 1
    return frames


def lower_duration_floor(seconds: float = MIN_UI_DURATION) -> None:
    """Let the pipeline generate below its 5 s floor."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    MiniMaxH3ModularPipeline.min_duration = property(lambda self: float(seconds))


def shifted_schedule(shift: float) -> list[float]:
    """``BASE_SCHEDULE`` pushed through H3's exponential sigma shift ``s*σ / (1 + (s - 1)*σ)``.

    ``base_schedule`` is what ``merge_lora_ckpt.py`` pins into ``_minimax_h3.base_schedule``, and the name
    is literal: it replaces the *base* grid — ``MiniMaxH3Scheduler.set_timesteps``'s
    ``linspace(1, 0, num_inference_steps)``, which the merge script calls "the server's uniform schedule" —
    and the release's per-modality ``sigma_shift_scales`` (video ``12.0``, audio ``3.0``) still apply after
    it. One schedule can only serve both modalities because it is pre-shift.

    Feeding the base grid to the model as if it were already shifted is what produced the ghosted,
    superimposed-frames output: every step conditioned the transformer on a timestep far from the latent's
    real noise level (σ 0.7 instead of 0.966, 0.4 instead of 0.889, 0.15 instead of 0.679), so each
    prediction was an average over noise levels the student was never distilled at. The shift fixes 1.0 and
    0.0, so the endpoints — and the 4-step count — are unchanged.
    """
    return [shift * sigma / (1 + (shift - 1) * sigma) for sigma in BASE_SCHEDULE]


def pin_distilled_schedule(scheduler) -> None:
    """Pin the shifted ``BASE_SCHEDULE`` onto a scheduler so *every* ``set_timesteps`` call lands on it.

    Setting the sigmas before calling the pipeline is not enough: the denoise step contains
    ``MiniMaxH3SetTimestepsStep``, which calls ``scheduler.set_timesteps(num_inference_steps, device=...)``
    itself and so rebuilds the uniform shifted grid, throwing the distilled one away. That grid is also
    where the missing step went — ``num_inference_steps`` is a *grid point* count for this scheduler, so
    the 4 we asked for produced a 4-sigma grid, i.e. ``len(sigmas) - 1 == 3`` model evaluations.

    The schedule pinned here is ``BASE_SCHEDULE`` under *this* scheduler's own ``shift``, so the video
    scheduler (``shift = 12.0``) and the audio one (``shift = 3.0``) keep the distinct grids the two
    modalities are denoised on. ``set_timesteps(sigmas=...)`` takes its argument verbatim, so the shift has
    to be applied here rather than left to the scheduler.

    Pinning is the equivalent of ``merge_lora_ckpt.py``'s ``_minimax_h3.base_schedule`` in
    ``model_index.json``: the served checkpoint carries its distilled schedule and the caller's step count
    cannot silently replace it.
    """
    if getattr(scheduler, "_flashgen_pinned", False):
        return

    original = scheduler.set_timesteps
    pinned = shifted_schedule(scheduler.shift)

    def set_timesteps(num_inference_steps=None, device=None, sigmas=None):
        return original(device=device, sigmas=pinned if sigmas is None else sigmas)

    scheduler.set_timesteps = set_timesteps
    scheduler._flashgen_pinned = True


# ── LoRA loading (weight folding) ─────────────────────────────────────────────
#
# This mirrors ``merge_lora_ckpt.py``, the merge script the LoRA repo itself ships, rather than a
# default ``load_lora_adapter`` + ``set_adapters``. That script folds the adapter into the base
# weights ahead of serving (``W' = W + scale * (lora_B @ lora_A)``, accumulated in fp32, with
# ``scale = lora_alpha / rank = 1.0`` because FlashGen's alpha equals its rank of 64), pins the
# distilled ``base_schedule``, and hard-fails if any LoRA target finds no base weight. We reproduce
# all of that here — merge-and-unload semantics, not a runtime adapter.
#
# The one thing the reference script never has to do is translate names: it merges into the
# *original* MiniMax-H3 partition, whose keys are exactly the LoRA's own
# (``blocks.N.attn.qkv_proj.weight``). Merging into the diffusers port instead means replaying the
# very same transforms ``scripts/convert_minimax_h3_to_diffusers.py`` applied to the base weights,
# because a delta is only valid in the layout of the weight it is added to:
#
#   * ``blocks.`` -> ``transformer_blocks.``, ``token_refiner.blocks.`` ->
#     ``token_refiner.refiner_blocks.``, ``final_layer.adaln_proj.linear`` -> ``norm_out.linear``,
#     ``attn.out_proj`` -> ``attn.to_out.0``, ``mlp.fc2`` -> ``ff.net.2`` (pure renames),
#   * ``mlp.fc1`` -> ``ff.net.0.proj`` with its two fused halves swapped, because diffusers' SwiGLU
#     reads ``[value; gate]`` where the checkpoint stores ``[gate; value]``,
#   * ``attn.qkv_proj`` -> ``to_q`` / ``to_k`` / ``to_v``, and this is the subtle one: the fused QKV
#     rows of the original checkpoint are **per-head interleaved**
#     (``[head0: q k v, head1: q k v, ...]``), so they must be de-interleaved into
#     ``[q_all; k_all; v_all]`` *before* being split into thirds. Splitting the raw rows into
#     contiguous thirds directly — the naive reading — scatters every head's q/k/v across all three
#     projections, which turns the adapter into structured noise on all 52 attention blocks.


def _reorder_interleaved_qkv(weight, num_heads: int, head_dim: int):
    """De-interleave per-head fused-QKV rows into ``[q_all; k_all; v_all]``.

    Identical to ``reorder_interleaved_qkv`` in diffusers' MiniMax-H3 conversion script, which
    applies it to the base fused QKV weight before splitting it. ``lora_B`` indexes that same fused
    output space, so the delta needs the same reorder.
    """
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    query, key, value = grouped.split(head_dim, dim=1)
    return torch.cat(
        [part.reshape(num_heads * head_dim, *weight.shape[1:]) for part in (query, key, value)], dim=0
    )


def _lora_target_name(source_name: str) -> str:
    """Map an original-checkpoint base name (without ``.lora_A/B.*``) to the diffusers parameter path."""
    if source_name.startswith("token_refiner.blocks."):
        target = source_name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif source_name.startswith("blocks."):
        target = source_name.replace("blocks.", "transformer_blocks.", 1)
    else:
        target = source_name
    return target.replace("final_layer.adaln_proj.linear", "norm_out.linear")


def _lora_targets(name: str, b_weight, num_heads: int, head_dim: int):
    """Yield ``(diffusers_param_key, row_transformed_B)`` for one original LoRA base name."""
    target = _lora_target_name(name)

    if target.endswith(".attn.qkv_proj"):
        prefix = target.removesuffix("qkv_proj")
        # De-interleave the per-head rows first, then split contiguous thirds — exactly the
        # composition the base fused QKV weight went through during conversion.
        rows = _reorder_interleaved_qkv(b_weight, num_heads, head_dim)
        for kind, part in zip(("q", "k", "v"), rows.split(num_heads * head_dim, dim=0)):
            yield f"{prefix}to_{kind}.weight", part.contiguous()
    elif target.endswith(".mlp.fc1"):
        # SwiGLU gate/value swap: the checkpoint stores [gate, value]; diffusers stores [value, gate].
        gate, value = b_weight.chunk(2, dim=0)
        yield target.replace(".mlp.fc1", ".ff.net.0.proj") + ".weight", torch.cat([value, gate]).contiguous()
    elif target.endswith(".mlp.fc2"):
        yield target.replace(".mlp.fc2", ".ff.net.2") + ".weight", b_weight
    elif target.endswith(".attn.out_proj"):
        yield target.replace(".attn.out_proj", ".attn.to_out.0") + ".weight", b_weight
    else:
        # adaln_proj.linear (block-level) and norm_out.linear: identical row layout.
        yield target + ".weight", b_weight


def load_and_apply_lora(transformer) -> str:
    """Download the FlashGen LoRA, remap keys, and fold ``scale * (B @ A)`` into the base weights."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora_path = hf_hub_download(LORA_REPO, LORA_FILE)
    lora = load_file(lora_path)

    # Same pairing and validation contract as merge_lora_ckpt.py's `load_lora_pairs`.
    suffix_a, suffix_b = ".lora_A.default.weight", ".lora_B.default.weight"
    bases = sorted({key[: -len(suffix_a)] for key in lora if key.endswith(suffix_a)})
    unexpected = [key for key in lora if not key.endswith((suffix_a, suffix_b))]
    if unexpected:
        raise ValueError(f"{LORA_FILE} holds {len(unexpected)} non-LoRA tensors, e.g. {unexpected[:5]}")
    if not bases:
        raise ValueError(f"No lora_A/lora_B pairs found in {LORA_FILE}")

    ranks = set()
    for name in bases:
        if f"{name}{suffix_b}" not in lora:
            raise ValueError(f"LoRA is missing the lora_B twin of {name}{suffix_a}")
        ranks.add(lora[f"{name}{suffix_a}"].shape[0])
    if len(ranks) != 1:
        raise ValueError(f"LoRA mixes ranks {sorted(ranks)}; a DMD2 student export uses a single rank")
    rank = ranks.pop()

    num_heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    # merge_lora_ckpt.py's --scale, i.e. lora_alpha / rank; 1.0 because FlashGen's alpha == rank.
    scale = float(os.environ.get("H3_LORA_SCALE", "1.0"))

    params = dict(transformer.named_parameters())
    folded, missed = 0, []
    with torch.no_grad():
        for name in bases:
            a = lora[f"{name}{suffix_a}"]
            b = lora[f"{name}{suffix_b}"]
            for key, b_part in _lora_targets(name, b, num_heads, head_dim):
                param = params.get(key)
                if param is None:
                    missed.append(key)
                    continue
                # Accumulate in fp32 so a bf16 base does not swallow the update.
                delta = scale * (b_part.to(torch.float32) @ a.to(torch.float32))
                if delta.shape != param.shape:
                    raise ValueError(
                        f"LoRA delta for `{key}` has shape {tuple(delta.shape)}, "
                        f"base weight is {tuple(param.shape)}"
                    )
                param.data = (param.data.float() + delta.to(param.device)).to(param.dtype)
                folded += 1

    # merge_lora_ckpt.py treats an unmatched target as fatal, and so do we: a silently skipped
    # target means the name mapping is wrong and the demo would serve a half-applied adapter.
    if missed:
        raise ValueError(
            f"{len(missed)} LoRA targets matched no transformer weight, e.g. {missed[:5]}. "
            "The LoRA and the diffusers transformer disagree on module naming."
        )

    return (
        f"FlashGen LoRA merged · {len(bases)} targets, rank {rank}, scale {scale:g} "
        f"-> {folded} weight deltas · {LORA_FILE}"
    )


# ── Model loading ───────────────────────────────────────────────────────────


def load_models() -> str | None:
    """Load the denoising half at startup: transformer + VAEs + schedulers, then fold the LoRA in."""
    global PIPE, LOAD_ERROR, LOADED_IN

    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch
        from diffusers import ComponentsManager

        from h3_split_blocks import MiniMaxH3GeneratorBlocks

        lower_duration_floor()
        manager = ComponentsManager()
        blocks = MiniMaxH3GeneratorBlocks()
        print(f"[gen] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO, components_manager=manager, collection="h3")
        pipe.load_components(dtype=torch.bfloat16)

        pipe.vae.set_attention_backend("native")
        pipe.audio_vae.set_attention_backend("native")
        pipe.transformer.set_attention_backend(ATTENTION)

        # Fold the FlashGen LoRA into the transformer before any GPU placement.
        lora_status = load_and_apply_lora(pipe.transformer)
        print(f"[gen] {lora_status}", flush=True)

        # ... and pin the distilled schedule the merged weights were trained for, so the pipeline's own
        # `set_timesteps` call cannot fall back to a uniform grid (and to NUM_STEPS - 1 evaluations).
        pin_distilled_schedule(pipe.scheduler)
        pin_distilled_schedule(pipe.audio_scheduler)
        print(
            f"[gen] distilled base_schedule {BASE_SCHEDULE} pinned -> {NUM_STEPS} steps; "
            f"video σ (shift {pipe.scheduler.shift:g}) "
            f"{[round(s, 4) for s in shifted_schedule(pipe.scheduler.shift)]}, "
            f"audio σ (shift {pipe.audio_scheduler.shift:g}) "
            f"{[round(s, 4) for s in shifted_schedule(pipe.audio_scheduler.shift)]}",
            flush=True,
        )

        if PLACEMENT == "pack":
            pipe.transformer.to("cuda")

        PIPE = pipe
        LOADED_IN = time.time() - started
        print(f"[gen] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = f"**Loading failed** after {time.time() - started:.0f}s: `{type(error).__name__}: {error}`"
    return LOAD_ERROR


# ── Remote conditioning ──────────────────────────────────────────────────────


@cache
def conditioner():
    from gradio_client import Client

    return Client(CONDITIONER_SPACE)


def encode_remote(prompt, image_path, last_image_path, canvas, num_frames, ip_token=None, rewrite_prompt=False):
    """Call the conditioner Space to get ``prompt_embeds`` + ``text_token_tags`` + resolved geometry.

    ``rewrite_prompt`` is the same prompt-enhancement path ``multimodalart/minimax-h3`` uses: the conditioner
    Space already hosts H3's Qwen3-VL, so it first has that model *rewrite* the prompt into MiniMax-H3's
    structured shot description (the LLM-based rewriting step) and then encodes the rewritten text instead of
    the raw one. The rewritten text comes back in the plan under ``refined_prompt`` (``None`` when no rewrite
    happened). Encoding the enhanced prompt on the conditioner keeps the enhancement and the embeddings in
    sync — nothing about the FlashGen merge, the 4-step schedule, or the VAEs is involved.
    """
    from gradio_client import handle_file
    from safetensors import safe_open

    def call():
        return conditioner_client(ip_token).predict(
            prompt=prompt,
            image_path=handle_file(image_path) if image_path else None,
            last_image_path=handle_file(last_image_path) if last_image_path else None,
            canvas=canvas,
            num_frames=num_frames,
            rewrite_prompt=bool(rewrite_prompt),
            api_name="/encode",
        )

    try:
        path, plan = call()
    except Exception as first:
        print(f"[conditioner] retrying with a fresh client after: {first}", flush=True)
        conditioner.cache_clear()
        path, plan = call()

    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
        return handle.get_tensor("prompt_embeds"), handle.get_tensor("text_token_tags"), metadata, plan


def conditioner_client(ip_token):
    if not ip_token:
        return conditioner()
    from gradio_client import Client

    return Client(CONDITIONER_SPACE, headers={"x-ip-token": ip_token})


# ── Geometry helpers ────────────────────────────────────────────────────────


def _snap_canvas(aspect: float, current_canvas: str) -> str:
    """The supported canvas whose aspect ratio is closest to ``aspect``.

    Same selection ``multimodalart/minimax-h3`` makes: among the canvases sharing a ratio keep the
    smallest (fastest) one, then take the ratio nearest the image's — so a 3:4 photo renders 3:4 rather
    than being letterboxed into the 16:9 default. A canvas the user already picked is kept when it is at
    least as close to the image as the best candidate, so ``1344x768`` survives a 1.75 image.
    """
    fastest: dict[float, tuple[str, tuple[int, int]]] = {}
    for label, (h, w) in CANVASES.items():
        r = w / h
        if r not in fastest or w * h < fastest[r][1][0] * fastest[r][1][1]:
            fastest[r] = (label, (h, w))
    ratio = min(fastest, key=lambda r: abs(r - aspect))

    cur_h, cur_w = CANVASES[current_canvas]
    if abs(cur_w / cur_h - aspect) <= abs(ratio - aspect):
        return current_canvas
    return fastest[ratio][0]


def _cover_crop(image_path: str, canvas_label: str) -> str:
    """Center cover-crop the file at ``image_path`` to ``canvas_label``'s aspect ratio, in place.

    The keyframe has to arrive at the model already matching the canvas, otherwise the pipeline
    letterboxes/stretches it and the first frame does not line up with the generated ones.
    """
    from PIL import Image as _Image

    h, w = CANVASES[canvas_label]
    target = w / h
    img = _Image.open(image_path)
    if abs(img.width / img.height - target) <= 1e-3:
        return image_path
    if img.width / img.height > target:
        new_w = int(img.height * target)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = int(img.width / target)
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    img.save(image_path)
    return image_path


def _fit_keyframe(image_path, current_canvas):
    """Snap the canvas to a keyframe's aspect ratio, then cover-crop the keyframe to that canvas."""
    from PIL import Image as _Image

    with _Image.open(image_path) as img:
        aspect = img.width / img.height
    label = _snap_canvas(aspect, current_canvas)
    return _cover_crop(image_path, label), label


def _fit_keyframe_ui(image_path, current_canvas):
    """``image.upload`` handler — the reference Space's behaviour, ported.

    Doing this on upload (rather than only server-side inside ``generate``) is the whole point: the
    user immediately sees the keyframe cropped to the ratio the video will be rendered at, and the
    Canvas dropdown moves to the snapped resolution instead of sitting on the 16:9 default.
    """
    path = _as_path(image_path)
    if not path:
        return gr.update(), gr.update()
    cropped, label = _fit_keyframe(path, current_canvas)
    return gr.update(value=cropped), gr.update(value=label)


def _fit_last_keyframe_ui(last_image_path, image_path, current_canvas):
    """``last_image.upload`` handler. The first frame owns the canvas when there is one, so a last
    frame uploaded on top of it is only cover-cropped to the canvas already chosen."""
    path = _as_path(last_image_path)
    if not path:
        return gr.update(), gr.update()
    if _as_path(image_path):
        return gr.update(value=_cover_crop(path, current_canvas)), gr.update()
    return _fit_keyframe_ui(path, current_canvas)


def _as_path(value):
    if isinstance(value, dict):
        value = value.get("path") or (value.get("url") or "").removeprefix("/gradio_api/file=")
    return value or None


# ── GPU duration estimation ─────────────────────────────────────────────────

_DUR_B, _DUR_C = 1.1745e-4, 3.8396e-9
_DECODE_BASE, _DECODE_PER_DEFAULT_CANVAS, _DEFAULT_CANVAS_PIXELS = 15, 15, 960 * 544 * 124
_PLACEMENT_ALLOWANCE, _PAD = 12, 10


def get_duration(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, seed, *a, **k):
    """Estimate GPU duration for the ``_generate`` call. Steps is fixed at NUM_STEPS (4)."""
    height, width, num_frames = int(height), int(width), int(num_frames)
    steps = NUM_STEPS
    latent_frames = (num_frames - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2
    patches = (height // 32) * (width // 32)
    rows = latent_frames * patches + (int(image is not None) + int(last_image is not None)) * patches
    denoise = steps * (_DUR_B * rows + _DUR_C * rows**2)
    decode = _DECODE_BASE + _DECODE_PER_DEFAULT_CANVAS * (height * width * num_frames) / _DEFAULT_CANVAS_PIXELS
    return max(60, int(denoise + decode) + _PLACEMENT_ALLOWANCE + _PAD)


# ── Inference ───────────────────────────────────────────────────────────────


@spaces.GPU(duration=get_duration, size=GPU_SIZE)
def _generate(prompt_embeds, text_token_tags, image, last_image, height, width, num_frames, seed):
    """Denoise + decode on GPU time. The LoRA is already folded at startup; 4 steps with custom sigmas."""
    import torch

    if PLACEMENT == "pack":
        PIPE.vae.to("cuda")
        PIPE.audio_vae.to("cuda")
    elif PLACEMENT == "lazy":
        PIPE.to("cuda")

    # Inject the distilled base_schedule into both schedulers, and keep it there: the pipeline calls
    # `set_timesteps` again inside its denoise step, so the schedule has to survive that call.
    pin_distilled_schedule(PIPE.scheduler)
    pin_distilled_schedule(PIPE.audio_scheduler)
    PIPE.scheduler.set_timesteps(device="cuda")
    PIPE.audio_scheduler.set_timesteps(device="cuda")

    state = PIPE(
        prompt_embeds=prompt_embeds.to("cuda"),
        text_token_tags=text_token_tags,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
        num_frames=num_frames,
        # Grid points, terminal 0.0 included -> NUM_STEPS Euler steps. See NUM_SIGMA_POINTS above.
        num_inference_steps=NUM_SIGMA_POINTS,
        generator=torch.Generator("cpu").manual_seed(int(seed)),
    )
    # Read the step count back off the scheduler rather than trusting the constant: `step_index` counted
    # the Euler updates the denoise loop actually took, and `sigmas` is the grid it took them on.
    steps_taken = int(PIPE.scheduler.step_index or 0)
    sigmas_used = [round(float(s), 4) for s in PIPE.scheduler.sigmas.tolist()]
    return (
        state.get("videos")[0],
        state.get("audio")[0].cpu(),
        state.get("sampling_rate"),
        steps_taken,
        sigmas_used,
    )


def _caller_ip_token() -> str | None:
    from gradio.context import LocalContext

    request = LocalContext.request.get()
    return request.headers.get("x-ip-token") if request is not None else None


def generate(
    prompt,
    image_path=None,
    last_image_path=None,
    canvas=DEFAULT_CANVAS,
    duration=5,
    seed=42,
    enhance_prompt=False,
    progress=gr.Progress(track_tqdm=True),
):
    """One request: prompt (+ optional first/last frame) -> video with synchronized audio.

    ``enhance_prompt`` turns on the conditioner's LLM prompt rewriting (the same enhancement
    ``multimodalart/minimax-h3`` offers); off, the raw prompt is encoded verbatim as before. It is last and
    defaults to off, so a positional API client written against the previous signature is unaffected.
    """
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR.replace("**", "").replace("`", ""))
    if PIPE is None:
        raise gr.Error("The model is still loading — watch the Space logs and retry shortly.")
    if not prompt or not prompt.strip():
        raise gr.Error("MiniMax-H3 always takes a prompt, keyframes or not.")

    from PIL import Image, ImageOps

    from diffusers.utils import encode_video

    # The UI already snapped + cropped on upload, so this is a no-op there; it still runs for API
    # callers (and for a canvas the user changed afterwards) so the output geometry always follows the
    # I2V image. The *first* frame owns the canvas — a last frame is only cover-cropped to match it.
    first, last = _as_path(image_path), _as_path(last_image_path)
    if first:
        first, canvas = _fit_keyframe(first, canvas)
        if last:
            last = _cover_crop(last, canvas)
    elif last:
        last, canvas = _fit_keyframe(last, canvas)

    num_frames = snap_frames(duration)

    progress(0.0, desc="Enhancing the prompt, then encoding ..." if enhance_prompt else "Encoding prompt ...")
    conditioned = time.time()
    prompt_embeds, text_token_tags, metadata, plan = encode_remote(
        prompt,
        first,
        last,
        canvas,
        num_frames,
        ip_token=_caller_ip_token(),
        rewrite_prompt=bool(enhance_prompt),
    )
    condition_seconds = time.time() - conditioned
    height, width, num_frames = (int(metadata[key]) for key in ("height", "width", "num_frames"))
    # `None` whenever no rewrite happened — either none was asked for, or the conditioner declined one.
    enhanced = (plan or {}).get("refined_prompt") or ""

    def keyframe(path):
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB") if path else None

    progress(0.1, desc=f"Generating {num_frames / FPS:.1f}s at {width}x{height} in {NUM_STEPS} steps ...")
    started = time.time()
    frames, audio, sampling_rate, steps_taken, sigmas_used = _generate(
        prompt_embeds,
        text_token_tags,
        keyframe(first),
        keyframe(last),
        height,
        width,
        num_frames,
        seed,
    )
    generate_seconds = time.time() - started

    path = os.path.join(OUTPUT_DIR, f"h3-flashgen-{int(time.time() * 1000)}.mp4")
    encode_video(frames, fps=FPS, output_path=path, audio=audio, audio_sample_rate=sampling_rate)

    # Reported, not assumed: `steps_taken` is the number of Euler updates the scheduler actually took.
    schedule = " → ".join(f"{s:g}" for s in sigmas_used)
    report = (
        f"{width}x{height} · {num_frames} frames ({num_frames / FPS:.3f}s) · "
        f"{steps_taken} steps (FlashGen σ {schedule}) · "
        f"conditioner {condition_seconds:.0f}s{' (prompt enhanced)' if enhanced else ''} · "
        f"denoise + decode {generate_seconds:.0f}s · seed {int(seed)}"
    )
    print(f"[gen] {report}", flush=True)
    expected_sigmas = [round(s, 4) for s in shifted_schedule(PIPE.scheduler.shift)]
    if steps_taken != NUM_STEPS or sigmas_used != expected_sigmas:
        print(
            f"[gen] WARNING: expected {NUM_STEPS} steps on {expected_sigmas} "
            f"(base_schedule {BASE_SCHEDULE} at shift {PIPE.scheduler.shift:g}), "
            f"ran {steps_taken} on {sigmas_used}",
            flush=True,
        )
    return path, report, enhanced, gr.update(visible=bool(enhanced))


# ── UI ──────────────────────────────────────────────────────────────────────


INTRO = """# MiniMax-H3 4-Step FlashGen LoRA

<div align="center">
  <a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank" rel="noopener"><strong>[ base model ]</strong></a> &nbsp;
  <a href="https://huggingface.co/Beidouqixing/minimax-h3-4step-lora-flashgen" target="_blank" rel="noopener"><strong>[ LoRA ]</strong></a>
</div>

**MiniMax-H3** is a 33B omni-modal model that generates video with native stereo audio.
This Space applies the **FlashGen 4-step distillation LoRA** by
[Beidouqixing](https://huggingface.co/Beidouqixing), which uses distribution-matching distillation
to reduce inference from ~20 steps to just **4 steps** — a ~5x speedup.

Type a prompt (and optionally add a first frame, last frame, or both for FL2VA mode),
pick a canvas, and generate. The output includes a synchronized soundtrack."""

CSS = """
.main.fillable { max-width: 1100px !important; }
"""

EXAMPLE_PROMPTS = [
    "Cinematic, medium wide shot. A golden retriever puppy chases autumn leaves through a sunlit park, leaves swirling around it. Warm afternoon light, shallow depth of field. The dog barks playfully and leaves rustle.",
    "A serene Japanese garden with a koi pond. Cherry blossom petals drift slowly on the water's surface. A small wooden bridge arcs over the pond. Gentle wind chimes tinkle in the breeze. Ambient birdsong and water sounds.",
    "Close-up of a chef's hands kneading dough on a flour-dusted wooden board. Flour particles float in the warm kitchen light. The dough stretches and folds rhythmically. Soft thumping sounds of dough and ambient kitchen noise.",
    "A neon-lit cyberpunk street in the rain. Reflected neon signs shimmer in puddles. A figure in a long coat walks away from camera. Rain patters on metal awnings, distant synth music echoes.",
    "A majestic eagle soars over snow-capped mountains at golden hour. The camera follows from behind, sweeping past peaks. Wind rushes past. Distant orchestral music swells.",
]

with gr.Blocks(title="MiniMax-H3 4-Step FlashGen") as demo:
    gr.Markdown(INTRO)

    with gr.Row(equal_height=True):
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=4,
                value=EXAMPLE_PROMPTS[0],
                placeholder="Describe the scene, shot style, and soundscape ...",
            )
            enhance = gr.Checkbox(
                label="Enhance prompt",
                value=False,
            )
            with gr.Accordion("Image-to-video (optional)", open=True):
                with gr.Row():
                    image = gr.Image(label="First frame (optional)", type="filepath", height=200)
                    last_image = gr.Image(label="Last frame (optional)", type="filepath", height=200)

            run = gr.Button("Generate", variant="primary", size="lg")

            with gr.Accordion("Advanced", open=False):
                canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
                duration = gr.Slider(
                    label="Duration (s)",
                    minimum=MIN_UI_DURATION,
                    maximum=MAX_UI_DURATION,
                    step=1,
                    value=5,
                )
                seed = gr.Number(label="Seed", value=42, precision=0)

        with gr.Column():
            result = gr.Video(label="Video + soundtrack", autoplay=True)
            report = gr.Markdown()
            # An output, so it is revealed only for a request that actually asked for an enhancement.
            with gr.Accordion("Enhanced prompt", open=False, visible=False) as enhanced_panel:
                enhanced_box = gr.Textbox(show_label=False, lines=8, interactive=False)

    OUTPUTS = [result, report, enhanced_box, enhanced_panel]

    # An uploaded I2V frame decides the output geometry: snap Canvas to the nearest supported ratio and
    # cover-crop the frame to it, right away, so what is shown is what will be rendered.
    image.upload(_fit_keyframe_ui, [image, canvas], [image, canvas])
    last_image.upload(_fit_last_keyframe_ui, [last_image, image, canvas], [last_image, canvas])

    gr.Examples(
        examples=[[p] for p in EXAMPLE_PROMPTS],
        inputs=[prompt],
        fn=generate,
        outputs=OUTPUTS,
        cache_examples=True,
        cache_mode="lazy",
    )

    run.click(
        generate,
        inputs=[prompt, image, last_image, canvas, duration, seed, enhance],
        outputs=OUTPUTS,
        api_name="generate",
    )
    prompt.submit(
        generate,
        inputs=[prompt, image, last_image, canvas, duration, seed, enhance],
        outputs=OUTPUTS,
        api_name="generate_submit",
    )

load_models()

if __name__ == "__main__":
    demo.launch(show_error=True, theme=gr.themes.Citrus(), css=CSS, allowed_paths=[OUTPUT_DIR])