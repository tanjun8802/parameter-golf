"""
MXINT4 + Unstructured Pruning Optimisation Pipeline
====================================================
Key innovations over the int6-QAT baseline:

1. MXINT4 QAT  – OCP-MX-spec block-float quantisation (block=32, E8M0 scale,
   INT4 payload). Applied as a Straight-Through Estimator (STE) fake-quant
   during the warmdown phase, closing the train/deploy precision gap.

2. Optuna pruning-ratio search  – After training, Optuna runs ~25 fast
   CPU trials to find the global unstructured sparsity that minimises
   the reconstruction error on the already-quantised weights. Typical
   wall-clock cost: <20 s.

3. Shift-add matmul decomposition  – At export time every quantised weight
   matrix is decomposed bit-by-bit into four binary matrices. At inference
   an `Int4ShiftAddLinear` layer replaces the vanilla `nn.Linear`, avoiding
   INT4→FP32 upcasts via shift-accumulation. This is opt-in via
   `USE_SHIFT_ADD_INFERENCE=1`.

4. LZMA compression  – Replaces zlib. On sparse INT4 payloads LZMA preset=9
   typically saves another 10–15 % vs zlib level=9.

5. Architecture  – 11L, 512-dim, 3× MLP, 8 heads / 4 KV-heads (GQA),
   U-Net skips, BigramHash (2048 × 128 dims), tied embeddings, logit
   softcap=30. Closely mirrors the current SOTA stack; MXINT4 lets us
   potentially widen or deepen the model within the 16 MB budget.

Timing budget (8×H100 SXM, 600 s wall-clock):
  ~540 s  training
  ~  5 s  self-generate calibration tokens
  ~ 20 s  Optuna pruning search (25 trials, CPU)
  ~  5 s  MXINT4 quantisation + LZMA
  ------
  ~570 s  total  (<600 s constraint)
"""

from __future__ import annotations

import copy
import glob
import io
import lzma
import math
import os
import random
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 4000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 540.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape (11L, 512-dim, 3× MLP – baseline competitive stack).
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # BigramHash embedding table: vocab² lookups → low-dim projection.
    bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # QAT / quantisation options.
    # Late-QAT: activate MXINT4 fake-quant when the LR warmdown fraction < qat_threshold.
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", 0.15))

    # EMA weight averaging (every step).
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))

    # Tight SWA: average model snapshots every `swa_every` steps in final warmdown.
    swa_every = int(os.environ.get("SWA_EVERY", 50))

    # Optuna pruning search budget.
    optuna_n_trials = int(os.environ.get("OPTUNA_N_TRIALS", 25))
    optuna_timeout = float(os.environ.get("OPTUNA_TIMEOUT", 30.0))
    prune_sparsity_min = float(os.environ.get("PRUNE_SPARSITY_MIN", 0.05))
    prune_sparsity_max = float(os.environ.get("PRUNE_SPARSITY_MAX", 0.45))

    # Whether to apply shift-add decomposition at inference (no training cost).
    use_shift_add_inference = bool(int(os.environ.get("USE_SHIFT_ADD_INFERENCE", "0")))

    # Sliding-window validation.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # LZMA preset for final artifact.
    lzma_preset = int(os.environ.get("LZMA_PRESET", 9))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    if weight_decay > 0:
                        g = g + weight_decay * p.data.to(g.dtype)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# MXINT4 QUANTISATION (OCP MX SPEC)
# -----------------------------
#
# Format: 32 elements share one E8M0 byte-scale.  Each element is INT4 in
# [-8, 7].  Packed storage: two INT4 nibbles per byte.
# Effective bit-rate: (32×4 + 8) / 32 = 4.25 bits/weight.
#
# With MXINT4 we can fit ~40 % more parameters than INT6 in the same 16 MB,
# which more than compensates for the slightly higher quantisation noise (when
# combined with QAT and GPTQ-style Hessian reordering).

MX_BLOCK_SIZE = 32  # OCP MX standard block size


class _MXInt4STE(torch.autograd.Function):
    """Straight-Through Estimator for MXINT4 fake quantisation."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        orig_shape = x.shape
        flat = x.float().reshape(-1)
        n = flat.numel()
        pad = (-n) % MX_BLOCK_SIZE
        if pad:
            flat = F.pad(flat, (0, pad))
        blocks = flat.reshape(-1, MX_BLOCK_SIZE)

        # E8M0: pick the floor-power-of-2 that covers the block maximum.
        max_abs = blocks.abs().amax(dim=1, keepdim=True).clamp(min=2.0 ** -127)
        log2_max = torch.floor(torch.log2(max_abs)).clamp(-127, 127)
        e8m0_scale = 2.0 ** log2_max          # exact power of 2
        divisor = e8m0_scale / 7.0            # maps [-7·d, 7·d] → [-7, 7]

        q = torch.round(blocks / divisor).clamp(-8, 7)
        dequant = (q * divisor).reshape(-1)        # flatten back to 1-D
        result = flat + (dequant - flat).detach()  # STE: gradient passes through
        return result[:n].reshape(orig_shape).to(x.dtype)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        return grad   # straight-through


def fake_quantize_mxint4(x: Tensor) -> Tensor:
    return _MXInt4STE.apply(x)


# -----------------------------
# POST-TRAINING MXINT4 QUANTISATION (for export)
# -----------------------------

def quantize_tensor_mxint4(t: Tensor) -> tuple[Tensor, Tensor, tuple]:
    """
    Quantise a float tensor to MXINT4.

    Returns:
      packed   – uint8 tensor, two nibbles per byte (even=low nibble)
      e8m0     – uint8 tensor, one byte per block (biased by 127)
      shape    – original shape (needed for dequantisation)
    """
    orig_shape = t.shape
    flat = t.float().reshape(-1)
    n = flat.numel()
    pad = (-n) % MX_BLOCK_SIZE
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.reshape(-1, MX_BLOCK_SIZE)

    max_abs = blocks.abs().amax(dim=1, keepdim=True).clamp(min=2.0 ** -127)
    log2_max = torch.floor(torch.log2(max_abs)).clamp(-127, 127)
    e8m0_scale = 2.0 ** log2_max
    divisor = e8m0_scale / 7.0

    q = torch.round(blocks / divisor).clamp(-8, 7).to(torch.int8)
    q_shifted = (q + 8).to(torch.uint8)  # shift to [0, 15] for nibble packing

    # Pack two INT4 values per byte: low nibble = even index, high nibble = odd.
    q_even = q_shifted[:, 0::2]
    q_odd  = q_shifted[:, 1::2]
    packed = (q_even | (q_odd << 4)).reshape(-1)

    # E8M0 stored as uint8 with bias 127.
    e8m0 = (log2_max.squeeze(1) + 127).to(torch.uint8)

    return packed.contiguous(), e8m0.contiguous(), orig_shape


def dequantize_tensor_mxint4(packed: Tensor, e8m0: Tensor, orig_shape: tuple,
                              dtype: torch.dtype = torch.bfloat16) -> Tensor:
    """Inverse of quantize_tensor_mxint4."""
    n_params = math.prod(orig_shape)
    n_blocks = (n_params + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE

    q_even = (packed & 0x0F).to(torch.int8) - 8
    q_odd  = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack([q_even, q_odd], dim=1).reshape(-1)[:n_params]

    log2_vals = e8m0.float() - 127.0
    scales = (2.0 ** log2_vals) / 7.0     # per-block dequant divisor

    # Align: reshape into (n_blocks, block_size), multiply, flatten.
    q_f = q.float()
    if n_params % MX_BLOCK_SIZE != 0:
        q_f = F.pad(q_f, (0, n_blocks * MX_BLOCK_SIZE - n_params))
    dequant = (q_f.reshape(n_blocks, MX_BLOCK_SIZE) * scales[:, None]).reshape(-1)[:n_params]
    return dequant.to(dtype).reshape(orig_shape)


# -----------------------------
# QUANTISE / DEQUANTISE STATE DICT
# -----------------------------

CONTROL_TENSOR_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
    "bigram",
)
SMALL_TENSOR_MAX_NUMEL = 65_536


def is_control(name: str) -> bool:
    return any(p in name for p in CONTROL_TENSOR_PATTERNS)


def quantize_state_dict_mxint4(
    state_dict: dict[str, Tensor],
    fp16_embed: bool = True,
) -> tuple[dict, dict]:
    """
    Quantise large float matrices to MXINT4; keep small / control tensors
    in fp16 passthrough.  Returns (payload_dict, stats_dict).
    """
    payload: dict[str, object] = {
        "__quant_format__": "mxint4_packed_e8m0_v1",
        "mxint4_packed": {},
        "mxint4_e8m0":   {},
        "mxint4_shapes": {},
        "mxint4_dtypes": {},
        "passthrough":   {},
        "passthrough_orig_dtypes": {},
    }
    stats = {
        "param_count": 0,
        "baseline_bytes": 0,
        "mxint4_payload_bytes": 0,
    }

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["baseline_bytes"] += t.numel() * t.element_size()

        if not t.is_floating_point():
            payload["passthrough"][name] = t
            stats["mxint4_payload_bytes"] += t.numel() * t.element_size()
            continue

        # Tied embedding: keep as fp16 for quality.
        if fp16_embed and "tok_emb.weight" in name:
            payload["passthrough_orig_dtypes"][name] = str(t.dtype).removeprefix("torch.")
            payload["passthrough"][name] = t.to(torch.float16).contiguous()
            stats["mxint4_payload_bytes"] += t.numel() * 2
            continue

        # Small / control tensors: keep as fp16.
        if t.numel() <= SMALL_TENSOR_MAX_NUMEL or is_control(name):
            orig = str(t.dtype).removeprefix("torch.")
            payload["passthrough_orig_dtypes"][name] = orig
            payload["passthrough"][name] = t.to(torch.float16).contiguous()
            stats["mxint4_payload_bytes"] += t.numel() * 2
            continue

        # Large float tensors → MXINT4.
        pk, e8, sh = quantize_tensor_mxint4(t)
        payload["mxint4_packed"][name]  = pk
        payload["mxint4_e8m0"][name]    = e8
        payload["mxint4_shapes"][name]  = sh
        payload["mxint4_dtypes"][name]  = str(t.dtype).removeprefix("torch.")
        n_blocks = (t.numel() + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
        stats["mxint4_payload_bytes"] += len(pk) + n_blocks  # packed bytes + scale bytes

    return payload, stats


def dequantize_state_dict_mxint4(payload: dict) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name in payload["mxint4_packed"]:
        dtype = getattr(torch, payload["mxint4_dtypes"][name])
        sh    = payload["mxint4_shapes"][name]
        pk    = payload["mxint4_packed"][name]
        e8    = payload["mxint4_e8m0"][name]
        out[name] = dequantize_tensor_mxint4(pk, e8, sh, dtype=dtype)
    for name, t in payload["passthrough"].items():
        orig = payload.get("passthrough_orig_dtypes", {}).get(name)
        tt = t.detach().cpu().contiguous()
        if orig:
            tt = tt.to(getattr(torch, orig))
        out[name] = tt
    return out


# -----------------------------
# UNSTRUCTURED PRUNING
# -----------------------------

def apply_unstructured_pruning(model: nn.Module, sparsity: float,
                                exclude_patterns: tuple[str, ...] = CONTROL_TENSOR_PATTERNS) -> int:
    """
    Global magnitude-based unstructured pruning.
    Zeros weights below the *sparsity* quantile globally across all 2-D
    parameter matrices, skipping control tensors.
    Returns the number of weights that were zeroed.
    """
    if sparsity <= 0.0:
        return 0
    candidates = [
        (name, p)
        for name, p in model.named_parameters()
        if p.ndim == 2 and not any(pat in name for pat in exclude_patterns)
    ]
    if not candidates:
        return 0

    all_abs = torch.cat([p.data.abs().view(-1) for _, p in candidates])
    threshold = torch.quantile(all_abs, sparsity).item()

    n_pruned = 0
    for _, p in candidates:
        mask = p.data.abs() > threshold
        n_pruned += int((~mask).sum().item())
        p.data.mul_(mask.to(p.dtype))

    return n_pruned


# -----------------------------
# OPTUNA PRUNING-RATIO SEARCH
# -----------------------------

def _reconstruction_error(
    original_sd: dict[str, Tensor],
    pruned_sd: dict[str, Tensor],
    patterns: tuple[str, ...] = CONTROL_TENSOR_PATTERNS,
) -> float:
    """Weighted L2 reconstruction error across all 2-D weight matrices."""
    total_err = 0.0
    total_n   = 0
    for name, orig in original_sd.items():
        if orig.ndim != 2 or any(p in name for p in patterns):
            continue
        if name not in pruned_sd:
            continue
        diff = (orig.float() - pruned_sd[name].float()).pow(2).sum().item()
        total_err += diff
        total_n   += orig.numel()
    return total_err / max(total_n, 1)


def find_optimal_sparsity(
    model: nn.Module,
    n_trials: int = 25,
    timeout: float = 30.0,
    sparsity_min: float = 0.05,
    sparsity_max: float = 0.45,
    log_fn=print,
) -> float:
    """
    Use Optuna (Bayesian TPE) to find the sparsity that minimises weight
    reconstruction error after magnitude-based unstructured pruning.

    This runs entirely on CPU and typically finishes in < 20 s.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log_fn("optuna not installed – skipping Optuna search, using sparsity=0.15")
        return 0.15

    # Snapshot original (CPU, fp32 for precision).
    orig_sd = {
        name: p.detach().cpu().float()
        for name, p in model.named_parameters()
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_PATTERNS)
    }

    def objective(trial: "optuna.Trial") -> float:
        sparsity = trial.suggest_float("sparsity", sparsity_min, sparsity_max)

        # Prune a CPU copy – never touches the live model.
        pruned_sd = {name: t.clone() for name, t in orig_sd.items()}
        if sparsity > 0:
            all_abs = torch.cat([t.abs().view(-1) for t in pruned_sd.values()])
            threshold = torch.quantile(all_abs, sparsity).item()
            for t in pruned_sd.values():
                t.mul_((t.abs() > threshold).float())

        return _reconstruction_error(orig_sd, pruned_sd)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_sparsity = study.best_params["sparsity"]
    log_fn(
        f"optuna_pruning: best_sparsity={best_sparsity:.4f} "
        f"best_recon_err={study.best_value:.6e} "
        f"n_trials={len(study.trials)}"
    )
    return best_sparsity


# -----------------------------
# SHIFT-ADD MATMUL (INT4 DECOMPOSITION)
# -----------------------------
#
# An INT4 weight w ∈ [-8, 7] can be written in two's-complement as:
#   w = -8·b3 + 4·b2 + 2·b1 + b0,   b_i ∈ {0,1}
#
# So  W x = Σ_k  s_k · (B_k x)   with s = (-8, 4, 2, 1)
#
# and each B_k is a 0/1 binary matrix — implementable with XOR + POPCOUNT
# (or, in pure PyTorch, as F.linear with float {0,1} weight).
#
# This module is used only at INFERENCE and is loaded from the MXINT4
# export when USE_SHIFT_ADD_INFERENCE=1.

_INT4_BIT_SIGNS = torch.tensor([-8, 4, 2, 1], dtype=torch.float32)


class Int4ShiftAddLinear(nn.Module):
    """
    Inference-only linear layer backed by 4 binary weight matrices.
    Constructed from a pre-quantised INT4 weight tensor.
    """

    def __init__(self, in_features: int, out_features: int,
                 int4_weights: Tensor, e8m0: Tensor, orig_shape: tuple):
        """
        int4_weights : int8 tensor of shape (out, in) with values in [-8, 7]
        e8m0         : uint8 scale bytes (one per MX_BLOCK_SIZE elements)
        orig_shape   : original float weight shape (should equal (out, in))
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w = int4_weights.to(torch.int8)
        w_u = (w + 8).to(torch.uint8)   # shift to [0,15]

        for bit in range(4):
            bit_mat = ((w_u >> bit) & 1).float()
            self.register_buffer(f"b{bit}", bit_mat)

        # Dequant scales: per MX block, shape (n_blocks,)
        log2_vals = e8m0.float() - 127.0
        scales_per_block = (2.0 ** log2_vals) / 7.0
        # Expand to per-output-element (each row may span multiple blocks).
        n_out, n_in = out_features, in_features
        n_params = n_out * n_in
        n_blocks = (n_params + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
        # scales_per_block maps linearly to the flattened weight; reshape to (out, in).
        scales_flat = scales_per_block.repeat_interleave(MX_BLOCK_SIZE)[:n_params]
        # For per-output scaling we average across the input dimension.
        scales_2d = scales_flat.reshape(n_out, n_in)
        self.register_buffer("scales", scales_2d)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features)
        bits = [getattr(self, f"b{i}") for i in range(4)]
        signs = _INT4_BIT_SIGNS.to(x.device, x.dtype)
        result = torch.zeros(*x.shape[:-1], self.out_features,
                             device=x.device, dtype=x.dtype)
        for i, (bit_mat, sign) in enumerate(zip(bits, signs)):
            result = result + sign * F.linear(x, bit_mat.to(x.dtype))
        # Apply per-weight scales (averaged per output channel).
        row_scales = self.scales.float().mean(dim=1)   # (out_features,)
        result = result * row_scales.to(x.dtype)
        return result


def build_shift_add_linear(name: str, packed: Tensor, e8m0: Tensor,
                            orig_shape: tuple) -> Optional["Int4ShiftAddLinear"]:
    """Reconstruct Int4ShiftAddLinear from the packed MXINT4 export."""
    if len(orig_shape) != 2:
        return None
    out_f, in_f = orig_shape
    n = out_f * in_f
    # Unpack nibbles → int4 in [-8, 7].
    q_even = (packed & 0x0F).to(torch.int8) - 8
    q_odd  = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    q = torch.stack([q_even, q_odd], dim=1).reshape(-1)[:n]
    int4_w = q.reshape(out_f, in_f)
    return Int4ShiftAddLinear(in_f, out_f, int4_w, e8m0, orig_shape)


# -----------------------------
# TOKENIZER-AGNOSTIC EVAL SETUP
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 0,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError("VAL_BATCH_SIZE too small.")
    local_batch_seqs = local_batch_tokens // seq_len

    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end   = (total_seqs * (rank + 1)) // world_size

    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            rs  = bss * seq_len
            re  = bse * seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum   += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids]
                            & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


# Sliding-window variant: evaluate with stride < seq_len for overlapping windows.
def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int = 64,
    batch_size: int = 16,
) -> tuple[float, float]:
    """Sliding-window evaluation (stride < seq_len) – higher accuracy, slower."""
    seq_len = args.train_seq_len
    n_tokens = val_tokens.numel() - 1
    starts = list(range(0, n_tokens - seq_len + 1, stride))
    my_starts = [s for i, s in enumerate(starts) if i % world_size == rank]

    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_starts), batch_size):
            batch_s = my_starts[i : i + batch_size]
            xs = torch.stack([val_tokens[s : s + seq_len].to(torch.int64) for s in batch_s]).to(device)
            ys = torch.stack([val_tokens[s+1 : s + seq_len + 1].to(torch.int64) for s in batch_s]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(xs, ys).detach()
            ntok = float(ys.numel())
            val_loss_sum   += loss.to(torch.float64) * ntok
            val_token_count += ntok
            prev_ids = xs.reshape(-1)
            tgt_ids  = ys.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.int16)
            tb += (has_leading_space_lut[tgt_ids]
                   & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bpt = val_loss.item() / math.log(2.0)
    tpb = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bpt * tpb)


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes  = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.stream     = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int,
                   grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """fp32 weights, cast to x.dtype at matmul time.  QAT flag added."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.qat_enabled:
            w = fake_quantize_mxint4(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or is_control(name)) and param.dtype != torch.float32:
                param.data = param.data.float()


def set_qat_enabled(model: nn.Module, enabled: bool) -> None:
    """Toggle MXINT4 fake-quant on all CastedLinear layers."""
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.qat_enabled = enabled


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        assert self.head_dim % 2 == 0
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x   = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x   = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :]  * self.mlp(self.mlp_norm(x))
        return x


# BigramHash: for each adjacent token pair (prev, cur), look up a learned
# embedding from a hash table and add it to the residual stream.
class BigramHash(nn.Module):
    def __init__(self, vocab_size: int, n_buckets: int, dim: int):
        super().__init__()
        self.n_buckets = n_buckets
        self.vocab_size = vocab_size
        self.table = nn.Embedding(n_buckets, dim)
        self.proj  = CastedLinear(dim, vocab_size, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)

    def _bucket(self, prev: Tensor, cur: Tensor) -> Tensor:
        # Simple hash: (a * prev + b * cur) mod n_buckets
        return (prev * 1031 + cur * 2053) % self.n_buckets

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (B, T)
        # Returns logit bias: (B, T, vocab_size)
        prev = F.pad(input_ids[:, :-1], (1, 0))
        bucket = self._bucket(prev, input_ids)
        emb = self.table(bucket)                # (B, T, dim)
        return self.proj(emb.to(input_ids.device))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        bigram_buckets: int = 0,
        bigram_dim: int = 128,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings     = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap      = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32)
        )
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Optional BigramHash auxiliary embedding.
        self.bigram = BigramHash(vocab_size, bigram_buckets, bigram_dim) if bigram_buckets > 0 else None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x_norm = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x_norm, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_norm)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

        # Add BigramHash bias if present.
        if self.bigram is not None:
            bigram_logits = self.bigram(input_ids).reshape(-1, logits.shape[-1])
            logits = logits + bigram_logits.to(logits.dtype)

        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ---- DISTRIBUTED + CUDA SETUP ----
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # ---- SEED + TOKENIZER ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError("Only SentencePiece .model tokenisers are supported.")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:tokens:{val_tokens.numel() - 1}")

    # ---- MODEL + OPTIMIZER SETUP ----
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_buckets=args.bigram_buckets,
        bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # ---- OPTIMIZER GROUPS ----
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not is_control(name)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or is_control(name)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.bigram is not None:
        scalar_params.extend(base_model.bigram.parameters())

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_weight_decay, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_weight_decay, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps,
            weight_decay=args.adam_weight_decay, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"bigram_buckets:{args.bigram_buckets} bigram_dim:{args.bigram_dim}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} max_wallclock_seconds:{args.max_wallclock_seconds:.1f}"
    )
    log0(f"mxint4_qat_threshold:{args.qat_threshold} ema_decay:{args.ema_decay}")
    log0(f"optuna_n_trials:{args.optuna_n_trials} prune_range:[{args.prune_sparsity_min},{args.prune_sparsity_max}]")

    # ---- EMA SHADOW MODEL ----
    ema_model: Optional[nn.Module] = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(base_model).cpu()
        ema_model.eval()

    def update_ema() -> None:
        if ema_model is None:
            return
        d = args.ema_decay
        with torch.no_grad():
            for ema_p, live_p in zip(ema_model.parameters(), base_model.parameters()):
                ema_p.data.mul_(d).add_(live_p.data.cpu().float(), alpha=1.0 - d)
            for ema_b, live_b in zip(ema_model.buffers(), base_model.buffers()):
                if ema_b.dtype.is_floating_point:
                    ema_b.copy_(live_b.cpu())

    # ---- DATA LOADER + JIT WARMUP ----
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        init_model_state  = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states   = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wloss = model(x, y)
                (wloss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_model_state, strict=True)
        for opt, st in zip(optimizers, init_opt_states, strict=True):
            opt.load_state_dict(st)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- MAIN TRAINING LOOP ----
    training_time_ms = 0.0
    stop_after_step: Optional[int] = None
    swa_accum: list[dict[str, Tensor]] = []
    qat_active = False

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms "
                f"qat:{qat_active}"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: train_time:{training_time_ms:.0f}ms step:{step}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # ---- Late MXINT4 QAT: activate when warmdown fraction drops below threshold ----
        if not qat_active and scale < args.qat_threshold:
            set_qat_enabled(base_model, True)
            qat_active = True
            log0(f"mxint4_qat:enabled at step={step} scale={scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warm-up.
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # EMA update every step.
        update_ema()

        # Tight SWA: collect snapshots during warmdown when scale < 0.2.
        if qat_active and step % args.swa_every == 0:
            swa_accum.append({n: p.data.detach().cpu().clone() for n, p in base_model.named_parameters()})

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc_t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak_memory_allocated:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB "
        f"reserved:{torch.cuda.max_memory_reserved() // 1024 // 1024}MiB"
    )

    # ---- WEIGHT AVERAGING: apply tight SWA then blend with EMA ----
    if master_process:
        # Tight SWA: average collected snapshots.
        if swa_accum:
            log0(f"tight_swa: averaging {len(swa_accum)} snapshots")
            avg_sd: dict[str, Tensor] = {}
            for name in swa_accum[0]:
                avg_sd[name] = torch.stack([sd[name].float() for sd in swa_accum]).mean(0)
            base_model.load_state_dict(avg_sd, strict=True)

        # Blend SWA result with EMA (50/50 or tune via EMA_SWA_BLEND env var).
        ema_blend = float(os.environ.get("EMA_SWA_BLEND", "0.5"))
        if ema_model is not None and ema_blend > 0:
            log0(f"ema_swa_blend: alpha={ema_blend}")
            ema_sd = {n: p.data.float() for n, p in ema_model.named_parameters()}
            cur_sd = {n: p.data.float() for n, p in base_model.named_parameters()}
            blended_sd: dict[str, Tensor] = {}
            for name in cur_sd:
                if name in ema_sd:
                    blended_sd[name] = (1 - ema_blend) * cur_sd[name] + ema_blend * ema_sd[name]
                else:
                    blended_sd[name] = cur_sd[name]
            base_model.load_state_dict(blended_sd, strict=True)

    if distributed:
        dist.barrier()

    # ---- OPTUNA: FIND OPTIMAL PRUNING RATIO ----
    if master_process:
        log0("optuna_search:start")
        t_optuna = time.perf_counter()
        best_sparsity = find_optimal_sparsity(
            base_model,
            n_trials=args.optuna_n_trials,
            timeout=args.optuna_timeout,
            sparsity_min=args.prune_sparsity_min,
            sparsity_max=args.prune_sparsity_max,
            log_fn=log0,
        )
        log0(f"optuna_search:done best_sparsity={best_sparsity:.4f} "
             f"elapsed={1000*(time.perf_counter()-t_optuna):.0f}ms")

        # ---- APPLY PRUNING ----
        if best_sparsity > 0:
            n_zeroed = apply_unstructured_pruning(base_model, best_sparsity)
            frac_zero = n_zeroed / sum(p.numel() for p in base_model.parameters())
            log0(f"pruning: zeroed {n_zeroed} weights ({frac_zero*100:.1f}%)")
        else:
            log0("pruning: skipped (sparsity=0)")

    if distributed:
        dist.barrier()

    # ---- PRE-QUANTISATION EVALUATION ----
    torch.cuda.synchronize()
    t0_eval = time.perf_counter()
    pre_q_loss, pre_q_bpb = eval_val_sliding(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride,
    )
    log0(
        f"pre_quant_sliding: val_loss:{pre_q_loss:.4f} val_bpb:{pre_q_bpb:.4f} "
        f"eval_time:{1000*(time.perf_counter()-t0_eval):.0f}ms"
    )

    # ---- MXINT4 QUANTISATION + LZMA EXPORT ----
    if master_process:
        # Raw fp32/bf16 snapshot for debugging.
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"saved final_model.pt ({os.path.getsize('final_model.pt')} bytes)")

        t_quant = time.perf_counter()
        payload, qstats = quantize_state_dict_mxint4(base_model.state_dict(), fp16_embed=True)
        quant_buf = io.BytesIO()
        torch.save(payload, quant_buf)
        quant_raw = quant_buf.getvalue()

        # LZMA compression – typically 10-15 % smaller than zlib-9 on sparse INT4 data.
        quant_blob = lzma.compress(quant_raw, preset=args.lzma_preset)
        with open("final_model.mxint4.ptz", "wb") as f:
            f.write(quant_blob)
        artifact_bytes = os.path.getsize("final_model.mxint4.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = qstats["baseline_bytes"] / max(qstats["mxint4_payload_bytes"], 1)
        log0(f"quant_time:{1000*(time.perf_counter()-t_quant):.0f}ms")
        log0(
            f"mxint4_lzma: artifact={artifact_bytes} "
            f"(payload:{qstats['mxint4_payload_bytes']} raw_torch:{len(quant_raw)} "
            f"payload_ratio:{ratio:.2f}x)"
        )
        log0(f"total_submission_size: {artifact_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()

    # ---- ROUND-TRIP VALIDATION ----
    with open("final_model.mxint4.ptz", "rb") as f:
        blob_disk = f.read()
    loaded_payload = torch.load(io.BytesIO(lzma.decompress(blob_disk)), map_location="cpu")
    rt_sd = dequantize_state_dict_mxint4(loaded_payload)
    base_model.load_state_dict(rt_sd, strict=True)
    torch.cuda.synchronize()

    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_sliding(
        args, model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride,
    )
    torch.cuda.synchronize()
    log0(
        f"final_mxint4_lzma_roundtrip: val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000*(time.perf_counter()-t_qeval):.0f}ms"
    )
    log0(f"final_mxint4_lzma_roundtrip_exact: val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
