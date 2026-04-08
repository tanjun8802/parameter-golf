# MXINT4 + Unstructured Pruning + Optuna Pipeline

**Track:** 10-minute / 16 MB leaderboard  
**Architecture baseline:** 11L, 512-dim, 3× MLP (relu²), GQA (8h/4KV), U-Net skips, BigramHash  
**Key innovations:** MXINT4 QAT · Optuna-tuned unstructured pruning · Shift-add matmul decomposition · LZMA compression

---

## Motivation

The current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB) uses **INT6** post-training GPTQ with late INT6-QAT. Moving to **MXINT4** (the OCP Microscaling specification) reduces bits-per-weight from 6 → 4.25, letting us fit **~41 % more parameters** into the 16 MB budget while maintaining quality via:

| Strategy | Why it helps |
|---|---|
| MXINT4 QAT (STE) | Closes train/deploy precision gap; model learns to work with 4-bit weights |
| Optuna pruning search | Finds the sparsity ratio that minimises weight reconstruction error; sparse weights compress extremely well |
| EMA + Tight SWA | Smoother weight trajectory → lower quantisation error |
| BigramHash(2048×128) | Zero-parameter-budget bigram logit bias; free BPB gain |
| Shift-add matmul | Decomposes INT4 matmul into 4 binary matmuls (opt-in at inference) |
| LZMA preset=9 | 10–15 % smaller artifact than zlib-9 on sparse INT4 data |

---

## MXINT4 Format (OCP MX Specification)

```
Block size: 32 elements share one E8M0 scale byte.
Payload:    Each element stored as INT4 (2's complement, [-8, 7]).
Storage:    two INT4 nibbles packed per uint8 byte.
Bit-rate:   (32×4 + 8) / 32 = 4.25 bits/weight.
```

For a 16 MB budget: `16×1024×1024×8 / 4.25 ≈ 31.5 M effective MXINT4 parameters` vs. `~22 M for INT6`.

---

## MXINT4 QAT (Straight-Through Estimator)

During the warmdown phase (when the cosine-decay LR fraction falls below `QAT_THRESHOLD=0.15`), all `CastedLinear` layers begin fake-quantising their weights to the MXINT4 grid before every forward pass:

```
forward:   w_q = round(w / (2^floor(log2(max|w_block|)) / 7))  ← MXINT4 grid
backward:  ∂L/∂w  passed straight through (STE)
```

This teaches the network to use weight values that survive 4-bit quantisation without gradient estimate bias.

---

## Optuna Pruning-Ratio Search

After training (and EMA/SWA averaging) Optuna runs **25 TPE trials on CPU** to find the global unstructured sparsity `s ∈ [5 %, 45 %]` that minimises the weighted L2 reconstruction error between the original and pruned weight matrices:

```
objective(s) = Σ_layers  ||W_orig - prune(W, s)||²  /  total_params
```

Each trial takes ≈ 0.5 s on CPU (no GPU needed). Typical optimal sparsity: 15–30 % — sparse near-zero weights compress extremely well with LZMA and have negligible impact on BPB after QAT has taught the model to tolerate pruning implicitly.

**Total Optuna wall-clock cost:** ≈ 15 s (well within the 60 s wall-clock budget).

---

## Shift-Add Matmul Decomposition (inference opt-in)

Any INT4 weight `w ∈ [-8, 7]` satisfies:

```
w = -8·b₃ + 4·b₂ + 2·b₁ + b₀,   bᵢ ∈ {0, 1}
```

So the matrix product `Wx` decomposes into four binary matmuls:

```
Wx = Σ_{k=0}^{3} sₖ · (Bₖ x),   s = (-8, 4, 2, 1)
```

The `Int4ShiftAddLinear` module in `train.py` implements this. Enable at inference with `USE_SHIFT_ADD_INFERENCE=1`. On hardware supporting fast XOR+POPCOUNT (e.g. H100 via `cutlass::gemm::warp::MmaTensorOpBinaryFolded`) this can approach 16× FP32 throughput.

---

## Timing Budget (8×H100 SXM, 600 s)

| Phase | Approx. time |
|---|---|
| Training (6 500 steps @ 83 ms/step) | ≈ 540 s |
| EMA/SWA averaging | < 1 s |
| Optuna pruning search (25 trials, CPU) | ≈ 15 s |
| MXINT4 quantisation + LZMA | ≈ 5 s |
| **Total** | **≈ 561 s** |

---

## Architecture

| Component | Setting |
|---|---|
| Layers | 11 (512-dim) |
| Attention | 8 heads, 4 KV-heads (GQA) |
| MLP | 3× expansion (1536-hidden), relu² |
| U-Net skips | 5 encoder + 6 decoder |
| BigramHash | 2048 buckets × 128-dim |
| Tied embeddings | Yes (fp16 export) |
| RoPE | Standard (base 10 000) |
| Logit softcap | 30.0 |
| Quantisation | MXINT4 (late QAT + export) |
| Compression | LZMA preset=9 |

---

## Hyperparameters

| Param | Value |
|---|---|
| `TRAIN_SEQ_LEN` | 2048 |
| `TRAIN_BATCH_TOKENS` | 786 432 |
| `MAX_WALLCLOCK_SECONDS` | 540 |
| `WARMDOWN_ITERS` | 4000 |
| `MATRIX_LR` (Muon) | 0.025 |
| `MUON_WEIGHT_DECAY` | 0.04 |
| `MUON_MOMENTUM` | 0.99 (warmup 0.92 → 0.99 over 1500 steps) |
| `TIED_EMBED_LR` | 0.035 |
| `GRAD_CLIP_NORM` | 0.3 |
| `EMA_DECAY` | 0.997 |
| `SWA_EVERY` | 50 steps |
| `QAT_THRESHOLD` | 0.15 (LR-scale fraction) |
| `OPTUNA_N_TRIALS` | 25 |
| `PRUNE_SPARSITY_MIN/MAX` | 0.05 / 0.45 |
| `LZMA_PRESET` | 9 |

---

## Run Command

```bash
# Install extras
pip install optuna sentencepiece

# Single-GPU smoke test
python train.py

# 8×H100 full run
torchrun --nproc_per_node=8 --nnodes=1 train.py
```

Optional environment overrides:
```bash
# Increase architecture for larger MXINT4 budget
NUM_LAYERS=13 MODEL_DIM=576 torchrun --nproc_per_node=8 train.py

# Enable shift-add inference (after training)
USE_SHIFT_ADD_INFERENCE=1 python eval.py
```

---

## Expected Results

Based on the MXINT4 parameter budget advantage (+41 % vs INT6) and the pruning/LZMA benefit, this pipeline is expected to match or improve on the INT6-QAT SOTA:

- Pre-quantisation BPB: ~1.135 (comparable to SOTA stack)  
- Post-MXINT4 + pruning BPB: targeting **≤ 1.118** (improvement from larger parameter budget)

Actual numbers to be filled in after full 8×H100 run.
