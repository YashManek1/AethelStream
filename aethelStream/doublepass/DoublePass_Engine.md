# MODULE 5 — DOUBLE-PASS BACKWARD ENGINE (Refactored)

**Status:** Refactor of the original M5 plan. This document supersedes the original
"Algorithms 1–7" section. It keeps the project's brand name ("Double-Pass") and every
frozen seam, but corrects a throughput-fatal recomputation pattern, removes an
approximation that was not actually required, and adds the throughput machinery the
original plan was missing.

**Owner:** 1 developer (deep autograd internals + Rust/CUDA FFI)
**Language:** CUDA C++ (recompute + fused kernels) · Rust (orchestration crate `doublepass`) · Python (autograd-parity reference + PyO3 surface for M7)
**Depends on:** M1 (layer format), M2 RamFlow (pinned memory / NVMe — *via M3 only*), M3 FlowCast (the only I/O surface M5 touches), M4 (GaLore optimizer), M6 (LoRA).
**Consumes:** M9 `TrainingPlan` (frozen). **Checkpointed by:** M10 (resume). **Wrapped by:** M7 (Python loop).

> **Why this module is load-bearing.** Every other module is plumbing or policy. M5 is
> the only place where *the math actually happens*. If the gradient M5 produces does not
> match a reference autograd gradient to tolerance, the entire framework is invalid no
> matter how fast or memory-clean the rest is. So this module is judged on two axes at
> once — **numerical correctness** (parity with PyTorch) and **tokens/sec** — and the
> design below is organized around keeping both true simultaneously.

---

## 0. What changed vs. the original plan (changelog)

| # | Original plan | Problem | Refactor |
|---|---|---|---|
| C1 | **Algorithm 2** recomputes from the nearest checkpoint **for every layer independently** (inner loop `for j in checkpoint_layer..=i`). | Quadratic recompute *inside each checkpoint block*: a block of size `k` does `0+1+…+(k-1) = k(k-1)/2` layer-recomputes for `k` backward layers. For `k=8` that is **~3.5× extra forward compute**, and in the weight-streaming regime it is *also* ~3.5× extra weight **re-streaming** — the dominant cost. The existing M2/M3 bottleneck note ("≈7/8 of forward computation is redundant") is describing exactly this. | **Segment-wise recompute (§5 A2).** Recompute each checkpoint segment **once** (forward through the segment, retaining its `k` activations in VRAM/RAM), then walk backward through the segment reusing them. Recompute drops from `O(k(k-1)/2)` to `O(k)` per block — i.e. exactly **1× extra forward pass total**, the standard gradient-checkpointing cost. This is the single biggest throughput fix in the refactor. |
| C2 | Loop order is implicitly **sample-major** (one micro-batch traverses all layers, then the next). | In a weight-streaming system this re-streams every layer **once per micro-batch**: weight bytes scale as `O(G · L)`. Weight streaming is the bottleneck, so this throws away the main lever. | **Layer-major amortized execution (§5 A1, §3).** Stream each layer's weights **once per step** and push all `G` grad-accumulation micro-batches through it before evicting. Weight bytes scale as `O(L)`, independent of `G`. tokens/sec then scales **linearly in `G`** in the I/O-bound regime (FlexGen-for-training). |
| C3 | **Algorithm 3** LOMO hook applies the optimizer update *during* the per-micro-batch backward and discards the gradient. | This is incompatible with gradient accumulation across micro-batches (you cannot accumulate if you already applied), and it forces LOMO's **grouped/local gradient clipping** approximation because no global norm is available before the first update. | **Low-Rank Deferred-Apply LOMO (§5 A3, A6).** The hook still never stores a full gradient — it projects to GaLore low-rank space immediately — but it **accumulates the tiny low-rank gradient** and defers the Adam apply. Because GaLore/LoRA accumulators are small (tens of MB for the whole model), we can retain *all* of them through the backward and then apply, which (a) makes grad-accum work and (b) gives **exact global-norm clipping for free**, removing the approximation entirely on the GaLore/LoRA path. |
| C4 | **Algorithm 7** "Numerical Parity Guard" detects drift vs. PyTorch and *fixes it by resetting GaLore projections via SVD*. | Category error. GaLore projections are an intentional low-rank approximation; resetting them does **not** restore parity with full-rank autograd, it just perturbs the optimizer trajectory. It conflates a *diagnostic* with a *remedy*. | **Decoupled (§5 A7).** (i) **Projection refresh** (periodic SVD) is standard GaLore hygiene and belongs to **M4**, on a fixed cadence. (ii) The **parity guard** becomes a pure *diagnostic*: it compares one micro-step against an in-memory full-precision reference and, on drift, **escalates recompute precision** for the offending layer and/or **halts for a checkpoint** — it never silently mutates the optimizer to mask a bug. |
| C5 | Selective **INT8 recompute** of "non-critical" layers proposed as a speed win. | In the **I/O-bound** regime recompute is *free* (hidden behind weight streaming — see §3), so lowering its precision buys almost no speed but injects activation drift that propagates into the gradient and breaks parity. | **Recompute at forward precision by default (§5 A2/A9).** Reduced-precision recompute is offered *only* for segments the cost model proves are compute-bound, and only under the parity guard. Honest default: correctness first. |
| C6 | No mention of **dropout / RNG** during recompute. | Recompute that uses a *different* dropout mask than the forward computes a gradient at the wrong activation → silent parity failure for any model with dropout (most fine-tunes). | **Deterministic recompute (§5 A10).** Capture per-`(layer, micro-batch)` RNG state at forward; restore it at recompute. Mandatory; it is also what makes M10's bit-reproducible resume possible. |
| C7 | Full logits implied to be materialized then reduced; "only the final scalar stored" stated but not specified. | For vocab `V≈128k`, the `[G·S·V]` logit tensor is enormous and the LM-head weight is the single biggest matrix in the model. | **Streaming Cut-Cross-Entropy + tiled LM-head (§5 A8).** Loss and its gradient are computed in vocab chunks; the full logit tensor is never materialized; the LM-head weight is streamed tile-wise. Peak is `O(chunk)`, not `O(V)`. |
| C8 | Checkpoint frequency "determined by Module 3's warm-up profiler." | In the current architecture M3 only *measures*; the inverse-config decision is **M9 ElasticScale's** job, and `checkpoint_freq` is a field in the frozen `TrainingPlan`. | M5 **reads `checkpoint_freq`, `micro_batch`, `grad_accum`, `precision_schedule`, `optimizer_rank`, `tier` from the `TrainingPlan`** (§7). M3's profiler feeds M9; M5 obeys the plan. |

### v2 advancements (state-of-the-art elevation — added after auditing the real M2/M3 code)

The eight items above make M5 *correct and throughput-sound*. The three below make it
*state-of-the-art* against the 2020–2025 rematerialization / memory-efficient-training
literature, and were added after reading the shipped FlowCast/RamFlow reports (they change
what M5 can lean on). C9 is the headline research contribution.

| # | What v1 did | Why it's below SOTA | v2 advancement |
|---|---|---|---|
| **C9** | **A2 recomputes whole layers at *layer* granularity**, and the **recompute-vs-offload choice (N5/HAM) is a per-segment greedy heuristic** with `k*` picked from bandwidth alone. | This is the *baseline* checkpointing problem (Chen 2016). The optimal-rematerialization line proved heuristics are strictly dominated: **Checkmate** (MLSys'20, ILP) and **POET** (ICML'22, joint remat **+ paging** MILP) show greedy "page-until-infeasible-then-recompute" (Capuchin) wastes up to ~140% overhead vs. a *jointly optimized* schedule. None of that prior work modeled the case where **the weights themselves are the dominant paged object** — which is exactly AethelStream. | **Streaming-Aware Rematerialization–Paging planner (§4 N1′, §5 A2′/A9′).** Generalize Checkmate/POET to the weight-streaming regime: per segment, choose `{RECOMPUTE \| RETAIN_VRAM \| PAGE→compressed-RAM(LZ4) \| PAGE→NVMe}` to **maximize tokens/sec** under a *shared* I/O budget (weight prefetch and activation paging contend for the same NVMe/PCIe channel) and joint VRAM+RAM caps. Because a transformer is a regular linear chain, this collapses from a general MILP to a **DP over block-types** solvable in ms. The solve lives in **M9** (it already owns the inverse-config solve and emits the frozen plan); M5 **executes** the emitted per-segment schedule. M5 keeps the v1 greedy heuristic only as a fallback when M9 emits no schedule. |
| **C10** | A2 recomputes the **entire** layer (all sub-ops) during backward. | **Selective activation recomputation** (Korthikanti et al., MLSys'23) shows most layer FLOPs need *not* be recomputed: recompute only the **memory-heavy / compute-cheap** region (the attention softmax+dropout path) and **retain** the small-but-expensive linear outputs. ~5× activation-memory cut at ~90% less recompute overhead. In AethelStream this **lowers the compute knee `G*`**, so tokens/sec scales linearly with `G` for *longer* before compute saturates. | **Selective-region recompute (§5 A2′).** The per-segment plan records, per op-type, a `recompute \| retain` bit. Default policy retains the cheap-to-store activations (LayerNorm stats, the residual stream tap, linear outputs that are small relative to their FLOPs) and recomputes only the attention interior. Falls back to full-layer recompute under hard VRAM pressure. |
| **C11** | N3/A6 claims **"exact global-norm clipping for free"** because the GaLore projection is orthonormal (Frobenius-norm-preserving). | True for **SVD** projection (GaLore). But the current SOTA optimizers M4 may adopt — **APOLLO** (MLSys'25), **Q-GaLore** (INT4 projection), **Fira** — use **random projection** (cheaper, no SVD). Random projection preserves norm only *approximately* (Johnson–Lindenstrauss), so the "exact for free" claim is **false** under those optimizers. | **Projection-agnostic clipping (§5 A6′).** The hook treats the projection as an opaque `Project` trait from M4. The norm guarantee is stated **conditionally**: *exact* under orthonormal projection (SVD GaLore); *approximate with a JL concentration bound* under random projection (APOLLO/Q-GaLore). When M4 reports a non-orthonormal projector, M5 either (a) clips on the JL-corrected estimate, or (b) for layers flagged clip-critical, retains a true Frobenius accumulator (one scalar per layer) computed from the *pre-projection* grad in the hook — still O(1) memory, restoring exactness. |

Everything else (LOMO hook concept, MeSP `h=xA` recompute, the dynamic-loss-scaling
fallback, the six original tests) is **retained** and folded into the structure below.

### Seam corrections from the shipped M2/M3 code (not new research — required for M5 to bind correctly)

Reading `FLOWCAST_REPORT.md`, `RAMFLOW_REPORT.md`, and `flowcast/src/{lib,ready}.rs` changed
four things M5 must do. These are **contract facts**, not options:

1. **Honor the `copy_event` double-buffer contract.** `ReadyLayer` carries `copy_event: Option<CudaEvent>` (M3-New-4, `cuda-double-buffer`). When it is `Some`, M5 **must** call `cuda_stream_wait_event(compute_stream, copy_event)` before dispatching any kernel on `slab_device_ptrs` — the RAM→VRAM DMA into the double-buffer slot is in flight. Skipping this is a silent race. (Was unspecified in v1.)
2. **Do not reinvent residency or recompute ordering — drive M3's existing machinery.** FlowCast already ships an **A1 `Recompute { window, cursor }` FSM arm**, an **A8 `PriorityScheduler` (min-heap) for recomputation ordering**, and an **A5 `HotSet` residency cache with evict-on-pressure**. M5's "keep_resident" set and segment-recompute prefetch sequence are expressed as calls into these (`on_layer_start(.., Direction)` with the recompute window), **not** a parallel scheduler. M5 never touches RamFlow directly.
3. **Offload targets already exist.** C9's `PAGE→compressed-RAM` uses M2's shipped **LZ4 eviction cache** (`lz4-cache`, byte-identical round-trip, ~4 GB/s decompress, zero SSD wear); `PAGE→NVMe` uses M3 write-back. M5's planner emits *which tier*; it does not implement a tier.
4. **Write-back contends with prefetch through a real arbiter.** M3 ships an **A4/A9 delayed write-back with gradient-skip** and a **duplex token bucket** (read/write bandwidth split). M5's deferred gradient/checkpoint writes go through `on_weights_updated`; under write-token exhaustion they are *deferred, not dropped*. M5 must not assume a write completes synchronously.

---

## 1. Mission and position in AethelStream

M5 implements one training **step** end-to-end on a model whose layers do not fit in VRAM:

```
TrainingPlan (M9) ─┐
                   ▼
  M7 (Python loop) ── calls ──► M5 doublepass.step(batch)
                                   │
                                   ├── asks M3 FlowCast for layer weights (the ONLY I/O surface)
                                   │       FlowCast drives M2 RamFlow (pinned buffers, io_uring, write-back)
                                   ├── calls M4 to project/accumulate/apply optimizer updates (GaLore)
                                   ├── calls M6 for LoRA adapter forward/backward (if LoRA mode)
                                   ├── stores sparse checkpoints in M2 pinned buffers (via FFI)
                                   └── at step end, hands a consistent snapshot to M10 (resume)
```

M5 owns: the forward/backward **schedule**, the **recompute** policy, **checkpoint**
placement, the **hook** that fuses projection+accumulation, **clipping**, **loss
scaling**, the **loss** kernel, and the **parity** diagnostic. M5 owns *no* I/O mechanism
and *no* allocation — those belong to M2/M3.

---

## 2. The core problem, precisely

Classic gradient checkpointing assumes **weights are resident** and only **recompute
FLOPs** cost anything; it minimizes recompute subject to an activation-memory budget.
AethelStream breaks that assumption: **weights are streamed from NVMe**, so a recompute
of layer *j* is not just FLOPs — it may be a **re-read of `W_j` bytes from disk**, and
NVMe bandwidth is the system bottleneck. The right objective is therefore different:

> **Minimize re-streamed weight bytes (and recompute FLOPs only where they are not hidden
> behind I/O), subject to the VRAM budget for the active layer + the live activation
> wavefront + the resident recompute segment.**

This single reframing drives the entire refactor:

1. Recompute **redundancy** is expensive twice over (FLOPs *and* re-streamed bytes) → eliminate it (segment recompute, C1).
2. Weight streaming should be **amortized over as many tokens as possible** → layer-major + grad-accum (C2).
3. When the step is I/O-bound, the GPU is **idle during streaming**, so recompute in that shadow is *free* → prefer recompute over activation-offload, and keep it full-precision (C5, §3).
4. The gradient must still be **bit-faithful** to a reference → deterministic recompute, exact clipping, parity diagnostic (C3, C4, C6).

---

## 3. The throughput model (tokens/sec) and the operating point

### 3.1 Per-layer roofline (extends LoHan's `T_iter = max(...)`)

For layer *i* with weight bytes `W_i` (precision-dependent), let
`B_ssd`, `B_pcie` be effective NVMe-read and RAM→VRAM bandwidths, `F` the realized GPU
throughput (FLOP/s = `η · FLOP_peak`), and `T = G · s` the tokens processed per step
(`G` = grad-accum micro-batches, `s` = tokens per micro-batch). Per layer:

```
t_io(i)   = W_i / B_eff          where B_eff = harmonic combine of (B_ssd, B_pcie) along the path
t_fwd(i)  ≈ 2 · P_i · s / F       (2 FLOP/param/token, one micro-batch)
t_bwd(i)  ≈ 4 · P_i · s / F
```

### 3.2 Per-step cost (with the refactored schedule)

With **layer-major** execution, each layer's weights stream **once per step**; with
**segment recompute + resident segment**, the backward re-streams each layer **once** as
well. So total weight bytes per step ≈ `2 · Σ_i W_i` (forward + backward), **independent
of `G`**. Total compute per step (forward + 1× recompute-forward + backward), summed over
all `G` micro-batches:

```
T_iter  ≈  max(  2·Σ W_i / B_eff ,           # I/O term  (fixed in G)
                 (Σ 2P_i + Σ 2P_i + Σ 4P_i)·G·s / F )   # compute term  (linear in G)
        =  max( T_io ,  (8·P_total · G · s) / F )
```

(The `8·P` lumps forward `2P` + recompute-forward `2P` + backward `4P`.)

### 3.3 The operating point — the headline rule

```
tokens/sec = (G · s) / T_iter
```

- **I/O-bound regime** (`T_io` dominates): `T_iter ≈ T_io` is **fixed**, so
  `tokens/sec ≈ (G·s)·B_eff / (2·ΣW_i)` grows **linearly in `G`**. Crank `G`.
- **The knee:** increasing `G` raises the compute term until it meets `T_io`. At the knee
  `G* ≈ T_io · F / (8·P_total · s)` the GPU is saturated and throughput is maximal for
  the hardware. Past `G*` you are compute-bound and throughput per extra `G` flattens.

> **M9 picks `G` (= `grad_accum`) and `s` (= `micro_batch`) to sit at the knee subject to
> VRAM.** M5's job is to make the schedule *actually achieve* `T_iter = max(...)` by
> overlapping I/O and compute perfectly (§6), so the model is predictive rather than
> aspirational.

### 3.4 Worked analytical estimate (illustrative — **not measured**; verify in M8)

7B model, BF16 weights ≈ 14 GB. Per step streamed ×2 ≈ 28 GB. PCIe-4 NVMe `B_eff ≈ 6 GB/s`
→ `T_io ≈ 4.7 s`. RTX 4090 realized BF16 `F ≈ 50 TFLOP/s`. With `s = 2048`:
`G* ≈ 4.7 · 5e13 / (8 · 7e9 · 2048) ≈ 2`. So `G*≈2`, `tokens/step ≈ 4096`,
**`tokens/sec ≈ 4096 / 4.7 ≈ 870`** (analytical). Faster storage (PCIe-5 / NVMe RAID-0)
lowers `T_io` and pushes `G*` and tokens/sec up until compute-bound. **These numbers are a
model output; M8 must report measured tokens/sec and label mock vs. real explicitly.**

---

## 4. Novel contributions (what goes in the paper)

Each is grounded in the existing AethelStream stack and guarded by a named test (§12).

**N1 — Streaming-Aware Recomputation Scheduling (SARS).**
A recompute cost model whose objective is *re-streamed weight bytes* (not recompute FLOPs),
plus the **segment-recompute** execution that realizes the standard 1×-forward overhead in
the streaming regime, plus a derivation of the optimal checkpoint interval `k*` and the
resident-segment decision from `(B_ssd, B_pcie, F, VRAM_cap)`. Generalizes LoHan's convex
activation-swap analysis from the *activation-offload* regime to the *weight-streaming*
regime. *Novelty:* the optimal `k*` here can be **smaller** than FLOP-only analyses
predict, because each recompute also pays disk bandwidth. *Guarded by:* T-SEG, T-THRU.

**N2 — Amortized Layer-Major Wavefront Execution.**
Layer-major traversal + grad-accum that streams each layer once per step and reuses it
across `G` micro-batches, making weight bytes `O(L)` and tokens/sec linear in `G` until the
roofline knee. The training-time analog of FlexGen's throughput-first block schedule.
*Guarded by:* T-AMORT, T-THRU.

**N3 — Low-Rank Deferred-Apply LOMO (LRDA) with exact global clipping.**
A synthesis of LOMO (no full-grad storage), GaLore (low-rank projection), and grad-accum:
the post-accumulate hook projects each micro-batch gradient to low rank and *accumulates*
it; the Adam apply is deferred to the end of backward. Because the low-rank accumulators
for the whole model are tens of MB, retaining all of them is cheap — which yields **exact
global-norm gradient clipping**, strictly removing LOMO's grouped/local-clip approximation
on the GaLore/LoRA path while preserving the one-layer full-gradient memory peak.
*Guarded by:* T-HOOK, T-CLIP.

**N4 — Deterministic, Parity-Guarded Recompute (DPGR).**
Per-`(layer, micro-batch)` RNG capture/restore so recompute reproduces the forward
*exactly* (dropout-correct), plus a runtime parity diagnostic that compares a micro-step
against an in-memory full-precision reference and responds by escalating precision or
halting — explicitly decoupled from GaLore projection refresh. This is also the mechanism
that makes M10's bit-reproducible resume possible. *Guarded by:* T-RNG, T-PARITY, T-RESUME.

**N5 — Hybrid Activation Materialization (HAM).**
Per-segment choice between *recompute* (spend idle compute) and *activation-offload to
pinned RAM* (spend idle PCIe), selected by SARS based on whether the segment is I/O- or
compute-bound. Default is recompute in the I/O-bound regime because activation-offload
would contend for the very PCIe path weight streaming uses. **In v2 this heuristic becomes
the *fallback* path; the primary path is the N1′ optimal planner.** *Guarded by:* T-HAM.

**N6 — Streaming Cut-Cross-Entropy + tiled LM-head.**
Chunked CE that never materializes `[G·S·V]` logits, fused with tile-wise streaming of the
LM-head weight, giving `O(chunk)` final-layer memory. *Guarded by:* T-CE.

### v2 contributions (the state-of-the-art elevation)

**N1′ — Streaming-Aware Rematerialization–Paging (SARP) — the headline contribution.**
SARS (N1) chose checkpoint spacing `k*` from bandwidth alone. SARP replaces it with a
*provably-near-optimal joint schedule* generalizing Checkmate (optimal rematerialization,
MLSys'20) and POET (joint rematerialization **+ paging**, ICML'22) to a regime neither
modeled: **the weights themselves are the dominant paged object**, and weight-prefetch I/O
and activation-paging I/O **share one NVMe/PCIe channel**. For each segment the planner
assigns one of `{RECOMPUTE, RETAIN_VRAM, PAGE→compressed-RAM (M2 LZ4 tier), PAGE→NVMe}` so
as to **maximize tokens/sec** subject to (i) VRAM cap, (ii) RAM cap, (iii) the shared I/O
budget `Σ(weight bytes) + Σ(paged-activation bytes) ≤ B_eff · T_iter`. Because a transformer
is a regular linear chain of a few block-types, the general MILP collapses to a **dynamic
program over (segment-boundary × residency-state)** that solves in milliseconds — so it runs
**once at plan time inside M9** (which already solves the inverse-config problem and emits the
frozen `TrainingPlan`). M5 is the *executor* of the emitted schedule; the v1 SARS/HAM
heuristics remain only as the no-plan fallback. The novelty for the paper: prior remat/paging
optimizers assume resident weights; SARP is the first to co-schedule weight streaming and
activation (re)materialization against a shared bandwidth budget. *Guarded by:* T-SARP,
T-THRU, T-HAM.

**N1″ — Selective-region recompute (folds Korthikanti MLSys'23 into A2′).**
When the SARP schedule says RECOMPUTE, M5 recomputes only the **memory-heavy / compute-cheap**
sub-ops (attention softmax+dropout interior) and **retains** the cheap-to-store outputs,
cutting recompute FLOPs ~90% at near-equal memory. This **lowers the compute knee `G*`**, so
the linear-in-`G` throughput region (N2) extends further before compute saturates. *Guarded
by:* T-SELREC.

**N3′ — Projection-agnostic clipping with a stated guarantee.**
N3's "exact clip for free" holds only under an **orthonormal** projector (SVD GaLore). M4 may
adopt random-projection optimizers (**APOLLO** MLSys'25, **Q-GaLore** INT4, **Fira**), under
which the projection is norm-preserving only up to a Johnson–Lindenstrauss bound. N3′ makes
the hook treat the projector as opaque and states the guarantee **conditionally**: *exact*
under orthonormal projection; *approximate (JL bound)* under random projection, with an
optional O(1)-per-layer true-Frobenius accumulator for clip-critical layers to restore
exactness. *Guarded by:* T-CLIP (both projector arms).

---

## 5. Refactored algorithms

Notation: `L` layers; segment size `k = checkpoint_freq`; segment `S_b = [b·k, (b+1)·k−1]`;
`G = grad_accum`; `a[m]` = wavefront activation of micro-batch `m`. Row-vector convention:
`y = x @ W`, `x:[…,d_in]`, `W:[d_in,d_out]`. LoRA: `y = x@W + scale·(x@A@B)`,
`A:[d_in,r]`, `B:[r,d_out]`, `scale = 1.0`. (**Correction to the M6 doc's merge formula:**
the merged weight is `W_merged = W + scale·(A @ B)`, *not* `B @ A^T` — the latter does not
conform in this convention.)

### A1 — Layer-major forward with sparse checkpointing + RNG capture

```text
for m in 0..G: a[m] = embed(input_ids[m])         # G wavefronts live simultaneously
for i in 0..L:
    ready = flowcast.take_ready(i)                # weights of layer i, streamed once
    flowcast.on_layer_start(i, Direction::Forward)# triggers prefetch of i+1..i+W
    upload(ready -> VRAM)                          # or UVA for slab tensors
    for m in 0..G:
        rng_capture(i, m)                          # save dropout/stochastic RNG state
        a[m] = layer_i.forward(a[m])               # full precision of this layer
        if i % k == 0:
            checkpoint[i][m] = pin_clone(a[m])     # into M2 pinned buffer (FFI), NOT a py dict
    evict(layer_i)                                 # release weight buffer back to M3/M2
# loss handled by A8 (no full logits)
```
VRAM at any instant ≈ `1 layer weights + G wavefronts + workspace`. Checkpoints live in
**M2 pinned RAM**, sized `⌈L/k⌉ · G · |activation|`; under soft memory pressure M2 may
INT8-compress them (M2 `compress_checkpoint_fp16_to_int8`, `set_compressed(true)`), and
M5 decompresses on read (M2 `decompress_checkpoint_int8_to_fp16`).

### A2 — Segment-wise recompute backward (the fix, N1)

```text
init_upstream()                                    # from A8's loss grad, per micro-batch
for b in (num_segments-1)..=0:                     # segments in REVERSE
    seg = S_b = [b*k .. min((b+1)*k-1, L-1)]

    # ---- recompute-forward through the segment ONCE (per micro-batch) ----
    for j in seg (ascending):
        ready_j = flowcast.take_ready(j)
        flowcast.on_layer_start(j, Direction::Recompute)
        upload(ready_j); RESIDENT[j] = ready_j      # keep resident IF VRAM budget allows
        for m in 0..G:
            rng_restore(j, m)                        # reproduce the forward's dropout mask
            grad_tracking = (j == top_of_remaining)  # only the active layer tracks grad
            seg_act[j][m] = layer_j.forward(input_for(j,m))   # input = checkpoint or seg_act[j-1][m]
        if not keep_resident: evict(layer_j)         # else reuse in backward (saves 1x stream)

    # ---- backward through the segment, layer-major, reusing seg_act ----
    for i in seg (descending):
        wi = RESIDENT[i] if keep_resident else (re = flowcast.take_ready(i); upload(re))
        m4.begin_layer_accum(i)                      # zero low-rank grad accumulators
        for m in 0..G:
            (g_in[m], _) = layer_i.backward(seg_act[i-1][m], upstream[m])  # see A3 hook
            upstream[m] = g_in[m]
        # NOTE: apply is DEFERRED to A6 (after the whole backward) for exact global clip
        evict(layer_i); free(seg_act[i][*])
```

**Recompute cost:** each segment is recomputed forward exactly once → **1× extra forward
pass total**, vs. the original plan's `~(k−1)/2×`. **Weight streaming:** `2× ΣW_i` per
step when `keep_resident` holds (forward 1× + backward 1×); `3×` when the segment cannot
stay resident (recompute-forward + backward each stream). `keep_resident` requires `k`
layers + their `G` activations to fit in VRAM; **M9 chooses `k` to satisfy this and sit at
the §3 knee.** Smaller `k` ⇒ cheaper residency and less recompute-per-block, at the cost of
more checkpoint RAM — exactly the convex trade SARS solves.

> **`keep_resident` vs. M3's HotSet.** Do **not** implement residency in M5. `keep_resident`
> is expressed by issuing `on_layer_start(j, Direction::Recompute)` for the segment window so
> FlowCast's **A1 Recompute FSM + A5 HotSet** keep those layers pinned; M5 reads them back via
> `take_ready`. If `ReadyLayer.copy_event` is `Some`, M5 calls
> `cuda_stream_wait_event(compute_stream, copy_event)` before using `slab_device_ptrs`.

### A2′ — SARP executor + selective-region recompute (v2, N1′/N1″)

A2 above is the *uniform* schedule (every segment RECOMPUTE, residency all-or-nothing). A2′
**executes a per-segment schedule emitted by M9** instead of deciding locally:

```text
plan_seg = training_plan.activation_schedule[b]   # one of the 4 actions, decided by M9's SARP DP
match plan_seg.action:
    RETAIN_VRAM        -> seg activations were kept from forward; no recompute, no I/O
    PAGE_COMPRESSED_RAM-> fetch seg activations from M2 LZ4 tier (decompress ~4 GB/s); no recompute
    PAGE_NVME          -> fetch seg activations via M3 (they were written back in forward); no recompute
    RECOMPUTE          -> run the A2 recompute-forward, BUT only the ops flagged `recompute`:
                          for each op in layer:
                              if plan_seg.recompute_ops[op]: recompute it            # attn interior
                              else:                          reuse retained output    # cheap-to-store
```

`recompute_ops` is the **selective-recompute** mask (N1″): by default `retain` LayerNorm
statistics, the residual-stream tap, and small linear outputs; `recompute` the attention
softmax/dropout interior. M9 sets the mask from the warmup FLOP/byte profile; M5 honors it and
falls back to full-layer recompute only when a high-pressure callback fires mid-segment. The
four actions all produce **bit-identical gradients** to A2 (selective recompute changes *what*
is stored vs. recomputed, never the math) — T-SARP and T-SELREC assert this against the A2
baseline. When `training_plan.activation_schedule` is absent (older M9, or a degraded tier),
A2′ falls back to A2 + the N5 greedy heuristic.

### A3 — LRDA hook: project + accumulate (no full-grad storage), N3

```python
def make_hook(p, m4, layer_idx):
    def hook(grad):                       # PyTorch hands us the FULL grad transiently
        # GaLore projection to low rank happens immediately; full grad never persists
        m4.project_and_accumulate(grad, layer_idx, p.name)   # accumulates G_r = P^T·grad
        return None                       # discard the full gradient now (LOMO)
    return hook

for name, p in active_layer.named_parameters():
    p.register_post_accumulate_grad_hook(make_hook(p, m4, layer_idx))
```
Peak extra memory during a layer's backward = **one matrix's transient full gradient** (the
LOMO guarantee) + the *small* low-rank accumulator. Across `G` micro-batches the hook fires
`G` times and **sums** into the same low-rank accumulator (grad-accum). For **LoRA mode**,
only `A`,`B` have hooks; the frozen base `W` has `requires_grad=False` and never produces a
grad (T-ISO in M6).

### A4 — MeSP `h = xA` LoRA recompute (retained, right-sized)

For LoRA matrices, `∂L/∂B = h^T @ (scale·g_up)` with `h = x@A`. Since the layer input `x`
is already available (recomputed in A2) and `r` is tiny, `h` is recomputed by one small
matmul rather than stored — saving the `[G·s·r]` tensor. *Honest note:* with `r ∈ {16,64}`
this saving is small in absolute terms; the dominant savings come from N1/N2. Keep it for
correctness symmetry and because it composes with `keep_resident=false`. *Guarded by:* the
retained Test 4 (T-MESP).

### A5 — Mixed precision (BF16 default; FP16 fallback uses M2's scale table)

- **Ampere+ (BF16):** skip overflow checks entirely. Recompute and backward in BF16, FP32
  accumulation in matmuls/reductions. This is the default for the consumer target (RTX 30xx/40xx).
- **Turing / no-BF16 (FP16):** dynamic loss scaling via **M2's `PerLayerScaleTable`** (EWA
  per-layer scale) + **M2's `count_overflow_fp16` / `fused_overflow_check`** kernels. On
  overflow for a layer: **skip that layer's apply this step**, back off its scale; on a
  clean growth interval, grow. M5 does **not** re-implement the scaler — it reads/writes the
  M2 table.
- **Master weights:** *LoRA mode* keeps FP32 master copies of the tiny adapters only (base
  is frozen, read-only stream — **no write-back**). *Full-param GaLore mode* updates BF16
  weights with **stochastic rounding** (avoids the FP32-master NVMe doubling) and writes
  updated shards back through M3→M2→M10. The choice is a `TrainingPlan` field.

### A6 — Exact global-norm clipping via deferred low-rank apply (N3)

Because A3 leaves *all* layers' low-rank accumulators resident at end-of-backward (tens of
MB total), we can do **exact** global clipping that the single-pass LOMO design could not:

```text
# after A2 completes the full backward:
gsq = 0
for (layer, p) in trainable: gsq += m4.lowrank_grad_sqnorm(layer, p)   # exact in low-rank basis*
gnorm = sqrt(gsq)
clip = min(1.0, max_norm / (gnorm + 1e-6))
for (layer, p) in trainable:
    m4.apply_update(layer, p, scale=clip)        # one Adam step per param, clipped
    m4.zero_accum(layer, p)
```
\*The Frobenius norm is preserved under the orthonormal GaLore projection, so the low-rank
sq-norm equals the projected gradient's true norm; document this in the paper as the reason
exact clipping is admissible without ever forming the full gradient. **Fallback:** if a run
uses *no* low-rank projection (pure full-rank, rare here), fall back to LOMO **grouped/local
clipping** and cite LOMO's smooth-loss-surface justification — but the default GaLore/LoRA
path is exact. *Guarded by:* T-CLIP.

### A7 — Parity diagnostic + projection-refresh decoupling (N4, C4)

```text
# (i) GaLore projection refresh — OWNED BY M4, fixed cadence (e.g. every refresh_interval
#     steps). M5 only forwards the step counter. This is hygiene, not a drift remedy.

# (ii) Parity diagnostic — M5, every parity_check_interval steps (default 500):
stream_g  = aethelstream_layer_grad(micro_batch, ref_layer)        # full FULL-rank, our path
ref_g     = inmemory_fp32_autograd_grad(micro_batch, ref_layer)    # reference (no streaming)
rel = max|stream_g - ref_g| / (max|ref_g| + 1e-12)
if rel > tol_warn:    log_warn(rel); escalate_recompute_precision(ref_layer)   # BF16->FP32 recompute
if rel > tol_halt:    checkpoint_now(); raise ParityHalt(ref_layer, rel)        # never mask a bug
```
The diagnostic measures *implementation* drift (a kernel bug, a non-deterministic reduction,
a precision regression) against full-rank autograd — **not** the intended GaLore
approximation error. It therefore never "fixes" things by mutating projections.

### A8 — Streaming Cut-Cross-Entropy + tiled LM-head (N6)

```text
# hidden h: [G, s, d];  LM-head W_lm: [d, V] (huge) streamed in tiles over V
loss = 0; g_h = 0
for v_chunk in chunks(V, chunk):
    W_lm_chunk = flowcast.take_ready(lm_head, v_chunk)      # tile-wise stream
    logits_chunk = h @ W_lm_chunk                            # [G,s,chunk] only
    accumulate online-softmax stats (max, sumexp) and chunk's contribution to loss
    accumulate g_h += d(loss)/d(h) contribution; (optionally) project LM-head grad via M4
# full [G,s,V] logits NEVER materialized; peak = O(chunk)
upstream = g_h    # seeds A2's init_upstream
```
Uses the online-softmax / Cut-CE trick: two passes over vocab chunks (one for the
log-sum-exp normalizer, one for the gradient) keep memory at `O(chunk)`. The LM-head is also
streamed (it is often the largest single matrix), so its weight memory is `O(chunk·d)`.

### A9 — Hybrid materialization selector (N5)

```text
for each segment S_b:
    if SARS.is_io_bound(S_b):  mode = RECOMPUTE     # idle compute is free; don't touch PCIe
    else:                      mode = OFFLOAD       # idle PCIe; avoid extra recompute FLOPs
# OFFLOAD: during A1, push that segment's per-layer activations to pinned RAM; during A2,
# reload them instead of recomputing (no rng_restore needed, but more PCIe traffic + RAM).
```
Both modes must produce **bit-identical** gradients (T-HAM); OFFLOAD is the safety net when
a segment is compute-bound or when recompute would stall the pipeline.

### A10 — Determinism + step-boundary state for M10 (N4)

- **RNG:** per-`(layer, micro-batch)` capture at forward, restore at recompute (A1/A2).
  Persist the global RNG generator state at step start.
- **Deterministic kernels:** require deterministic reductions/atomics on the recompute and
  backward path (cuDNN/cuBLAS deterministic flags; flash-attention deterministic backward).
  Non-deterministic reductions are the most common silent parity breaker.
- **Step boundary:** after A6 applies all updates, `(updated weights pending write-back,
  optimizer states + version, RNG state, data position, step)` form a **consistent
  snapshot**. M5 calls `m10.begin_step(step)` at entry and records each weight commit through
  M3's write-back (which M10 wraps with `atomic_commit`); on resume, `m10.recover()` returns
  `{step, optimizer_version, rng_state, data_pos}` and M5 restarts from there bit-for-bit.

---

## 6. Execution schedule and overlap (how the model in §3 is realized)

M5 never blocks on I/O if the window is sized right. The contract with M3 FlowCast:

- `on_layer_start(i, dir)` — called the instant M5 *begins* compute on layer `i`; FlowCast
  immediately prefetches the next `W` layers in `dir` (`i+1..` forward; `i-1..` backward;
  for recompute the **descending-segments / ascending-within-segment** order — M5 emits the
  exact layer sequence so FlowCast prefetches along it).
- `take_ready(i)` — blocking-with-timeout fetch; a miss is an explicit `PrefetchMiss{i}`,
  never a silent stall or wrong buffer.
- `on_weights_updated(i, &buf)` — after A6 applies layer `i`'s update (full-param mode),
  enqueues a **delayed write-back** overlapped with the next layer's compute (M10 makes it
  atomic). LoRA mode skips this for the base; only adapter shards are written at merge time.

```
time ─────────────────────────────────────────────────────────────────────►
GPU :  [fwd L0 ×G][fwd L1 ×G][fwd L2 ×G] … │ [recompute Sb][backward Sb] …
I/O :  [stream L1 ][stream L2 ][stream L3] … │ [stream next-seg] [write-back L_i] …
        └ prefetch hidden behind compute ┘     └ recompute hidden behind streaming ┘
```

**Buffer accounting (what M9 must budget):** weight buffers = `W` (window) + `k`
(resident segment, backward) ; activation = `G` wavefronts (VRAM) + `⌈L/k⌉·G` checkpoints
(pinned RAM) + (`k·G` segment activations during a segment's backward). Optimizer low-rank
accumulators = tens of MB resident (M4). M5 asserts these against the `TrainingPlan` peaks
at startup and refuses to run a plan that would exceed them (preflight, ties to M9 A6).

---

## 7. Frozen seams M5 binds to

### 7.1 Consumed from M3 FlowCast (the only I/O surface)
```
FlowCast::on_layer_start(layer_idx, dir: Direction)
FlowCast::take_ready(layer_idx) -> ReadyLayer            # PrefetchMiss{idx} on miss
FlowCast::on_weights_updated(layer_idx, src: &PinnedBuffer)
FlowCast::warmup() -> HardwareProfile                    # feeds M9; M5 may read timing
```
M5 must **never** call M2 `prefetch`/`write_async`/`claim` directly — all through FlowCast.

### 7.2 Consumed from M2 RamFlow (only the pieces FlowCast doesn't hide)
```
PinnedBuffer                              # checkpoint storage (via FFI handle)
PerLayerScaleTable::{get_scale, gradient_variance, mark_resident, is_resident}   # FP16 path
kernels::{count_overflow_fp16, fused_overflow_check,
          compress_checkpoint_fp16_to_int8, decompress_checkpoint_int8_to_fp16}
```
Invariants honored: a `PinnedBuffer` stays alive until its CQE; `set_compressed(true)`
immediately after compressing a checkpoint; offsets 512-aligned (M1 guarantees).

### 7.3 Consumed from M4 optimizer (GaLore)
```
m4.project_and_accumulate(grad, layer_idx, name)   # low-rank projection + accumulate (A3)
m4.lowrank_grad_sqnorm(layer_idx, name) -> f64      # for exact global clip (A6)
m4.apply_update(layer_idx, name, scale)             # one Adam step, clipped (A6)
m4.zero_accum(layer_idx, name)
m4.notify_step(step)                                # M4 owns projection-refresh cadence (A7)
```
(If the current M4 surface differs, M5's prompts add a thin adapter; **M4's public API is not
modified** — a difference is a STOP-and-note, per project rules.)

### 7.4 Consumed from M6 LoRA
```
m6.lora_forward(x, W, A, B, scale)        # fused; tile-streaming aware (M6 A2/A3)
m6.lora_backward(...)                      # produces ∂A, ∂B via MeSP h=xA recompute (A4)
m6.rank_of(layer_idx) -> r                 # sensitivity schedule: r=64 edges, r=16 middle
m6.merge(layer_idx) -> writes W + scale·(A@B) back   # end-of-training (corrected formula)
```

### 7.5 Consumed from M9 `TrainingPlan` (frozen)
`checkpoint_freq (=k)`, `micro_batch (=s)`, `grad_accum (=G)`, `precision_schedule`
(per-layer dtype for stream & compute), `optimizer_rank`, `tier`
(`FullGaLore | LoraOnly | TopKFreeze(K) | Int4Everywhere | …`). M5 also reads `w_max_hint`
(window) and may receive a mid-run **plan delta** changing only `window/precision/k`.

### 7.6 Exposed to M7 (PyO3 surface) / M10
```
doublepass.step(batch) -> StepMetrics                  # one full training step
doublepass.set_plan(TrainingPlan)                       # at start; or apply_delta(delta)
doublepass.snapshot() -> ConsistentState                # for M10.atomic_commit at step end
doublepass.parity_probe(ref_layer) -> f64               # on-demand diagnostic
m10.begin_step(step) / StepGuard::record_commit(layer, gen) / m10.recover()
```

---

## 8. VRAM / RAM budget (peak)

```
VRAM_peak ≈  max( forward:  G·|act| + 1·|layer_w| + |workspace| + |W·prefetch_w| ,
                  backward: G·|act| + k·|layer_w (resident)| + |bwd_workspace| ,
                  loss:     G·s·chunk + |LM-head tile| )
RAM_peak  ≈  ⌈L/k⌉·G·|act (ckpt)|  + (HAM offload segments)·G·|act|
            + low-rank optimizer accumulators (tens of MB)  + pinned weight pool (M2)
```
`|act|` for one micro-batch ≈ `s · d · bytes` (e.g. 2048·4096·2 = 16 MB). These formulas are
what M5 asserts against the plan; they are also the inputs M9 optimizes.

---

## 9. Modes and the degradation ladder (ties to M9 A5)

M5 must run every rung M9 may select:
- **FullGaLore:** all layers trainable, GaLore low-rank states, A3/A6 exact clip, full-param
  write-back (stochastic-rounded BF16).
- **LoraOnly:** base frozen (read-only stream, **no write-back** — much cheaper I/O and TBW),
  only A/B trainable, MeSP recompute (A4), merge at end (A4/M6).
- **TopKFreeze(K):** freeze all but the K most-sensitive layers (e.g. edges); frozen layers
  skip the hook entirely and never write back. A cheap middle ground.
- **Int4Everywhere / CompressedRamTier / MmapTier:** M5 honors the per-layer precision and
  the checkpoint compression; recompute precision follows §A5/A9 (escalate under the guard).

The mode is **not** chosen by M5; it is read from `TrainingPlan.tier`. M5's responsibility is
that each mode is *correct* (parity tests run per mode) and *as fast as the model allows*.

---

## 10. Correctness & numerical strategy (summary)

1. **Recompute at forward precision** + **deterministic kernels** + **RNG restore** ⇒ the
   gradient is computed at the *same* activation the forward produced. This is the precondition
   for any parity claim.
2. **Exact global-norm clipping** (A6) on the GaLore/LoRA path ⇒ the clip behaves like
   standard training, not LOMO's local approximation.
3. **BF16 default, FP16-with-scale-table fallback** ⇒ no overflow handling on modern cards;
   honest dynamic scaling on old ones.
4. **Parity diagnostic** (A7) watches for *implementation* drift and escalates/halts — it
   never edits the optimizer to hide a bug.
5. **Mock vs. real** is labeled everywhere (benchmarks, parity numbers). A test that runs code
   but asserts nothing is a FAIL (project rule).

---

## 11. Risks, open questions, honest limitations

- **`keep_resident` VRAM pressure.** Holding `k` layers + `G` wavefronts resident in backward
  is what gets streaming to 2×. If VRAM forces `keep_resident=false`, streaming rises to 3×
  and tokens/sec drops ~33%. M9 must trade `k`, `G`, and residency jointly; document the chosen
  point per run.
- **Full-param write amplification.** FullGaLore writes updated shards every step → TBW and
  write bandwidth pressure. M2 `WriteBudgetManager` + M3 write batching + M10 atomic commit
  mitigate, but LoRA is the genuinely cheap path; be honest in the paper about which results are
  LoRA vs. full-param.
- **Reduced-precision recompute** (A9 compute-bound case) is a real parity risk; only enable it
  under the guard and report any layers it touched.
- **Determinism cost.** Deterministic kernels can be slower than non-deterministic ones; quantify
  the cost (it may shift the §3 knee) rather than assuming it is free.
- **Reference verification (audit rule).** Confirm arXiv IDs before citing. I could **not**
  verify **MeSP (2602.13069)** — the `2602` prefix implies a 2026 posting and the ID may be a
  placeholder — or **TERAIO (NeurIPS'25)**. Treat both as *to-verify*; do not let the `h=xA`
  derivation depend on an uncitable source (it is standard LoRA calculus regardless).

---

## 12. Test matrix (the contract for "it works")

| ID | Test | Asserts | Guards |
|---|---|---|---|
| **T-PARITY-1** | *Foundational.* 1-layer block, known weights, one fwd+bwd through M5's double-pass vs. PyTorch autograd. | `max|Δgrad| < 1e-5` (FP32), `< 1e-3` (BF16), **clipping disabled** to isolate the math. **If this fails, everything else is invalid.** | core math |
| **T-ACT-2** | Recompute fidelity: golden full-forward activations vs. mini-forward recompute (ckpt 0 → layer 4). | `max diff < 1e-5`. | A2 |
| **T-HOOK-3** | LRDA hook vs. standard store-then-update over 100 steps. | params differ `< 1e-6`; transient full-grad memory `≤ 2× largest param tensor`. | N3/A3 |
| **T-MESP-4** | LoRA `h=xA` recompute vs. stored `h`. | `max diff < 1e-6`. | A4 |
| **T-CONV-5** | 125M model, 500 steps, M5 vs. PyTorch full training. | M5 loss within **3%** of reference at step 500. | end-to-end |
| **T-PARITY-6** | Inject `0.01` offset into one layer's grad. | guard detects within `parity_check_interval`; responds (escalate/halt); parity restored after fix. | A7/N4 |
| **T-SEG** | Segment-recompute backward vs. full-autograd backward (multi-segment model). | `max|Δgrad| < 1e-5`; **measured weight-load count == 2×L** (resident) or 3×L (non-resident). | N1/A2 |
| **T-AMORT** | Vary `G`; count weight bytes streamed per step. | bytes **independent of `G`** (== `2·ΣW_i`); tokens/sec scales ~linearly with `G` below the knee. | N2 |
| **T-CLIP** | Exact global-norm clip vs. reference global clip, **both projector arms**: orthonormal (SVD) and random (APOLLO-style). | orthonormal: clip matches reference within tol; random: matches within JL bound, and the O(1)-Frobenius-accumulator path restores exact match on clip-critical layers; grouped-clip fallback engages only when no projection. | N3/N3′/A6 |
| **T-RNG** | Model **with dropout**: recompute reproduces forward activations exactly. | `max diff == 0` (deterministic) with RNG restore; **fails without it** (negative control). | N4/A10 |
| **T-CE** | Streaming Cut-CE loss+grad vs. `torch` cross-entropy. | loss `< 1e-4`, grad `< 1e-3`; **peak logit memory `O(chunk)`** not `O(V)`. | N6/A8 |
| **T-HAM** | RECOMPUTE vs. OFFLOAD on the same segment. | **bit-identical** gradients; selector picks RECOMPUTE when I/O-bound, OFFLOAD when compute-bound (simulated). | N5/A9 |
| **T-SARP** | Execute all four SARP actions (RETAIN / PAGE-RAM / PAGE-NVMe / RECOMPUTE) on the same segment. | all four yield **bit-identical** gradients vs. the A2 baseline; the planner's chosen action minimizes modeled `T_iter` on a known cost grid. | **N1′**/A2′ |
| **T-SELREC** | Selective-region recompute (attn-interior only) vs. full-layer recompute. | **bit-identical** gradients; **measured recompute FLOPs reduced** vs. full-layer; measured knee `G*` shifts up. | N1″/A2′ |
| **T-RESUME** | Crash mid-step → `m10.recover()` → resume. | resumed step is **bit-reproducible** vs. uninterrupted run (needs deterministic + RNG). | N4/A10 |
| **T-THRU** | Roofline: measured `T_iter` vs. `max(T_io, T_compute)` model across configs. | within **15%**; **mock vs. real labeled**; GPU-idle `< 20%` (simulated) and reported (real). | N1/N2 |

Run order in CI mirrors the sprint order (§ prompts file): **T-PARITY-1 first** — it gates
everything.

---

## 13. References

Confident (verify arXiv IDs before camera-ready):
- LOMO — *Full Parameter Fine-tuning for LLMs with Limited Resources*, Lv et al., 2023, **arXiv 2306.09782**. (fused backward+update; grouped clipping; smooth-loss-surface argument)
- GaLore — *Memory-Efficient LLM Training by Gradient Low-Rank Projection*, Zhao et al., 2024, **arXiv 2403.03507**. (low-rank projection; constant-projection stability)
- LoRA — Hu et al., 2021, **arXiv 2106.09685**.
- Gradient checkpointing — Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016, **arXiv 1604.06174**. (the segment-recompute baseline)
- FlexGen — Sheng et al., 2023, **arXiv 2303.06865**. (throughput-first offload schedule; the N2 analogy)
- FlashAttention — Dao et al., 2022, **arXiv 2205.14135** (+ FA-2, 2307.08691). (memory-efficient, deterministic-capable attention recompute)
- Cut Cross-Entropy — Apple, 2024, **arXiv 2411.09009**; and Liger-Kernel (2024). (no-logit-materialization CE — basis for A8)
- 8-bit optimizers — Dettmers et al., 2021, **arXiv 2110.02861**. (the bitsandbytes states M4 uses)
- ZeRO-Offload / ZeRO-Infinity — Ren et al. 2021 (**2101.06840**) / Rajbhandari et al. 2021 (**2104.07857**). (CPU/NVMe offload precedent)
- LoHan — consumer-GPU 100B fine-tune; source of the convex-swap `T_iter=max(...)` model that SARS generalizes. (verify venue/ID)

**v2 additions (verified during this session):**
- Selective activation recomputation — Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, **arXiv 2205.05198**, MLSys 2023. (basis for N1″ selective-region recompute)
- Checkmate — Jain et al., *Breaking the Memory Wall with Optimal Tensor Rematerialization*, **arXiv 1910.02653**, MLSys 2020. (optimal rematerialization as ILP — the N1′ ancestor)
- POET — Patil et al., *Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging*, **arXiv 2207.07697**, ICML 2022. (joint remat **+ paging** MILP — direct N1′ ancestor; shows greedy page-then-remat is suboptimal)
- DTR — Kirisame et al., *Dynamic Tensor Rematerialization*, ICLR 2021. (online greedy remat; the runtime-heuristic fallback analog)
- APOLLO — Zhu et al., *SGD-like Memory, AdamW-level Performance*, **arXiv 2412.05270**, MLSys 2025. (random-projection optimizer; motivates N3′ projection-agnostic clipping)
- Q-GaLore (INT4 projection, 2024) and Fira (full-rank residual, 2024) — GaLore successors M4 may adopt; relevant to N3′.

**To verify (could not confirm; do not let a derivation depend on them):**
- **MeSP** (cited as `2602.13069v1`) — `h=xA` recompute. The ID prefix implies 2026; treat as a placeholder and rest the derivation on standard LoRA calculus.
- **TERAIO** (cited as NeurIPS'25), ZenFlow, SSDTrain, DeepNVMe/FastPersist, MemAscend — confirm before citing.