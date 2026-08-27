# Porting INT4 Flash-Decoding from Triton to Pallas

- **Author:** Archana Suresh Patil
- **Date:** August 26, 2026
- **Base repository:** https://github.com/ArchanaChetan07/int4-kv-flash-attention
- **Duration:** 7 weeks, ~12 h/week · **Cash cost:** < $100

---

## 0. Summary

I have a working INT4 KV-cache quantizer fused with a flash-decoding attention kernel,
written twice — once in CUDA, once in Triton — and validated against a NumPy oracle.
I propose to write it a third time in **Pallas on JAX**, hold the algorithm and the oracle
fixed, and report exactly where the second kernel-programming model diverges from the first.

Porting a kernel I already wrote is the whole methodology. If I picked an unfamiliar
algorithm, every obstacle would be ambiguous: is this hard because Pallas is hard, or
because I don't understand the math yet? With the algorithm fixed and the oracle fixed,
every hour of difficulty is attributable to the tooling. That attribution *is* the deliverable.

Two scoping findings changed the plan before a line was written:

1. **The GPU-to-GPU comparison is not available on my hardware.** Pallas's Triton GPU backend
   is deprecated and slated for removal; its replacement, Mosaic GPU, targets Hopper/Blackwell.
   My local GPU is a Turing T1000. The port's primary target becomes **TPU** — the honest, and
   more interesting, comparison. (§3)
2. **The stated success criterion contains an ambiguity that would let me quietly pass.**
   "Matching the reference oracle to the tolerance the CUDA path already meets" reads as one
   threshold but is two; and on TPU a third effect — float32 matmuls executed as multiple
   bfloat16 passes — puts a hardware floor under achievable error that has nothing to do with
   whether my port is correct. §6 proposes a control that separates the two.

---

## 1. The control variable — what already exists

`src/quantize_int4_ref.py` and `src/flash_decode_ref.py` are plain NumPy and define the contract:

```
scale[c] = (max[c] - min[c]) / 15
zp[c]    = -min[c] / scale[c]
q        = clip(round((kv - min[c]) / scale[c]), 0, 15)

m_new = max(m_old, max(logits))
corr  = exp(m_old - m_new)
l_new = l_old * corr + sum(exp(logits - m_new))
o_new = o_old * corr + V @ exp(logits - m_new)
```

Every backend is measured against this and nothing else. The Pallas port adds a third caller
to the same oracle; it does not get a new one.

| Check | Result | Hardware |
|---|---|---|
| CUDA quantizer vs NumPy oracle | 0.000% bin disagreement, scales rtol 1e-4 | T1000 |
| Triton quantizer vs NumPy oracle | <=1 bin apart, <1% of elements | CI, interpreter mode |
| Fused INT4 attention vs oracle | MAE 3.1e-8 | T1000 |
| Variable-length + empty-block cases | MAE <= 3.4e-8 | T1000 |
| Kernel latency, b8 x h32 x d128 x s2048 | 2.84 ms — 3.9x over serial baseline | T1000 |
| KV compression vs FP16, incl. scale/zp | 3.98x | — |
| Test suite | 50 passing on GPU, 48 GPU-free | both |

**The CI trick this port depends on.** The Triton port is validated on CPU-only runners via
`TRITON_INTERPRET=1`. Pallas has the exact analogue — `pallas_call(..., interpret=True)` — and
§6 shows it does more than preserve CI: it becomes the instrument that separates a port bug
from a hardware precision floor.

---

## 2. Why this port, and why this kernel

**vLLM's TPU backend is Pallas.** The Ragged Paged Attention kernel serving as the primary TPU
attention path for both vLLM and SGLang is written in Pallas and compiled through Mosaic; RPA v3
explicitly supports quantization dtypes and arbitrary tensor-parallel degrees.

The base repository was built for paged-KV serving engines like vLLM. So this is not a synthetic
language comparison: **Pallas is the language this kernel would have to be written in to run in
the engine it was designed for, on TPU.** The port has a destination, and the destination has an
incumbent kernel to be measured against — RPA reports up to **86% memory-bandwidth utilization**
in decode on TPU7x. A one-person seven-week kernel should not expect to reach that, but a
proposal that reports its distance from a published number is worth more than one that reports a
raw millisecond count with nothing to divide it by.

---

## 3. The hardware reality that reshaped this plan

| Pallas backend | Status | Hardware floor | Available to me |
|---|---|---|---|
| Mosaic TPU | Primary, actively developed | TPU v3+ | **Yes — free tier** |
| Mosaic GPU | Recommended GPU path | Hopper / Blackwell | Rental only |
| Triton (`pallas.triton`) | **Deprecated**, removal announced | Ampere | No — and dying |

My local device is an **NVIDIA T1000 8GB — Turing, SM 7.5**, driver 596.51, CUDA 12.5 toolkit.
It sits below the floor of *both* live GPU backends. The Pallas changelog is unambiguous about
the one that would otherwise have covered it: the Triton backend "is deprecated and will be
removed in a future version of JAX."

**Consequence.** Porting Triton → Pallas *on GPU* would have lowered my Pallas code back through
Triton — comparing a language against itself through a deprecated adapter, and the one thing it
measured would be scheduled for deletion. **The port targets Mosaic TPU as its primary backend.**
Mosaic GPU on a rented H100 is a stretch leg (§9, W6), not a dependency.

This is a better project than the one I set out to scope. On GPU, much of Triton's mental model
would have transferred trivially because the compiler underneath was the same compiler. On TPU
there is no shared substrate: no warps, no shared memory to stage by hand, no thread count to
tune, a vector unit with fixed (8, 128) register tiling, and a grid that runs sequentially in
lexicographic order rather than as a swarm of independent programs. **That is where a mental
model can actively mislead, and therefore where the write-up has something to say.**

---

## 4. Deliverables

### D12 — The quantizer, reimplemented in Pallas

Per-channel asymmetric INT4 quantization, validated against the same NumPy oracle, to the
tolerance contract in §6.

The Triton kernel runs one program per channel and loops over rows inside the program. That
structure cannot survive on TPU: Pallas TPU fully unrolls in-kernel loops at compile time, so a
loop over 4096 rows becomes 4096 unrolled bodies. **The row loop has to become a grid dimension.**
The two-pass min/max-then-quantize structure then needs a running accumulator carried across grid
steps in VMEM scratch, initialized under `@pl.when(pl.program_id(0) == 0)` and relying on Pallas
TPU's guarantee that consecutive grid invocations may write the same output slice without a race.

Rounding stays half-up — `jnp.floor(x + 0.5)` — to match CUDA/Triton rather than the oracle's
banker's rounding, inheriting the existing <=1-bin tolerance.

```
+ src/quantize_int4_pallas.py
+ tests/test_pallas_quantize.py     (mirrors test_triton_quantize.py case for case)
~ .github/workflows/ci.yml          (adds a CPU-runner job at interpret=True)
```

### D13 — The fused attention kernel, ported

Online-softmax flash decoding over paged INT4 KV blocks, output-parity-checked against
`online_softmax_ref` including variable-length and empty-page cases.

This is the hard half. The CUDA kernel's parallel decomposition is warp-level: warps stride over
sequence positions, each carrying its own `(m, l)` state plus an output accumulator in lane
registers, reduced with `__shfl_down_sync` and merged with a log-sum-exp combine. **None of that
vocabulary exists on TPU.** The equivalent is `m_ref`/`l_ref`/`o_ref` as VMEM scratch carried
across a sequential grid — precisely how JAX's own TPU paged-attention kernel is built.

Staged so a failure is localized rather than total:

1. Fixed-length blocks, static shapes, single page — parity gate.
2. Multi-page with a block table via `PrefetchScalarGridSpec(num_scalar_prefetch=...)`.
3. Variable lengths and empty pages via `@pl.when` plus iota masking, not an early-out branch.

```
+ src/flash_decode_pallas.py
+ tests/test_pallas_flash_decode.py
~ src/ops.py                        (a third dispatch arm alongside cuda / reference)
```

### D14 — Three-backend benchmark, losses included

Identical shapes, identical seeded input tensors, one JSON artifact, one README table.
Method and anti-cheat provisions in §7.

I expect Pallas to lose at least one comparison outright, and the benchmark is designed so a
loss is reportable rather than embarrassing: the hand-written CUDA kernel has register-level
control of dequantization that neither Triton nor Pallas exposes, and seven weeks of Pallas will
not out-tune three months of CUDA. **A published loss with an explanation is the result.** A
benchmark that only shows wins would mean I chose the shapes after seeing the numbers.

```
+ benchmarks/bench_three_backend.py
+ results/three_backend.json
~ README.md                         (results table, wins and losses together)
```

### D15 — The write-up

Where the Triton mental model transferred, where it actively misled, and what the JAX memory and
sharding model forced me to restructure — organized around the §5 hypothesis table, each row
marked confirmed, refuted, or unresolved.

**The table is published in §5 before the work starts, and every row gets a verdict afterward,
including the rows where I was wrong.** A retrospective written after the fact can always make
the author look prescient. A pre-registered one cannot.

```
+ docs/TRITON_TO_PALLAS.md
```

---

## 5. Pre-registered hypotheses

`transfers` = the intuition carries over. `restructure` = the concept survives, the mechanism
changes. `misleads` = holding the Triton intuition will actively cost me time.

| Construct in the existing kernel | Predicted Pallas TPU fate | Prediction |
|---|---|---|
| `tl.program_id(0)` over channels, `grid=(n_ch,)` | `pl.program_id(0)`, same grid — one-to-one | transfers |
| `for start in range(0, n_rows, BLOCK)` in-program | Must become a **grid dimension**; TPU unrolls in-kernel loops fully | **misleads** |
| `tl.load(ptr + offs*n_ch + c, mask=...)` pointer math | No pointers. `BlockSpec(index_map=...)` declares *which block*, not which address | **misleads** |
| Running min/max in registers across the row loop | VMEM scratch accumulator, init under `@pl.when(program_id == 0)` | restructure |
| `__shfl_down_sync` warp reduction (CUDA) | No warp concept at all; `jnp.max`/`jnp.sum` over a tile axis on the VPU | **misleads** |
| Per-warp `(m, l)` state + LSE merge across warps | `m_ref`/`l_ref`/`o_ref` in VMEM, carried across sequential grid steps | restructure |
| Manual shared-memory staging of `s_scale`, `s_zps` | Deleted. `BlockSpec` brings tiles into VMEM; the pipeline is declared, not written | transfers |
| `uint8` one INT4/byte, nibble packing for storage | Forced, not chosen: **Pallas TPU supports every int/uint type except `int4`** | transfers |
| Nibble packing along last axis (d128 → 64 bytes) | Halving the last axis breaks the 128-lane register tile; expect to pack along rows | restructure |
| `block_lens` at runtime; `if blen == 0: skip` | Scalar prefetch into SMEM + `@pl.when` + iota masking; no data-dependent branch | restructure |
| `qf.to(tl.int32)` as truncation-based half-up | `jnp.floor(x + 0.5).astype(jnp.uint8)` — same semantics, clearer | transfers |
| Launch tuning: `BLOCK=1024`, 128 threads, occupancy | No thread count exists to tune. Block shape in `BlockSpec` is the only knob | **misleads** |
| Per-channel reduction = reduce over rows | Reduces over leading axes — the *fast* direction on TPU. A free win | transfers |
| Grid programs independent, any order | False on TPU: the grid runs sequentially in lexicographic order | **misleads** |

Six of fourteen rows predict the Triton intuition is *worse than useless*. If that fraction turns
out much lower, the write-up says the port was easy and the model transferred — a real and
publishable finding about Pallas. If higher, it is a warning to the next person making the same
transition. **Both outcomes are results; neither is a failure.**

**The sharding question, deliberately scoped small.** One experiment, not a thesis: take the
finished single-device kernel, wrap it in `shard_map` over the KV-head axis on a Kaggle v5e-8,
and record what breaks. The interesting part is that per-channel quantization scales are computed
by a reduction over rows — intact under a head-axis split, corrupted under a channel-axis split.
Whether Pallas and JAX let me express that constraint, or merely let me violate it silently, is
the finding.

---

## 6. The numerical contract

The criterion as written — "matching the reference oracle to the tolerance the CUDA path already
meets" — is one sentence describing two different numbers.

**The ambiguity.** The CUDA path *achieves* 0.000% bin disagreement. The Triton path is
*permitted* <=1 bin of disagreement on under 1% of elements, because half-up and banker's rounding
genuinely differ at exact `.5` boundaries. Read one way the bar is "zero disagreement"; read the
other it is "under one percent." I will not get to pick after seeing my numbers.

**Resolution.** The pass threshold is the Triton contract, because the Pallas kernel uses the
same half-up rounding and inherits the same legitimate boundary disagreement. The *target* is the
CUDA outcome. Both are reported, always, as two separate numbers. Scales at `rtol 1e-4`,
zero-points at `rtol 1e-3 / atol 1e-3`, unchanged from the existing test file.

### The TPU precision floor, and the control that isolates it

**On TPU, float32 matmuls do not execute in float32.** The default precision is bfloat16;
`Precision.HIGHEST` does not mean true float32 either — it approximates it with six bfloat16
passes. The existing MAE of 3.1e-8 was measured on a GPU doing real float32 arithmetic.
Demanding that number from a TPU demands something the hardware does not offer.

The trap is invisible in a results table: I run the port, see MAE around 1e-6, and cannot tell
whether I wrote a subtly wrong kernel or a correct kernel on hardware with a different error
floor. Loosening the tolerance until it passes is how that ends badly.

**Proposed control.** Run the **identical Pallas kernel** twice — once under
`pallas_call(..., interpret=True)`, which executes it as ordinary JAX on CPU in real float32,
and once on TPU.

- Interpret-mode MAE at the oracle's tolerance → **the algorithm is right**; any TPU excess is
  hardware precision.
- Interpret-mode MAE already elevated → **the port has a bug**, and no precision flag will fix it.

Two runs of one kernel separate a defect from a floor. Published tolerances are per-device, with
the interpret-mode number always alongside — never one blended figure. On TPU the attention path
uses `preferred_element_type=jnp.float32` accumulation and `Precision.HIGHEST`; if MAE still
lands above the GPU figure, the write-up reports the achieved number with the six-pass
explanation attached, rather than a relaxed threshold with no explanation.

---

## 7. Benchmark methodology

The three backends cannot all run on one device, and the deprecated bridge that might have united
them is the thing §3 ruled out. Reporting "CUDA 2.84 ms on a T1000 vs Pallas *x* ms on a v5e" and
drawing a conclusion would be meaningless.

**The normalizer.** This kernel is memory-bandwidth-bound. The primary published metric is
**memory-bandwidth utilization** — bytes necessarily moved, over elapsed time, over the device's
peak bandwidth. MBU is comparable across silicon in a way milliseconds are not, and it is the
metric the state-of-the-art TPU kernel reports, so the numbers land next to a published
reference point instead of floating free.

| Rule | Applies to |
|---|---|
| Absolute ms compared **only within one device** | every table |
| MBU % is the cross-device metric; device + peak BW printed in every row | every table |
| Identical seeded input tensors, byte-for-byte, across backends | every run |
| Warm-up discarded, autotune caches purged, clocks locked where possible | GPU + TPU |
| Shape sweep fixed **before** any backend is timed | D14 |

**Shape sweep, fixed in advance.** Anchored on b8 x h32 x d128 x s2048, block_size 256, then swept
one axis at a time: seq in {512, 2048, 8192, 32768}, batch in {1, 8, 32}, head_dim in {64, 128}.
Sequence length is the axis that matters for a decode kernel and where I most expect Pallas to
lose at short lengths, before per-call dispatch overhead is amortized.

**Anti-cheat.** Any benchmark graded on wall-clock is game-able, including by its author,
including unintentionally. Three assertions run inside the harness:

- **The kernel actually ran.** Inspect the compiled HLO and assert the Mosaic custom-call is
  present — that XLA did not quietly replace my kernel with a fused library op. A Pallas kernel
  that gets optimized away benchmarks beautifully.
- **The output is consumed.** A result nothing reads can be eliminated as dead code.
- **No shape specialization.** Every configuration re-run at `seq_len + 1`; a kernel correct only
  at powers of two is not a kernel.

---

## 8. Risks

| Risk | Mitigation | If it lands |
|---|---|---|
| **R1 · TPU precision floor** — attention MAE cannot reach the GPU figure | Interpret-mode control (§6); float32 accumulation + `HIGHEST` | Report per-device tolerance with the six-pass explanation. Not a failure. |
| **R2 · Ragged paging is genuinely hard** — a research paper's worth of work | Staged in three gates (D13); fixed-length parity banked before ragged begins | Ship stages 1–2, document stage 3 as open. Claim no RPA parity. |
| **R3 · Nibble unpacking slow on TPU** — narrow types support only some slicing patterns | Benchmark packed and unpacked-`uint8` variants as separate rows | A measured result about INT4 on TPU, which is thinly documented. |
| **R4 · Free-tier TPU quota** — Kaggle caps at 20 h/mo, 9 h/day | Develop on Colab v5e-1 and interpret mode; reserve v5e-8 for measurement runs | Slower calendar, unchanged deliverables. |
| **R5 · Mosaic GPU leg does not happen** — requires rented Hopper | Declared a stretch from the outset; TPU is primary | Benchmark is CUDA + Triton on T1000, Pallas on v5e, normalized by MBU. |
| **R6 · The port is uneventful** — everything transfers, no story | Hypotheses pre-registered (§5) so "I was wrong" is scoreable | Publishable negative result: Pallas is easier to reach than expected. |
| **R7 · JAX GPU on Windows** — no native Windows CUDA wheels | WSL2 for local GPU work; TPU work is browser-hosted anyway | Setup cost in W1, already budgeted. |

---

## 9. Timeline

| Week | Work | Gate |
|---|---|---|
| **W1** | **Harness before kernels.** JAX on WSL2 and Colab/Kaggle TPU. Wire the existing NumPy oracle to a JAX-side test harness. Add the `interpret=True` CI job. Port a deliberately trivial Pallas kernel end-to-end first. | CI green on a CPU runner with a Pallas kernel in the loop |
| **W2** | **Quantizer, interpret mode.** D12 against the oracle in pure-JAX CPU execution. Row loop becomes the grid; VMEM accumulator for min/max. | Scales rtol 1e-4; bins <=1 apart on <1% of elements |
| **W3** | **Quantizer on real TPU.** Same kernel on v5e. Nibble-packing axis experiment: last-axis vs row-axis against the (8, 128) tile. | Same tolerance on-device; packing variants measured |
| **W4** | **Attention, stage 1.** D13 fixed-length, static-shape, single-page. Online-softmax state moves from warp registers to VMEM scratch across a sequential grid. | Output parity vs `online_softmax_ref`, both execution modes |
| **W5** | **Attention, stages 2–3.** Block table via scalar prefetch; variable lengths and empty pages via `@pl.when` and iota masking. Highest-risk week. | Existing edge-case tests pass unmodified against the Pallas backend |
| **W6** | **Benchmark.** D14 across the fixed sweep, with anti-cheat assertions. Rented-H100 Mosaic GPU leg attempted here if W1–W5 held. One `shard_map` experiment over the KV-head axis on v5e-8. | `results/three_backend.json` committed, losses included |
| **W7** | **Write-up.** D15. Score every §5 hypothesis. Publish the comparison, the benchmark table, and the distance from the RPA reference point. | `docs/TRITON_TO_PALLAS.md` merged; every row carries a verdict |

W5 decides the project's shape, so it sits with two weeks of slack behind it rather than at the
end. If ragged paging fails, W6 and W7 still produce a complete three-backend benchmark and a
complete write-up on the fixed-length kernel — a narrower result, delivered on time, gap named.

---

## 10. Success criteria

| # | Criterion | Verified by |
|---|---|---|
| **SC1** | **Quantizer numerics.** Scales `rtol 1e-4`, zero-points `rtol 1e-3 / atol 1e-3`, bins <=1 apart on <1% of elements — the identical contract `tests/test_triton_quantize.py` enforces today, re-pointed at the Pallas backend. Achieved disagreement rate published next to the threshold. | `pytest tests/test_pallas_quantize.py` |
| **SC2** | **Attention parity.** Output MAE within oracle tolerance in interpret mode, with the on-TPU figure published beside it and the precision-mode explanation. Variable-length and empty-page cases included. Per-device, never blended. | `pytest tests/test_pallas_flash_decode.py` |
| **SC3** | **Three-backend benchmark.** CUDA, Triton, Pallas on identical seeded inputs across the pre-fixed sweep, published with MBU and absolute latency, device named in every row — **and at least one case where Pallas loses, in the same table as the wins.** | `results/three_backend.json`, README |
| **SC4** | **The comparison document.** Every §5 hypothesis carries a verdict, including at least one row where my prediction was wrong — stated as wrong, not quietly dropped. | `docs/TRITON_TO_PALLAS.md` |

**What failure looks like.** SC1 and SC2 can fail on numerics — a real failure, reported as one.
SC3 cannot fail on *direction*; Pallas losing is expected, not a failed criterion. SC3 fails only
if the shapes were chosen after the numbers were seen, or if losses go unpublished. SC4 fails if
the hypothesis table comes back with a suspiciously perfect score.

---

## 11. Budget and hardware access

| Resource | Role | Access | Cost |
|---|---|---|---|
| NVIDIA T1000 8GB (Turing) | CUDA + Triton baselines, already measured | Local, owned | $0 |
| CPU, `interpret=True` | Development, CI, the precision control | Local + GitHub runners | $0 |
| Colab TPU v5e-1 | Day-to-day TPU development | Free tier | $0 |
| Kaggle TPU v5e-8 | Measurement runs, the `shard_map` experiment | Free, 20 h/mo · 9 h/day | $0 |
| Rented H100 | **Stretch:** Mosaic GPU leg (§9, W6) | On-demand, ~20 h | ~$60 |

Total cash exposure under $100, and the stretch leg is the only line item. Everything on the
critical path runs on hardware I already have or can reach for free. Quotas and free-tier
configurations are as advertised in August 2026 and change; the plan degrades to the Colab
single-chip path if Kaggle's allowance moves.

---

## 12. References

1. [Writing Mosaic GPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/gpu/reference.html)
2. [Pallas Changelog — Triton backend deprecation (JAX 0.11.0)](https://docs.jax.dev/en/latest/pallas/CHANGELOG.html)
3. [Writing TPU kernels with Pallas — tiling, dtypes, control flow, grid order](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
4. [Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU (arXiv:2604.15464)](https://arxiv.org/abs/2604.15464)
5. [JAX TPU paged-attention kernel source](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py)
6. [Matmul precision on TPU — `Precision.HIGHEST` as six bfloat16 passes](https://github.com/jax-ml/jax/discussions/35664)
7. [Free-tier TPU configurations and quotas, August 2026](https://dev.to/googleai/tpu-mythbusting-cost-and-usage-50ch)
8. [Base repository — ArchanaChetan07/int4-kv-flash-attention](https://github.com/ArchanaChetan07/int4-kv-flash-attention)
