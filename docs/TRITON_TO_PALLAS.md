# Triton → Pallas: what transferred, what misled

A port of the INT4 KV-cache quantizer and the fused flash-decoding attention
kernel from Triton/CUDA to Pallas on JAX, holding the algorithm and the
validation oracle fixed so that every difficulty is attributable to the tooling.

**Status of the evidence in this document.** Everything numerical below was
measured by running the code in this repository. The execution environment was
CPU via `pallas_call(interpret=True)`, because Pallas has no CPU code generator
and this machine's GPU (Turing T1000, SM 7.5) is below the floor of both live
Pallas GPU backends. Claims that require TPU silicon to test are marked
**UNRESOLVED** and are not asserted.

---

## 1. Scorecard against the pre-registered hypotheses

The predictions were written down before the port started (see
`PALLAS_PORT_PROPOSAL.md` §5). Verdicts are filled in here, including the one I
got wrong.

| # | Prediction | Verdict | What actually happened |
|---|---|---|---|
| 1 | `tl.program_id(0)` → `pl.program_id(0)`, one-to-one | **CONFIRMED** transfers | Identical concept, identical spelling. |
| 2 | In-kernel row loop must become a grid dimension | **CONFIRMED** misleads | Forced, though for a reason I had half-wrong — see §3.1. |
| 3 | Pointer arithmetic → `BlockSpec(index_map=...)` | **CONFIRMED** misleads | The single largest rewrite. See §3.2. |
| 4 | Running min/max → scratch accumulator under `pl.when` | **CONFIRMED** restructure | Right idea, different mechanism: a *resident output block*, not scratch. §3.3 |
| 5 | `__shfl_down_sync` has no analogue | **CONFIRMED** misleads | Vanished entirely. 42 warp/lane/shuffle/sync tokens in the CUDA kernel; 0 in the port. |
| 6 | Per-warp `(m,l)` + LSE merge → resident `m/l/o` refs | **CONFIRMED** restructure | The entire cross-warp merge epilogue disappeared. §3.4 |
| 7 | Manual shared-memory staging deleted | **CONFIRMED** transfers | `s_scale`/`s_zps` staging is gone; `BlockSpec` does it. |
| 8 | `uint8` forced because Pallas TPU has no `int4` | **UNRESOLVED** | Documented in JAX, untestable without TPU. Port uses `uint8` regardless. |
| 9 | Nibble packing must move off the last axis | **UNRESOLVED** | Needs the (8,128) register tiling to matter; invisible in interpret mode. |
| 10 | `block_lens` needs scalar prefetch | **REFUTED** | A plain input `BlockSpec` worked. See §4 — this is the one I got wrong. |
| 11 | `qf.to(tl.int32)` → `jnp.floor(x+0.5)` | **CONFIRMED** transfers | Exact: 0.000% bin disagreement against the oracle. |
| 12 | No thread/occupancy dial exists | **CONFIRMED** misleads | Stronger than predicted — block size changes *nothing*. §3.5 |
| 13 | Per-channel reduction is the fast TPU direction | **UNRESOLVED** | No tiling model in interpret mode. |
| 14 | Grid steps are sequential, not independent | **CONFIRMED** misleads | Load-bearing: the whole accumulator design rests on it. §3.3 |

**10 confirmed, 1 refuted, 3 unresolved.** The three unresolved rows all require
TPU access and are the honest limit of what this port established.

---

## 2. Headline results

| Check | Result |
|---|---|
| Pallas quantizer vs NumPy oracle | **0.000% bin disagreement**, scales `rtol 1e-4` |
| Pallas attention, empty/ragged pages | correct, no NaN, exact zeros for an all-empty sequence |
| Pallas test suite | **21 passing** (8 quantizer, 13 attention) |
| Existing suite, unaffected | 48 passing in the base environment |
| Pallas attention MAE vs float64 truth, `head_dim=128` | **2.59e-07** |
| NumPy oracle MAE vs the same float64 truth | 9.26e-07 |
| Both, expressed against the float32 accumulation floor | 0.32x - 1.39x across platforms (see 5.1) |

The quantizer met the **CUDA target** (0.000% disagreement), not merely the
Triton threshold (≤1 bin on <1% of elements) that the proposal set as the pass
bar.

---

## 3. Where the mental model actively misled

### 3.1 The loop is the grid — but check *why*

Triton runs one program per channel and loops over rows inside it:

```python
c = tl.program_id(0)
for start in range(0, n_rows, BLOCK):        # loop lives INSIDE the program
    ...
```

I predicted this must become a grid dimension because Pallas TPU fully unrolls
in-kernel loops. That prediction produced the right code for a reason I could
not verify: on CPU there is no unrolling to observe. What *is* verifiable is
that the Pallas programming model pushes you there anyway — `BlockSpec` exists
to describe how the grid indexes the array, and writing a manual row loop means
declining the entire pipelining mechanism. The restructuring is forced by the
API's shape, independent of the TPU codegen argument.

**Practical consequence, unpredicted:** because the reduction must complete
across the whole grid before quantization can begin, the single fused Triton
kernel became **two `pallas_call`s**. Triton could fuse min/max and quantize
into one program precisely because it held the statistics in registers for the
lifetime of one channel. Pallas cannot, and the two-pass split is not a style
choice.

### 3.2 There are no pointers, and that is the biggest adjustment

This is the line that carries the most Triton muscle memory:

```python
x = tl.load(kv_ptr + offs * n_ch + c, mask=mask, other=float("inf"))
```

You compute an address, you mask it, you load it. The Pallas equivalent
computes nothing:

```python
in_specs=[pl.BlockSpec((block_r, n_ch), lambda i: (i, 0))]
```

You declare *which block* this grid step sees, and the block arrives as a ref.
Every instinct about coalescing, stride arithmetic and load masking is looking
for a place to go and finds none. This took the longest to internalize and
produced the fewest lines of code — which is the whole character of the
transition.

The `other=` argument has no equivalent either, and that matters (§5.1).

### 3.3 Accumulators are output blocks, not scratch

I predicted VMEM scratch under `pl.when`. The mechanism that actually works is
subtler and more elegant: give the output a `BlockSpec` whose `index_map`
**ignores the reduction axis**.

```python
out_specs=[
    pl.BlockSpec((1, 1, head_dim), lambda b, h, i: (b, h, 0)),   # note: no i
    ...
]
```

Every grid step for a given `(b, h)` maps to the same output block, so Pallas
keeps it resident and consecutive writes accumulate. This is only safe because
the grid runs **sequentially in lexicographic order** — hypothesis 14, and the
prediction that turned out to matter most. A Triton user's default assumption
that grid programs are independent and may run in any order is not merely
inaccurate here; holding it makes the correct design look like a race condition.

### 3.4 The cross-warp merge simply disappears

The CUDA kernel's ending is ~40 lines: publish per-warp `(m, l)` into shared
memory, `__syncthreads`, find `m_star` across warps, recombine with
log-sum-exp, gate against warps that saw no positions, normalize.

The Pallas equivalent is:

```python
@pl.when(i == n_blocks - 1)
def _normalize():
    denom = jnp.maximum(l_ref[..., 0], _L_FLOOR)
    o_ref[...] = o_ref[...] / denom[..., None]
```

There is nothing to merge because there were never multiple partial states —
the sequential grid gave one accumulator per `(b, h)`. An entire class of
CUDA-side complexity is not simplified, it is *absent*.

### 3.5 There is no dial to turn

`BLOCK=1024`, 128 threads, 4 warps, occupancy, register pressure — none of it
has a counterpart. `block_rows` is the only knob, and
`test_block_rows_invariance` asserts that changing it across 64/128/256/512/1024
produces **bit-identical output**. In Triton, `BLOCK` is a correctness-neutral
performance dial; in Pallas the analogous parameter is correctness-neutral *and*
(in interpret mode) performance-neutral. Whatever performance intuition is worth
having here has to be rebuilt from the tiling model, not carried over.

---

## 4. The prediction I got wrong

I predicted `block_lens` would require scalar prefetch —
`PrefetchScalarGridSpec(num_scalar_prefetch=...)` — because that is what JAX's
own TPU paged-attention kernel uses, and because data-dependent page lengths
look like exactly the thing that needs SMEM.

It did not. A plain input spec works:

```python
pl.BlockSpec((1,), lambda b, h, i: (i,)),   # block_lens, an ordinary input
```

and the length is read inside the kernel as `len_ref[0]`, then used for iota
masking. **Scalar prefetch is a performance mechanism, not a correctness
requirement.** It exists to get the block table into SMEM ahead of the pipeline
so DMA can be issued early — which is essential for a fast TPU kernel and
irrelevant to whether the kernel computes the right answer.

I had conflated "this is how the reference implementation does it" with "this is
required." Reading a production kernel for API vocabulary teaches you the
optimized form and hides the minimal form; the minimal form is what you want
while you are still establishing correctness.

---

## 5. Findings that were not on the list

### 5.1 The oracle is less accurate than the kernel it validates

This is the most consequential thing the port surfaced, and it invalidates part
of the original success criterion.

Measured against a float64 evaluation of the same mathematics:

| head_dim | NumPy oracle (f32) | Pallas (multiply+reduce) | Pallas (`dot_general`) |
|---:|---:|---:|---:|
| 64 | 2.13e-07 | 2.33e-07 | 6.83e-07 |
| 128 | **9.26e-07** | **2.59e-07** | 1.06e-06 |

At `head_dim=128` the Pallas kernel is 3.6× more accurate than the reference it
is being validated against. The predicted float32 accumulation floor,
`sqrt(D) · eps · |o| ≈ 8.0e-07`, matches the oracle's error almost exactly — so
the oracle is behaving normally; it is simply at the float32 noise floor.

**Correction, from CI.** I first gated this as *"the Pallas kernel must be at
least as accurate as the reference."* That passed locally and **failed in CI**,
and the failure is more interesting than the original claim:

| | `head_dim` 64 | `head_dim` 128 |
|---|---|---|
| NumPy reference | 2.13e-07 | 9.26e-07 |
| Pallas, jax 0.4.38 / Windows | 2.33e-07 | 2.59e-07 |
| Pallas, modern jax / Linux CI | **7.45e-07** | passes |

The reference is byte-identical on both platforms — it is NumPy, and
deterministic. Only the Pallas number moved, by 3.2×, because **XLA is free to
choose a different reduction order between versions.** Expressed against the
float32 floor, every one of those numbers lands between 0.32× and 1.39× of it:
all four are normal, and none is a defect.

So *which implementation is more accurate is not a property of either
implementation* — it is a property of the XLA build you happen to have. My
original gate encoded a platform accident as a numerical claim. The test now
asserts what is actually stable: both implementations sit within 3× of the
float32 accumulation floor. That still fails loudly for a real defect, which
misses by orders of magnitude rather than by 1.4×.

The general lesson is the one this section already argues, turned on my own
test: a tolerance that was never checked on a second toolchain is an assumption,
not a measurement. It took a Linux CI runner to find it.

The consequence: a criterion of the form *"Pallas attention MAE vs the reference
below 3.1e-08"* is **not satisfiable at this shape by any correct
implementation**, because the reference itself is 30× further from the truth
than that. Measuring agreement-with-the-oracle conflates the kernel's error with
the oracle's, and the two are not separable from that number alone.

`tests/test_pallas_flash_decode.py` therefore uses a float64 arbiter and asserts
the kernel is *at least as accurate as the reference against it* — shape
independent, falsifiable, and not satisfiable by loosening a threshold. The
proposal anticipated an ambiguity in this criterion; the real defect was worse
than the one predicted, and only building the thing exposed it.

### 5.2 The MXU path costs accuracy twice

`use_dot=True` routes the QK product through `dot_general` (the matrix unit on
TPU) instead of an elementwise multiply and reduce (the vector unit). The table
above shows it is **2.6–4× less accurate than multiply-and-reduce in genuine
float32 on CPU**, before TPU enters the picture at all — XLA's reduction is
better conditioned than a single dot product here.

On TPU this compounds: float32 `dot_general` is executed as multiple bfloat16
passes even at `Precision.HIGHEST`. So the obvious "use the matrix unit" move
pays an accuracy cost twice over, once for the dot formulation and once for the
hardware. Both paths are kept in the source, defaulted to the accurate one, and
the flag is documented rather than silently chosen.

### 5.3 The same bug class, a different shape of fix

The CUDA kernel guards empty pages with a branch:

```c
if (s_m[w] != -CUDART_INF_F) { l_star += s_l[w] * expf(s_m[w] - m_star); }
```

without which `exp(-inf - -inf)` is NaN. The identical hazard exists in Pallas,
but a per-lane branch is the wrong instrument. The fix is arithmetic:

```python
m_safe = jnp.where(jnp.isneginf(m_new), 0.0, m_new)
corr = jnp.exp(m_prev - m_safe)     # -inf - 0 = -inf → exp → 0
p    = jnp.exp(logits - m_safe)     # masked lanes → 0
```

Substituting a finite surrogate drives every exponential to exactly zero, which
is the arithmetically correct contribution of an empty page. Same defect, same
reasoning, entirely different remedy — the CUDA instinct ("add a guard") points
at a construct you should not reach for.

### 5.4 A failure mode with no Triton counterpart

Because the row loop *is* the grid, a broken accumulator manifests as
**block-size-dependent output**. `test_block_rows_invariance` exists for a bug
that cannot occur in the Triton version, where the loop is internal and the
block size cannot change the reduction's extent. Porting a kernel imports the
target model's failure modes along with its API.

### 5.5 Practical: Pallas will not run at all on this hardware

Two hard stops, both confirmed by running:

- `pallas_call(interpret=False)` on a CPU host raises **"Only interpret mode is
  supported on CPU backend."** There is no CPU code generator. Every timing in
  this repository's Pallas rows is interpret mode and is *not* a performance
  number.
- **jaxlib newer than 0.4.38 fails to load on this Windows machine** with
  `DLL load failed while importing _jax`, including in a clean virtualenv.
  0.4.38 is pinned. Versions 0.4.30/0.4.35/0.4.38 all load; 0.9.x and 0.10.x do
  not.

Combined with Mosaic GPU requiring Hopper/Blackwell and the Pallas Triton
backend being deprecated, the performance leg of this project is not obtainable
without TPU or rented Hopper access. That was predicted from documentation in
the proposal; it is now confirmed by execution.

---

## 6. Would I tell someone to make this transition?

Yes, with one warning and one correction to how I framed it.

**The warning:** budget the adjustment in §3.2, not the algorithm. The
mathematics ported in an afternoon. Unlearning pointer arithmetic took longer
than everything else combined, and the symptom is not confusion — it is writing
code that works but declines the entire pipelining mechanism, which you will not
notice until you look at performance you cannot yet measure.

**The correction:** I framed this as "where the Triton mental model transferred
and where it misled." The more useful axis turned out to be *what disappears*.
The warp machinery, the shared-memory staging, the cross-warp merge, the
occupancy tuning — these are not translated into Pallas equivalents. They cease
to be things. A transition guide organized around "here is the Pallas way to do
X" is the wrong shape for roughly half the work, because for that half the
answer is that X is no longer a thing you do.

**On what is not established here:** no TPU ran any of this. The three
unresolved hypotheses, every performance claim, and the entire question of
whether these kernels are *good* rather than merely *correct* remain open, and
nothing in this document should be read as answering them.

---

## Reproducing

```bash
python -m venv .venv-jax
.venv-jax/Scripts/python -m pip install "jax==0.4.38" "jaxlib==0.4.38" numpy pytest

.venv-jax/Scripts/python -m pytest tests/test_pallas_quantize.py tests/test_pallas_flash_decode.py -v
.venv-jax/Scripts/python benchmarks/bench_three_backend.py --quick --backends reference,pallas
```

On Linux or with an accelerator present, `pip install "jax[cpu]"` (or the TPU
wheel) works without the version pin; the pin is a Windows-specific workaround
documented in §5.5.
