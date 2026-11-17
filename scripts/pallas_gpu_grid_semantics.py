"""Reproduce the finding that makes compiled Pallas unsafe on GPU for this repo.

Both Pallas kernels accumulate across a grid axis in a resident output block.
This script runs that pattern in isolation -- sum 32 blocks into one output --
under every Pallas GPU lowering path, and reports whether the result is right.

Measured on an NVIDIA H100 80GB HBM3, JAX 0.12.0.1 (results/pallas_gpu_grid_semantics_h100.txt):
  Mosaic GPU, any dimension_semantics  -> AssertionError (no pallas_call lowering rule
                                          is registered for Mosaic GPU in this release)
  Triton backend (the default)         -> max error 97.0: grid programs run in
                                          parallel and the accumulator races

    python scripts/pallas_gpu_grid_semantics.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax._src.pallas.mosaic_gpu.pallas_call_registration  # noqa: F401  registers the lowering

print("jax", jax.__version__, jax.devices()[0].device_kind)


def report(label, fn):
    try:
        print(f"  {label:58s} {fn()}")
    except Exception as e:
        lines = [l for l in str(e).strip().splitlines() if l.strip()]
        print(f"  {label:58s} {type(e).__name__}: {(lines[0] if lines else '')[:150]}")


def double(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0


x = jnp.ones((512, 128), jnp.float32)
report("mosaic_gpu: trivial elementwise", lambda: "OK, correct=%s" % bool(jnp.all(
    pl.pallas_call(double, grid=(2,),
                   in_specs=[pl.BlockSpec((256, 128), lambda i: (i, 0))],
                   out_specs=pl.BlockSpec((256, 128), lambda i: (i, 0)),
                   out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                   compiler_params=plgpu.CompilerParams())(x) == 2.0)))


def accumulate(x_ref, o_ref):
    @pl.when(pl.program_id(0) == 0)
    def _init():
        o_ref[...] = jnp.zeros_like(o_ref)
    o_ref[...] += x_ref[...]


steps = 32
xs = (jnp.arange(steps * 128 * 128, dtype=jnp.float32).reshape(steps * 128, 128) % 7.0)
expected = xs.reshape(steps, 128, 128).sum(0)


def acc_err(**kw):
    out = pl.pallas_call(
        accumulate, grid=(steps,),
        in_specs=[pl.BlockSpec((128, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((128, 128), lambda i: (0, 0)),   # resident: ignores i
        out_shape=jax.ShapeDtypeStruct((128, 128), jnp.float32), **kw)(xs)
    err = float(jnp.abs(out - expected).max())
    return f"max error {err:.1f}  ->  {'CORRECT' if err == 0 else 'RACE / WRONG'}"


report("accumulator  mosaic_gpu  dimension_semantics=('sequential',)",
       lambda: acc_err(compiler_params=plgpu.CompilerParams(dimension_semantics=("sequential",))))
report("accumulator  mosaic_gpu  dimension_semantics=('parallel',)",
       lambda: acc_err(compiler_params=plgpu.CompilerParams(dimension_semantics=("parallel",))))
report("accumulator  triton backend (the default)", lambda: acc_err())
