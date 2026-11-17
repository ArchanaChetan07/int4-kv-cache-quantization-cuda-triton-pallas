"""Where the Pallas kernels in this repository may be compiled.

Both kernels accumulate across a grid axis in output blocks that Pallas keeps
resident (the index_map ignores that axis). That is only correct if the axis
executes sequentially:

- TPU (Mosaic TPU): the grid always runs in lexicographic order. Correct.
- Interpret mode: the grid runs as an ordinary sequential loop. Correct.
- GPU: measured on an NVIDIA H100 with JAX 0.12.0.1. `pallas_call` reaches
  only the Triton backend -- Mosaic GPU registers no `pallas_call` lowering
  rule in that release -- and Triton runs grid programs in parallel. The
  resident accumulator races: a 32-step accumulation test returned max error
  97-98 (scripts/pallas_gpu_grid_semantics.py), the quantizer compiled but put
  29-44% of bins wrong across runs -- races are nondeterministic -- and the
  attention kernel returned outputs off by up to 5.4, with no NaN to warn
  (results/pallas_compiled_on_gpu_h100.json).

So compiling for a GPU is refused with this explanation rather than allowed to
return plausible wrong answers. On GPU hosts the dispatch layer defaults to
interpret mode, which runs the kernel body as JAX ops on the GPU and is
correct. A GPU-native variant would move the reduction loop inside each
program -- the Triton decomposition -- and is future work.
"""

GPU_UNSUPPORTED = (
    "compiled Pallas is not supported on GPU for these kernels: they "
    "accumulate across a grid axis that must run sequentially, and JAX's "
    "Pallas GPU backend (Triton, in jax 0.12) runs grid programs in parallel. "
    "Measured on an H100: compiled anyway, the quantizer put 29-44% of bins "
    "wrong and attention was off by up to 5.4 "
    "(results/pallas_compiled_on_gpu_h100.json). Use interpret=True "
    "(correct, runs on the GPU as JAX ops), a TPU, "
    "or the CUDA / Triton backends."
)


def compiled_platform(interpret: bool) -> str:
    """Return the platform a compiled kernel would target, or raise.

    Empty string in interpret mode. Raises NotImplementedError when compiling
    for a platform on which the resident-accumulator design races.
    """
    if interpret:
        return ""
    import jax
    platform = jax.devices()[0].platform
    if platform != "tpu":
        raise NotImplementedError(f"{GPU_UNSUPPORTED} (platform: {platform})")
    return platform
