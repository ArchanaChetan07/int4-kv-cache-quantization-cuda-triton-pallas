# Getting started

Three ways in, depending on the hardware you have.

## 1. Any machine, no GPU

```bash
git clone https://github.com/ArchanaChetan07/int4-kv-cache-quantization-cuda-triton-pallas.git
cd int4-kv-cache-quantization-cuda-triton-pallas
pip install -e ".[dev]"
pytest tests/ -q                                   # GPU/JAX-gated tests skip
python scripts/validate_llama.py --simulate        # hardware-free SNR gate
```

## 2. Pallas / JAX (still no GPU)

Pallas has no CPU code generator; its kernels run via `pallas_call(interpret=True)`.

```bash
pip install -e ".[pallas]"
pytest tests/test_pallas_quantize.py tests/test_pallas_flash_decode.py -q   # 21 passed
python benchmarks/bench_three_backend.py --quick --backends reference,pallas
```

On Windows, jaxlib newer than 0.4.38 may fail with `DLL load failed while importing
_jax`; pin `jax==0.4.38 jaxlib==0.4.38` in a separate virtualenv.

## 3. NVIDIA GPU (CUDA + Triton)

Needs nvcc and a CUDA-enabled PyTorch. On Windows, run from PowerShell or cmd —
nvcc's host-compiler subprocess fails under Git Bash/MSYS.

```powershell
$env:FLASH_DECODE_JIT_CUDA = "1"          # JIT-compiles csrc/ on import (cached)
pytest tests/ -q                          # 121 passed on a T1000
python benchmarks/bench_three_backend.py --backends reference,cuda,triton
```

## Where to read next

- [README.md](README.md) — results, scope and limitations
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — quantization scheme and kernel design
- [docs/TRITON_TO_PALLAS.md](docs/TRITON_TO_PALLAS.md) — the port write-up
- [docs/H100_RUNBOOK.md](docs/H100_RUNBOOK.md) — the rented-GPU run
