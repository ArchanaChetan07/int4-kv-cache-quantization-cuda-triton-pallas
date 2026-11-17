"""Setup script for Flash Decoding INT4.

CPU-only by default. To build the CUDA extension (requires nvcc + PyTorch):

    FLASH_DECODE_FORCE_CUDA=1 pip install --no-build-isolation -e .

(--no-build-isolation so the extension compiles against the torch you have
installed, rather than a fresh copy in an isolated build environment.)
"""

import os

from setuptools import setup

ext_modules = []
cmdclass = {}

if os.environ.get("FLASH_DECODE_FORCE_CUDA"):
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="flash_decode_int4._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/flash_decode_int4.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="flash-decode-int4",
    version="0.1.0",
    description="INT4 KV-cache quantization + fused flash-decoding attention in CUDA, Triton and JAX/Pallas, validated against one NumPy oracle",
    author="Archana Suresh Patil",
    license="Apache-2.0",
    python_requires=">=3.10",
    # src/ IS the package: find_packages(where="src") would look for
    # subpackages inside it and find nothing, installing zero code
    packages=["flash_decode_int4"],
    package_dir={"flash_decode_int4": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
        ],
        "cuda": ["torch>=2.0.0"],
        "llama": [
            "transformers>=4.36.0",
            "datasets>=2.0.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
