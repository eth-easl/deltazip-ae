import os
import sys
import subprocess
from setuptools import setup, find_packages

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

common_setup_kwargs = {
    "version": "0.0.1.dev0",
    "name": "deltazip",
    "author": "Xiaozhe Yao",
    "description": "Serving LLMs",
    "long_description": "Serving LLMs",
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/eth-easl/deltazip",
    "keywords": ["large-language-models", "transformers"],
    "platforms": ["linux"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
}

PYPI_RELEASE = 0
BUILD_CUDA_EXT = int(os.environ.get("BUILD_CUDA_EXT", "0")) == 1

if BUILD_CUDA_EXT:
    try:
        import torch
    except:
        print(
            "Building cuda extension requires PyTorch(>=1.13.0) been installed, please install PyTorch first!"
        )
        sys.exit(-1)

    CUDA_VERSION = None

    default_cuda_version = torch.version.cuda
    CUDA_VERSION = "".join(
        os.environ.get("CUDA_VERSION", default_cuda_version).split(".")
    )

    if not CUDA_VERSION:
        print(
            f"Trying to compile deltazip for CUDA, but Pytorch {torch.__version__} "
            "is installed without CUDA support."
        )
        sys.exit(-1)

    # For the PyPI release, the version is simply x.x.x to comply with PEP 440.
    if not PYPI_RELEASE:
        common_setup_kwargs["version"] += f"+cu{CUDA_VERSION}"

include_dirs = ["deltazip/core/csrc"]
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

if BUILD_CUDA_EXT:
    from torch.utils import cpp_extension

    p = int(
        subprocess.run(
            "cat /proc/cpuinfo | grep cores | head -1",
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.split(" ")[2]
    )
    subprocess.call(
        [
            "python",
            "./deltazip/utils/qigen/generate.py",
            "--module",
            "--search",
            "--p",
            str(p),
        ]
    )

    from distutils.sysconfig import get_python_lib

    conda_cuda_include_dir = os.path.join(
        get_python_lib(), "nvidia/cuda_runtime/include"
    )

    print("conda_cuda_include_dir", conda_cuda_include_dir)
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)
        print(f"appending conda cuda include dir {conda_cuda_include_dir}")

    additional_setup_kwargs = {
        "cmdclass": {"build_ext": cpp_extension.BuildExtension},
    }

    common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    python_requires=">=3.8.0",
    **common_setup_kwargs,
)
