# RTIndeX Experiments

This repository contains the code for the paper _RTIndeX: Exploiting Hardware-Accelerated GPU Raytracing for Database Indexing_, available at TBA (preprint at <https://arxiv.org/abs/2303.01139>).

The code builds upon Ingo Wald's SIGGRAPH course, available at <https://github.com/ingowald/optix7course>.

## Prerequisites

- Your **GPU must support NVIDIA RTX (hardware raytracing acceleration)**. For consumer GPUs, this applies to **NVIDIA RTX 2000, 3000, and 4000** series, but there are workstation and data-center GPUs that also support RTX.
- Your GPU should have at least **24 GB of VRAM**. Smaller sizes will probably still be sufficient for most experiments, but are not guaranteed to.
- Ensure your system runs **NVIDIA's 522.25 GPU driver** or newer.
- Your system should have at least **32 GB of main memory**.
- All experiments are designed to run on **64-bit Linux**. We successfully tested **Ubuntu 20.04, Ubuntu 22.04, and Arch**.

## Installation

- Install the **CUDA 12.0 toolkit** (at least 11.8 is required):
  - Option 1: Through your package manager (if available in the required version).
  - Option 2: Manually by following the instructions on the NVIDIA website <https://developer.nvidia.com/cuda-downloads>. Depending on your current system setup, you might have to take additional steps, such as purging previous driver versions or disabling `nouveau`.
  - After the installation, test whether the CUDA toolkit is part of the `PATH` by running `nvcc --version`. If `nvcc` is not found, add the `bin` subdirectory of the CUDA toolkit (typically `/opt/cuda/bin` or `/usr/local/cuda/bin`) to your `PATH`.
- Parts of the code also depend on NVIDIA's **CUB and Thrust libraries**, which are included in the CUDA toolkit by default.
- Install **OptiX 7.6**:
  - Download OptiX from <https://developer.nvidia.com/designworks/optix/download> (requires an NVIDIA account).
  - Run the installer script, preferably in your home directory.
  - This will produce a directory called `NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64` or similar.
  - Set the environment variable `OptiX_INSTALL_DIR` to this directory, or add the directory to your `PATH`.
- Make sure `/usr/bin/env python3` points to **Python 3.8** or higher.
- Install **CMake 3.20** or later.
- Install **any version of `gcc`** that is compatible with your CUDA toolkit.
- (OPTIONAL) If you want to plot the results, install the `pandas` and `matplotlib` Python packages.
- If your system contains multiple GPUs, set the `CUDA_VISIBLE_DEVICES` variable to force execution on a specific one.

## Troubleshooting

- If CMake is unable to find CUDA, try setting the following variables:
  - `PATH=<CUDA toolkit root>/bin:$PATH`
  - `CUDA_HOME=<CUDA toolkit root>`
  - `CUDA_BIN_PATH=<CUDA toolkit root>/bin`
  - `CUDACXX=<CUDA toolkit root>/bin/nvcc`
- If you see something like `optix failed with code 7801`, update or re-install your driver.
- If you receive an error about unsupported GPU architectures, update or re-install your CUDA toolkit.

## Reproducing the results

- Clone the project from `https://gitlab.rlp.net/juhenneb/rtindex`
- `cd` to the root directory of the project
- Run `start-design-experiments.sh` to run all design experiments. The results will be copied to the `results` directory.
- Run `start-comparison-experiments.sh` to run all comparison experiments. The results will be copied to the `results` directory.
- Run `plot.sh` to create all plots from the paper (in the corresponding subdirectories of the `results` directory).

## Implementation Details

The codebase consists of two parts, which use very different experimentation workflows.
In either case, the input dataset is generated on-the-fly, but in a deterministic fashion to ensure reproducibility.

The _design experiments_ are located in the `experiments` directory.
The input data and test setup are specified in `src/benchmark.py`, with data being generated in `src/main.cu`.
This part of the code makes heavy use of conditional preprocessor macros to achieve the best possible compile-time optimization.
Therefore, the macros have to be reconfigured before each experiment, followed by a re-compilation, both of which is automated via `src/benchmark.py`.

The _comparison experiments_ are located in the `index-prototype` directory.
The input data is specified in `src/benchmarks.cuh` and generated in `src/input_generation.h`.
Here, compiling once will be sufficient to include all benchmarks.

