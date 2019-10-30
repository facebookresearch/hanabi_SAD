# Simplified Action Decoder in Hanabi

This repo contains code and models for [Simplified Action Decoder for
Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1912.02288).

To reference this work, please use:
```
@misc{hu2019simplified,
    title={Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning},
    author={Hengyuan Hu and Jakob N Foerster},
    year={2019},
    eprint={1912.02288},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

## Compile

### Prerequisite

Install `cudnn7`, `cuda9.2` and `gcc7`. This might be
platform dependent. Other versions might also work but we have only
tested with the above versions. Note that we discovered a deadlock
problem when using tensors with C++ multi-threading when using
`cuda10.0` on Pascal GPU.

### Build PyTorch from Source

Create a fresh conda env & **compile PyTorch** from source.
If PyTorch and this repo are compiled by compilers with
different ABI compatibility, mysterious bugs that unexpectedly corrupt memory
may occur. To avoid that, the current solution is to
compile & install PyTorch from source first and then compile
this repo against that PyTorch binary.
For convenience, we paste instructions of compling PyTorch here.

```bash
# create a fresh conda environment with python3
conda create --name [your env name] python=3.7
conda activate [your env name]

conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda92

# clone pytorch
git clone -b v1.3.0 --recursive https://github.com/pytorch/pytorch
cd pytorch

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# set cuda arch list so that the built binary can be run on both pascal and volta
TORCH_CUDA_ARCH_LIST="6.0;7.0" python setup.py install
```

### Additional dependencies

```bash
pip install tensorboardX
pip install psutil

# if the current cmake version is < 3.15
conda install -c conda-forge cmake
```

### Clone & Build this repo
For convenience, add the following lines to your `.bashrc`,
after the line of `conda activate xxx`.

```bash
# set path
CONDA_PREFIX=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CPATH=${CONDA_PREFIX}/include:${CPATH}
export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# avoid tensor operation using all cpu cores
export OMP_NUM_THREADS=1
```

Clone & build.
```bash
git clone --recursive https://github.com/facebookresearch/hanabi.git

cd hanabi
mkdir build
cd build
# next line may produce an error 'Target "torch" not found.',
# which can be ignored
cmake ..
make -j10
```

## Run

`hanabi/pyhanabi/tools` contains some example scripts to launch training
runs. `dev.sh` requires 2 gpus to run, 1 for training, 1 for simulation while
the rest require 3 gpus, 1 for training, 2 for simulation.

```bash
cd pyhanabi
sh tools/dev.sh
```

## Trained Models

Run the following command to download the trained models used to
produce tables in the paper.
```bash
cd model
sh download.sh
```
To evaluate a model, simply run
```bash
cd pyhanabi
python tools/eval_model.py --weight ../models/sad_2p_10.pthw --num_player 2
```

## Related Repos

The results on Hanabi can be further improved by running search on top
of our agents. Please refer to the [paper](https://arxiv.org/abs/1912.02318) and
[code](https://github.com/facebookresearch/Hanabi_SPARTA) for details.

We also open-sourced a single agent implementation of R2D2 tested on Atari
[here](https://github.com/facebookresearch/rela).

## Contribute

### Python
Use [`black`](https://github.com/psf/black) to format python code,
run `black *.py` before pushing

### C++
The root contains a `.clang-format` file that define the coding style of
this repo, run the following command before submitting PR or push
```bash
clang-format -i *.h
clang-format -i *.cc
```

## Copyright
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
