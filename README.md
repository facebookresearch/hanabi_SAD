# Other-Play & Simplified Action Decoder in Hanabi

## Important Update

The repo has been updated to include Other-Play, auxiliary task, as well as improved
training infrastructure. The build process has also been significantly simplfied. It
is no longer necessary to build pytorch from source and the code now works with newer
version of pytorch and cuda. It also avoids the hanging problem that may appear in
previous version of the codebase on certain hardware configuration.

## Intro

This repo contains code and models for
["Other-Play" for Zero-Shot Coordination](https://arxiv.org/abs/2003.02979)
and [Simplified Action Decoder for Deep Multi-Agent
Reinforcement Learning](https://arxiv.org/abs/1912.02288).

To reference this work, please use:

Other-Play
```
@misc{hu2020otherplay,
    title={"Other-Play" for Zero-Shot Coordination},
    author={Hengyuan Hu and Adam Lerer and Alex Peysakhovich and Jakob Foerster},
    year={2020},
    eprint={2003.02979},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
```

Simplfied Action Decoder
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
We have been using pytorch-1.5.1, cuda-10.1, and cudnn-v7.6.5 in our development environment.
Other setting may also work but we have not tested it extensively with different configurations.

```bash
# install pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install other dependencies
pip install numpy
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
cmake ..
make -j10
```

## Run

`hanabi/pyhanabi/tools` contains some example scripts to launch training
runs. `dev.sh` requires 2 gpus to run, 1 for training, 1 for simulation while
the rest require 3 gpus, 1 for training, 2 for simulation.

The important flags are `--sad 1` to enable `Simplified Action Decoder`,
`--pred_weight 0.25` to enable auxiliary task and multiply aux loss with 0.25,
and finally `--shuffle_color 1` to enable other-play.

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
