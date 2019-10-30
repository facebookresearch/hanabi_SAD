# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model


parser = argparse.ArgumentParser()
parser.add_argument("--weight", default=None, type=str, required=True)
parser.add_argument("--num_player", default=None, type=int, required=True)
args = parser.parse_args()

assert os.path.exists(args.weight)
# we are doing self player, all players use the same weight
weight_files = [args.weight for _ in range(args.num_player)]

# fast evaluation for 10k games
evaluate_saved_model(weight_files, 1000, 1, 0, num_run=10)
