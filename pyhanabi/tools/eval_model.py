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

import numpy as np
import torch
import r2d2
import utils
from eval import evaluate


def load_sad_model(weight_files, device):
    agents = []
    for weight_file in weight_files:
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        state_dict = torch.load(weight_file, map_location=device)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, False
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents


def load_op_model(method, idx1, idx2, device):
    """load op models, op models was trained only for 2 player
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # assume model saved in root/models/op
    folder = os.path.join(root, "models", "op", method)
    agents = []
    for idx in [idx1, idx2]:
        if idx >= 0 and idx < 3:
            num_fc = 1
            skip_connect = False
        elif idx >= 3 and idx < 6:
            num_fc = 1
            skip_connect = True
        elif idx >= 6 and idx < 9:
            num_fc = 2
            skip_connect = False
        else:
            num_fc = 2
            skip_connect = True
        weight_file = os.path.join(folder, f"M{idx}.pthw")
        if not os.path.exists(weight_file):
            print(f"Cannot find weight at: {weight_file}")
            assert False

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]
        agent = r2d2.R2D2Agent(
            False,
            3,
            0.999,
            0.9,
            device,
            input_dim,
            hid_dim,
            output_dim,
            2,
            5,
            False,
            num_fc_layer=num_fc,
            skip_connect=skip_connect,
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)
    return agents


def evaluate_legacy_model(agents, num_game, seed, bomb, device, num_run=1, verbose=True):
    num_player = len(agents)
    assert num_player > 1, "1 weight file per player"

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            True,  # in op paper, sad was a default
            device=device,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default="sad", type=str, help="sad/op")
    parser.add_argument("--num_game", default=5000, type=int)
    parser.add_argument(
        "--num_run", default=1, type=int, help="total num game = num_game * num_run"
    )
    # config for model from sad paper
    parser.add_argument("--weight", default=None, type=str)
    parser.add_argument("--num_player", default=None, type=int)
    # config for model from op paper
    parser.add_argument(
        "--method", default="sad-aux-op", type=str, help="sad-aux-op/sad-aux/sad-op/sad"
    )
    parser.add_argument("--idx1", default=1, type=int, help="which model to use?")
    parser.add_argument("--idx2", default=1, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)

    args = parser.parse_args()

    if args.paper == "sad":
        assert os.path.exists(args.weight)
        # we are doing self player, all players use the same weight
        weight_files = [args.weight for _ in range(args.num_player)]
        agents = load_sad_model(weight_files, args.device)
    elif args.paper == "op":
        agents = load_op_model(args.method, args.idx1, args.idx2, args.device)

    # fast evaluation for 5k games
    evaluate_legacy_model(
        agents, args.num_game, 1, 0, num_run=args.num_run, device=args.device
    )
