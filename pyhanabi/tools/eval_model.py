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
import pandas as pd
import torch
import r2d2
import utils
from eval import evaluate


def evaluate_legacy_model(
    weight_files, num_game, seed, bomb, num_run=1, verbose=True
):
    # model_lockers = []
    # greedy_extra = 0
    agents = []
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    for weight_file in weight_files:
        if verbose:
            print(
                "evaluating: %s\n\tfor %dx%d games" % (weight_file, num_run, num_game)
            )
        if "sad" in weight_file or "aux" in weight_file:
            sad = True
        else:
            sad = False

        device = "cuda:0"

        state_dict = torch.load(weight_file)
        input_dim = state_dict["net.0.weight"].size()[1]
        hid_dim = 512
        output_dim = state_dict["fc_a.weight"].size()[0]

        agent = r2d2.R2D2Agent(
            False, 3, 0.999, 0.9, device, input_dim, hid_dim, output_dim, 2, 5, False
        ).to(device)
        utils.load_weight(agent.online_net, weight_file, device)
        agents.append(agent)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,
            sad,
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
    parser.add_argument("--weight_1", default=None, type=str, nargs='+', required=True)
    parser.add_argument("--weight_2", default=None, type=str, nargs='+', required=True)
    parser.add_argument("--num_player", default=None, type=int, required=True)
    args = parser.parse_args()
    print("weights_1 is ", args.weight_1)
    print("weights_2 is ", args.weight_2)
    
    scores_arr = np.zeros([len(args.weight_1), len(args.weight_2)])
    sem_arr = np.zeros([len(args.weight_1), len(args.weight_2)])
    ag1_names, ag2_names = [], []

    ## check if everything in weights_1 exist
    for ag1 in args.weight_1:
        assert os.path.exists(ag1)
        ag1_names.append(ag1.split("/")[2].split(".")[0])

    ## check if everything in weights_2 exist
    for ag2 in args.weight_2:
        assert os.path.exists(ag2)
        ag2_names.append(ag2.split("/")[2].split(".")[0])
    
    print("ag1 names is ", ag1_names)
    print("ag2 names is ", ag2_names)

    for ag1_idx, ag1 in enumerate(args.weight_1):
        for ag2_idx, ag2 in enumerate(args.weight_2):
            ## we are doing cross player, the 2 players use different weights
            print("Current game is ", str(ag1_idx) + " vs " + str(ag2_idx))
            weight_files = [ag1, ag2]
            # # fast evaluation for 10k games
            mean, sem, _ = evaluate_legacy_model(weight_files, 1000, 1, 0, num_run=10)
            scores_arr[ag1_idx, ag2_idx] = mean
            sem_arr[ag1_idx, ag2_idx] = sem 


    scores_df = pd.DataFrame(data=scores_arr, index=ag1_names, columns=ag2_names)
    sem_df = pd.DataFrame(data=sem_arr, index=ag1_names, columns=ag2_names)

    scores_df.to_csv('scores_data.csv')
    sem_df.to_csv('sem_data.csv')


