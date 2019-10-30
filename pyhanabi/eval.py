# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import numpy as np
import torch

from create_envs import create_eval_env
import rela
import iql_r2d2
import utils


def evaluate(
    model_lockers,
    num_game,
    seed,
    eval_eps,
    num_player,
    bomb,
    greedy_extra,
    *,
    log_prefix=None,
):
    context, games = create_eval_env(
        seed,
        num_game,
        model_lockers,
        eval_eps,
        num_player,
        bomb,
        greedy_extra,
        log_prefix,
    )
    context.start()
    while not context.terminated():
        time.sleep(0.5)

    context.terminate()
    while not context.terminated():
        time.sleep(0.5)
    scores = [g.get_episode_reward() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def evaluate_saved_model(
    weight_files, num_game, seed, bomb, num_run=1, log_prefix=None, verbose=True
):
    model_lockers = []
    greedy_extra = 0
    num_player = len(weight_files)
    assert num_player > 1, "1 weight file per player"

    for weight_file in weight_files:
        if verbose:
            print(
                "evaluating: %s\n\tfor %dx%d games" % (weight_file, num_run, num_game)
            )
        if (
            "GREEDY_EXTRA1" in weight_file
            or "sad" in weight_file
            or "aux" in weight_file
        ):
            player_greedy_extra = 1
            greedy_extra = 1
        else:
            player_greedy_extra = 0

        device = "cpu"
        game_info = utils.get_game_info(num_player, player_greedy_extra)
        input_dim = game_info["input_dim"]
        output_dim = game_info["num_action"]
        hid_dim = 512

        actor = iql_r2d2.R2D2Agent(1, 0.99, 0.9, device, input_dim, hid_dim, output_dim)
        state_dict = torch.load(weight_file)
        if "pred.weight" in state_dict:
            state_dict.pop("pred.bias")
            state_dict.pop("pred.weight")

        actor.online_net.load_state_dict(state_dict)
        model_lockers.append(rela.ModelLocker([actor], device))

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            model_lockers,
            num_game,
            num_game * i + seed,
            0,
            num_player,
            bomb,
            greedy_extra,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate
