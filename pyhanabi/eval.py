# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import json
import numpy as np
import torch

from create import *
import rela
import r2d2
import utils


def evaluate(agents, num_game, seed, bomb, eps, sad, *, hand_size=5, runners=None, device="cuda:0"):
    """
    evaluate agents as long as they have a "act" function
    """
    assert agents is None or runners is None
    if agents is not None:
        runners = [rela.BatchRunner(agent, device, 1000, ["act"]) for agent in agents]
    num_player = len(runners)

    context = rela.Context()
    games = create_envs(
        num_game,
        seed,
        num_player,
        hand_size,
        bomb,
        [eps],
        -1,
        sad,
        False,
        False,
    )

    for g in games:
        env = hanalearn.HanabiVecEnv()
        env.append(g)
        actors = []
        for i in range(num_player):
            actors.append(rela.R2D2Actor(runners[i], 1))
        thread = hanalearn.HanabiThreadLoop(actors, env, True)
        context.push_env_thread(thread)

    for runner in runners:
        runner.start()

    context.start()
    while not context.terminated():
        time.sleep(0.5)
    context.terminate()
    while not context.terminated():
        time.sleep(0.5)

    for runner in runners:
        runner.stop()

    scores = [g.last_score() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def evaluate_saved_model(
    weight_files,
    num_game,
    seed,
    bomb,
    *,
    overwrite=None,
    num_run=1,
    verbose=True,
):
    agents = []
    sad = []
    hide_action = []
    if overwrite is None:
        overwrite = {}
    overwrite["vdn"] = False
    overwrite["device"] = "cuda:0"
    overwrite["boltzmann_act"] = False

    for weight_file in weight_files:
        agent, cfg = utils.load_agent(
            weight_file,
            overwrite,
        )
        agents.append(agent)
        sad.append(cfg["sad"] if "sad" in cfg else cfg["greedy_extra"])
        hide_action.append(bool(cfg["hide_action"]))

    hand_size = cfg.get("hand_size", 5)

    assert all(s == sad[0] for s in sad)
    sad = sad[0]
    if all(h == hide_action[0] for h in hide_action):
        hide_action = hide_action[0]
        process_game = None
    else:
        hide_actions = hide_action
        process_game = lambda g: g.set_hide_actions(hide_actions)
        hide_action = False

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p, _ = evaluate(
            agents,
            num_game,
            num_game * i + seed,
            bomb,
            0,  # eps
            sad,
            hide_action,
            process_game=process_game,
            hand_size=hand_size,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    if verbose:
        print("score: %f +/- %f" % (mean, sem), "; perfect: %.2f%%" % (100 * perfect_rate))
    return mean, sem, perfect_rate, scores
