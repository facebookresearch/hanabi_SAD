# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import set_path

set_path.append_sys_path()

import os
import pprint

import torch
import rela
import hanalearn

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


def create_train_env(
    method,
    seed,
    num_thread,
    num_game_per_thread,
    actor_cons,
    max_len,
    num_player,
    bomb,
    greedy_extra,
):
    assert method in ["vdn", "iql"]
    context = rela.Context()
    games = []
    actors = []
    threads = []
    print("training with bomb: %d" % bomb)
    for thread_idx in range(num_thread):
        env = rela.VectorEnv()
        for game_idx in range(num_game_per_thread):
            unique_seed = seed + game_idx + thread_idx * num_game_per_thread
            game = hanalearn.HanabiEnv(
                {
                    "players": str(num_player),
                    "seed": str(unique_seed),
                    "bomb": str(bomb),
                },
                max_len,
                greedy_extra,
                False,
            )
            games.append(game)
            env.append(game)

        assert max_len > 0
        if method == "vdn":
            # assert len(actor_cons) == 1
            actor = actor_cons(thread_idx)
            actors.append(actor)
            thread = hanalearn.HanabiVDNThreadLoop(actor, env, False)
        else:
            assert len(actor_cons) == num_player
            env_actors = []
            for i in range(num_player):
                env_actors.append(actor_cons[i](thread_idx))
            actors.extend(env_actors)
            thread = hanalearn.HanabiIQLThreadLoop(env_actors, env, False)

        threads.append(thread)
        context.push_env_thread(thread)
    print(
        "Finished creating environments with %d games and %d actors"
        % (len(games), len(actors))
    )
    return context, games, actors, threads


def create_eval_env(
    seed,
    num_thread,
    model_lockers,
    eval_eps,
    num_player,
    bomb,
    greedy_extra,
    log_prefix=None,
):
    context = rela.Context()
    games = []
    for i in range(num_thread):
        game = hanalearn.HanabiEnv(
            {"players": str(num_player), "seed": str(seed + i), "bomb": str(bomb),},
            -1,
            greedy_extra,
            False,
        )
        games.append(game)
        env = rela.VectorEnv()
        env.append(game)
        env_actors = []
        for j in range(num_player):
            env_actors.append(rela.R2D2Actor(model_lockers[j], 1, eval_eps))
        if log_prefix is None:
            thread = hanalearn.HanabiIQLThreadLoop(env_actors, env, True)
        else:
            log_file = os.path.join(log_prefix, "game%d.txt" % i)
            thread = hanalearn.HanabiIQLThreadLoop(env_actors, env, True, log_file)
        context.push_env_thread(thread)
    return context, games
