# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch

from create_envs import create_train_env, create_eval_env
import vdn_r2d2
import iql_r2d2
import common_utils
import rela
from eval import evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="vdn")

    # game settings
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--greedy_extra", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument(
        "--batchsize", type=int, default=128,
    )
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 10)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    game_info = utils.get_game_info(args.num_player, args.greedy_extra)

    if args.method == "vdn":
        agent = vdn_r2d2.R2D2Agent(
            args.multi_step,
            args.gamma,
            0.9,
            args.train_device,
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        agent_cls = vdn_r2d2.R2D2Agent
    elif args.method == "iql":
        agent = iql_r2d2.R2D2Agent(
            args.multi_step,
            args.gamma,
            0.9,
            args.train_device,
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        agent_cls = iql_r2d2.R2D2Agent

    # eval is always in IQL fashion
    eval_agents = []
    eval_lockers = []
    for _ in range(args.num_player):
        ea = iql_r2d2.R2D2Agent(
            1,
            0.99,
            0.9,
            "cpu",
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        locker = rela.ModelLocker([ea], "cpu")
        eval_agents.append(ea)
        eval_lockers.append(locker)

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    ref_models = []
    model_lockers = []
    act_devices = args.act_device.split(",")
    for act_device in act_devices:
        ref_model = [agent.clone(act_device) for _ in range(3)]
        ref_models.extend(ref_model)
        model_locker = rela.ModelLocker(ref_model, act_device)
        model_lockers.append(model_locker)

    actor_eps = utils.generate_actor_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_thread
    )
    print("actor eps", actor_eps)

    if args.method == "vdn":
        actor_cons = lambda thread_idx: rela.R2D2Actor(
            model_lockers[thread_idx % len(model_lockers)],
            args.multi_step,
            args.num_game_per_thread,
            args.gamma,
            args.max_len,
            actor_eps[thread_idx],
            args.num_player,
            replay_buffer,
        )
    elif args.method == "iql":
        actor_cons = []
        for _ in range(args.num_player):
            actor_cons.append(
                lambda thread_idx: rela.R2D2Actor(
                    model_lockers[thread_idx % len(model_lockers)],
                    args.multi_step,
                    args.num_game_per_thread,
                    args.gamma,
                    args.max_len,
                    actor_eps[thread_idx],
                    1,
                    replay_buffer,
                )
            )

    context, games, actors, threads = create_train_env(
        args.method,
        args.seed,
        args.num_thread,
        args.num_game_per_thread,
        actor_cons,
        args.max_len,
        args.num_player,
        args.train_bomb,
        args.greedy_extra,
    )

    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    for epoch in range(args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()

        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                for model_locker in model_lockers:
                    model_locker.update_model(agent)

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            weight = weight.to(args.train_device).detach()
            loss, priority = agent.loss(batch)
            loss = (loss * weight).mean()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()
            replay_buffer.update_priority(priority)

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        tachometer.lap(
            actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        )
        stat.summary(epoch)

        context.pause()
        for i in range(args.num_player):
            eval_lockers[i].update_model(agent)
        eval_seed = (args.seed + epoch * 1000) % 7777777
        score, perfect, _, _ = evaluate(
            eval_lockers,
            1000,
            eval_seed,
            0,
            args.num_player,
            args.eval_bomb,
            args.greedy_extra,
        )
        model_saved = saver.save(agent, agent.online_net.state_dict(), score)
        print(
            "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
            % (epoch, score, perfect * 100, model_saved)
        )
        context.resume()
        print("==========")
