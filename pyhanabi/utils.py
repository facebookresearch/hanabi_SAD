# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
from collections import OrderedDict
import json
import torch
import numpy as np

import r2d2
from create import *
import common_utils


def parse_first_dict(lines):
    config_lines = []
    open_count = 0
    for i, l in enumerate(lines):
        if l.strip()[0] == "{":
            open_count += 1
        if open_count:
            config_lines.append(l)
        if l.strip()[-1] == "}":
            open_count -= 1
        if open_count == 0 and len(config_lines) != 0:
            break

    config = "".join(config_lines).replace("'", '"')
    config = config.replace("True", 'true')
    config = config.replace("False", 'false')
    config = json.loads(config)
    return config, lines[i + 1 :]


def get_train_config(weight_file):
    log = os.path.join(os.path.dirname(weight_file), "train.log")
    if not os.path.exists(log):
        return None

    lines = open(log, "r").readlines()
    cfg, rest = parse_first_dict(lines)
    # net_size, _ = parse_first_dict(rest)
    # cfg.update(net_size)
    return cfg


def flatten_dict(d, new_dict):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_dict)
        else:
            new_dict[k] = v


def load_agent(weight_file, overwrite):
    """
    overwrite has to contain "device"
    """
    cfg = get_train_config(weight_file)
    assert cfg is not None

    if "core" in cfg:
        new_cfg = {}
        flatten_dict(cfg, new_cfg)
        cfg = new_cfg

    game = create_envs(
        1,
        1,
        cfg["num_player"],
        cfg["train_bomb"],
        [0], # explore_eps,
        [100], # boltzmann_t,
        cfg["max_len"],
        cfg["sad"] if "sad" in cfg else cfg["greedy_extra"],
        cfg["shuffle_obs"],
        cfg["shuffle_color"],
        cfg["hide_action"],
        True,
    )[0]

    config = {
        "vdn": overwrite["vdn"] if "vdn" in overwrite else cfg["method"] == "vdn",
        "multi_step": overwrite.get("multi_step", cfg["multi_step"]),
        "gamma": overwrite.get("gamma", cfg["gamma"]),
        "eta": 0.9,
        "device": overwrite["device"],
        "in_dim": game.feature_size(),
        "hid_dim": cfg["hid_dim"] if "hid_dim" in cfg else cfg["rnn_hid_dim"],
        "out_dim": game.num_action(),
        "num_lstm_layer": cfg.get("num_lstm_layer", overwrite.get("num_lstm_layer", 2)),
        "boltzmann_act": overwrite.get("boltzmann_act", cfg["boltzmann_act"]),
        "uniform_priority": overwrite.get("uniform_priority", False),
    }

    agent = r2d2.R2D2Agent(**config).to(config["device"])
    load_weight(agent.online_net, weight_file, config["device"])
    agent.sync_target_with_online()
    return agent, cfg


def log_explore_ratio(games, expected_eps):
    explore = []
    for g in games:
        explore.append(g.get_explore_count())
    explore = np.stack(explore)
    explore = explore.sum(0)  # .reshape((8, 10)).sum(1)

    step_counts = []
    for g in games:
        step_counts.append(g.get_step_count())
    step_counts = np.stack(step_counts)
    step_counts = step_counts.sum(0)  # .reshape((8, 10)).sum(1)

    factor = []
    for i in range(len(explore)):
        if step_counts[i] == 0:
            factor.append(1.0)
        else:
            f = expected_eps / max(1e-5, (explore[i] / step_counts[i]))
            f = max(0.5, min(f, 2))
            factor.append(f)
    print(">>>explore factor:", len(factor))

    explore = explore.reshape((8, 10)).sum(1)
    step_counts = step_counts.reshape((8, 10)).sum(1)

    print("exploration:")
    for i in range(len(explore)):
        ratio = 100 * explore[i] / step_counts[i]
        print(
            "\tbucket [%2d, %2d]: %5d, %5d, %2.2f%%"
            % (i * 10, (i + 1) * 10, explore[i], step_counts[i], ratio)
        )

    # print('timestep visit count:')
    # for i in range(len(step_counts)):
    #     print('\tbucket [%2d, %2d]: %.2f' % (i*10, (i+1)*10, 100 * step_counts[i]))

    for g in games:
        g.reset_count()

    return factor


class Tachometer:
    def __init__(self):
        self.num_act = 0
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self):
        self.t = time.time()

    def lap(self, actors, replay_buffer, num_train, factor):
        t = time.time() - self.t
        self.total_time += t
        num_act = get_num_acts(actors)
        act_rate = factor * (num_act - self.num_act) / t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        print(
            "Speed: train: %.1f, act: %.1f, buffer_add: %.1f, buffer_size: %d"
            % (train_rate, act_rate, buffer_rate, replay_buffer.size())
        )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Time: %s, %ds"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        print(
            "Total Sample: train: %s, act: %s"
            % (common_utils.num2str(self.num_train), common_utils.num2str(self.num_act))
        )

    def lap2(self, actors, num_buffer, num_train):
        t = time.time() - self.t
        self.total_time += t
        num_act = get_num_acts(actors)
        act_rate = (num_act - self.num_act) / t
        # num_buffer = replay_buffer.num_add()
        buffer_rate = (num_buffer - self.num_buffer) / t
        train_rate = num_train / t
        print(
            "Speed: train: %.1f, act: %.1f, buffer_add: %.1f"
            % (train_rate, act_rate, buffer_rate)
        )
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        print(
            "Total Time: %s, %ds"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        print(
            "Total Sample: train: %s, act: %s"
            % (common_utils.num2str(self.num_train), common_utils.num2str(self.num_act))
        )


def load_weight(model, weight_file, device):
    state_dict = torch.load(weight_file, map_location=device)
    source_state_dict = OrderedDict()
    target_state_dict = model.state_dict()
    for k, v in target_state_dict.items():
        if k not in state_dict:
            print("warning: %s not loaded" % k)
            state_dict[k] = v
    for k in state_dict:
        if k not in target_state_dict:
            # print(target_state_dict.keys())
            print("removing: %s not used" % k)
            # state_dict.pop(k)
        else:
            source_state_dict[k] = state_dict[k]

    # if "pred.weight" in state_dict:
    #     state_dict.pop("pred.bias")
    #     state_dict.pop("pred.weight")

    model.load_state_dict(source_state_dict)
    return


# def get_game_info(num_player, greedy_extra, feed_temperature, extra_args=None):
#     params = {"players": str(num_player)}
#     if extra_args is not None:
#         params.update(extra_args)
#     game = hanalearn.HanabiEnv(
#         params,
#         [0],
#         [],
#         -1,
#         greedy_extra,
#         False,
#         False,
#         False,
#         feed_temperature,
#         False,
#         False,
#     )

#     if num_player < 5:
#         hand_size = 5
#     else:
#         hand_size = 4

#     info = {
#         "input_dim": game.feature_size(),
#         "num_action": game.num_action(),
#         "hand_size": hand_size,
#         "hand_feature_size": game.hand_feature_size(),
#     }
#     # print(info)
#     return info


# def compute_input_dim(num_player):
#     hand = 126 * num_player
#     board = 76
#     discard = 50
#     last_action = 51 + 2 * num_player
#     card_knowledge = num_player * 5 * 35
#     return hand + board + discard + last_action + card_knowledge


# returns the number of steps in all actors
def get_num_acts(actors):
    total_acts = 0
    for actor in actors:
        if isinstance(actor, list):
            total_acts += get_num_acts(actor)
        else:
            total_acts += actor.num_act()
    return total_acts


# # num_acts is the total number of acts, so total number of acts is num_acts * num_game_per_actor
# # num_buffer is the total number of elements inserted into the buffer
# # time elapsed is in seconds
# def get_frame_stat(num_game_per_thread, time_elapsed, num_acts, num_buffer, frame_stat):
#     total_sample = (num_acts - frame_stat["num_acts"]) * num_game_per_thread
#     act_rate = total_sample / time_elapsed
#     buffer_rate = (num_buffer - frame_stat["num_buffer"]) / time_elapsed
#     frame_stat["num_acts"] = num_acts
#     frame_stat["num_buffer"] = num_buffer
#     return total_sample, act_rate, buffer_rate


def generate_explore_eps(base_eps, alpha, num_env):
    if num_env == 1:
        if base_eps < 1e-6:
            base_eps = 0
        return [base_eps]

    eps_list = []
    for i in range(num_env):
        eps = base_eps ** (1 + i / (num_env - 1) * alpha)
        if eps < 1e-6:
            eps = 0
        eps_list.append(eps)
    return eps_list


def generate_log_uniform(min_val, max_val, n):
    log_min = np.log(min_val)
    log_max = np.log(max_val)
    uni = np.linspace(log_min, log_max, n)
    uni_exp = np.exp(uni)
    return uni_exp.tolist()


@torch.jit.script
def get_v1(v0_joind, card_counts, ref_mask):
    v0_joind = v0_joind.cpu()
    card_counts = card_counts.cpu()

    batch, num_player, dim = v0_joind.size()
    num_player = 3
    v0_joind = v0_joind.view(batch, 1, num_player * 5, 25)

    mask = (v0_joind > 0).float()
    total_viable_cards = mask.sum()
    v1_old = v0_joind
    thres = 0.0001
    max_count = 100
    weight = 0.1
    v1_new = v1_old
    for i in range(max_count):  # can't use a variable count for tracing
        # torch.Size([256, 99, 25]) torch.Size([256, 99, 10, 25])
        # Calculate how many cards of what types are sitting in other hands.
        hand_cards = v1_old.sum(2)
        total_cards = card_counts - hand_cards
        # Exclude the cards I am holding myself.
        excluding_self = total_cards.unsqueeze(2) + v1_old
        # Negative numbers shouldn't happen, but they might (for all I know)
        excluding_self.clamp_(min=0)
        # Calculate unnormalised likelihood of cards: Adjusted count * Mask
        v1_new = excluding_self * mask
        # this is avoiding NaNs for when there are no cards.
        v1_new = v1_old * (1 - weight) + weight * v1_new
        v1_new = v1_new / (v1_new.sum(-1, keepdim=True) + 1e-8)
        # if False: # this is strictly for debugging / diagnostics
        #     # Normalise the diff by total viable cards.
        #     diff = (v1_new - v1_old).abs().sum() / total_viable_cards
        #     xent = get_xent(data, v1_old[:,:,:5,:])
        #     print('diff %8.3g  xent %8.3g' % (diff, xent))
        v1_old = v1_new

    return v1_new


@torch.jit.script
def check_v1(v0, v1, card_counts, mask):
    ref_v1 = get_v1(v0, card_counts, mask)
    batch, num_player, dim = v1.size()
    # print('v1:', v1.size())
    # print('v0:', v0.size())
    # print('ref_v1:', ref_v1.size())
    v1 = v1.view(batch, 1, 3 * 5, 25).cpu()
    # print('v1:', v1.size())
    # print('ref_v1:', ref_v1.size())
    print("diff: ", (ref_v1 - v1).max())
    if (ref_v1 - v1).max() > 1e-4:
        print((ref_v1 - v1)[0][0][0])
        assert False


def check_trajectory(batch):
    assert batch.obs["h"][0][0].sum() == 0
    length = batch.obs["h"][0].size(0)
    end = 0
    for i in range(length):
        t = batch.terminal[0][i]

        if end != 0:
            assert t

        if not t:
            continue

        if end == 0:
            end = i
    print("trajectory ends at:", end)
