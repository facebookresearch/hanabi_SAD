# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Dict
import common_utils


class R2D2Net(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = 2

        self.net = nn.Sequential(nn.Linear(self.in_dim, self.hid_dim), nn.ReLU())

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,  # , batch_first=True
        ).to(device)
        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        self.lstm.flatten_parameters()

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)

        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def duel(
        self, v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor
    ) -> torch.Tensor:
        assert a.size() == legal_move.size()
        legal_a = a * legal_move
        q = v + legal_a - legal_a.mean(2, keepdim=True)
        return q

    @torch.jit.script_method
    def act(
        self, s: torch.Tensor, legal_move: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert s.dim() == 2, "should be 2 [batch, dim], get %d" % s.dim()
        s = s.unsqueeze(0)
        x = self.net(s)
        o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        a = a.squeeze(0)
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert s.dim() == 4, "[seq_len, batch, num_player, dim]"

        seq_len, batch, num_player, dim = s.size()
        s = s.view(seq_len, batch * num_player, dim)
        legal_move = legal_move.view(seq_len, batch * num_player, self.out_dim)
        action = action.view(seq_len, batch * num_player)

        x = self.net(s)
        if len(hid) == 0:
            o, (h, c) = self.lstm(x)
        else:
            o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = self.duel(v, a, legal_move)

        # q: [seq_len, batch * num_player, num_action]
        # action: [seq_len, batch * num_player]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)
        qa = qa.view(seq_len, batch, num_player)
        sum_q = qa.sum(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch * num_player]
        greedy_action = legal_q.argmax(2).detach()
        greedy_action = greedy_action.view(seq_len, batch, num_player)
        return sum_q, greedy_action


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = ["multi_step", "gamma", "eta"]

    def __init__(self, multi_step, gamma, eta, device, in_dim, hid_dim, out_dim):
        super().__init__()
        self.online_net = R2D2Net(device, in_dim, hid_dim, out_dim)
        self.target_net = R2D2Net(device, in_dim, hid_dim, out_dim)
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device):
        cloned = R2D2Agent(
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self, s: torch.Tensor, legal_move: torch.Tensor, hid: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q, new_hid = self.online_net.act(s, legal_move, hid)
        legal_q = (1 + q - q.min()) * legal_move
        greedy_action = legal_q.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(
        self, obs: Dict[str, torch.Tensor], hid: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape [batchsize]
        """
        batchsize, num_player, dim = obs["s"].size()
        s = obs["s"].view(batchsize * num_player, dim)
        legal_move = obs["legal_move"].view(batchsize * num_player, -1)
        eps = obs["eps"].view(batchsize * num_player)

        greedy_action, new_hid = self.greedy_act(s, legal_move, hid)
        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(0), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()
        action = (greedy_action * (1 - rand) + random_action * rand).long()

        action = action.view(batchsize, num_player)
        greedy_action = greedy_action.view(batchsize, num_player)
        return (
            {"a": action.cpu().detach(), "greedy_a": greedy_action.cpu().detach()},
            new_hid,
        )

    @torch.jit.script_method
    def compute_priority(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,  # todo remove this?
        bootstrap: torch.Tensor,
        next_obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        next_hid: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        compute priority for one batch
        """
        batchsize, num_player, dim = obs["s"].size()
        s = obs["s"].unsqueeze(0)
        legal_move = obs["legal_move"].unsqueeze(0)
        assert action["a"].dim() == 2
        a = action["a"].unsqueeze(0)
        online_q = self.online_net(s, legal_move, a, hid)[0].squeeze(0)

        # computing next_q with double q-learning
        next_s = next_obs["s"]
        next_legal_move = next_obs["legal_move"]
        online_next_a, _ = self.greedy_act(
            next_s.view(batchsize * num_player, dim),
            next_legal_move.view(batchsize * num_player, -1),
            next_hid,
        )
        online_next_a = online_next_a.view(1, batchsize, num_player)
        bootstrap_q = self.target_net(
            next_s.unsqueeze(0), next_legal_move.unsqueeze(0), online_next_a, next_hid
        )[0].squeeze(0)

        assert reward.size() == bootstrap.size()
        assert reward.size() == bootstrap_q.size()
        target = (
            reward + bootstrap.float() * (self.gamma ** self.multi_step) * bootstrap_q
        )
        priority = (target - online_q).abs()
        return priority.cpu().detach()

    @torch.jit.script_method
    def aggregate_priority(
        self, priority: torch.Tensor, seq_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Given priority, compute the aggregated priority.
        Assumes priority is float Tensor of size [batchsize, seq_len]
        """
        mask = torch.arange(0, priority.size(0), device=seq_len.device)
        mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
        assert priority.size() == mask.size()
        priority = priority * mask

        p_mean = priority.sum(0) / seq_len
        p_max = priority.max(0)[0]
        agg_priority = self.eta * p_max + (1.0 - self.eta) * p_mean
        return agg_priority.cpu().detach()

    def _err(
        self,
        obs: Dict[str, torch.Tensor],
        hid: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: torch.Tensor,
        terminal: torch.Tensor,
        bootstrap: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        max_seq_len = obs["s"].size(0)
        s = obs["s"]
        legal_move = obs["legal_move"]
        action = action["a"]

        # flat_hid = {}
        # for key, val in hid.items():
        #     assert val.sum() == 0, val.sum()
        #     hd0, hd1, hd2, hd3 = val.size()
        #     assert hd1 == batchsize and hd2 == num_player
        #     flat_hid[key] = val.view(hd0, hd1 * hd2, hd3)

        # hid = flat_hid

        # hid is None for simplification
        hid = {}

        # this only works because the trajectories are padded,
        # i.e. no terminal in the middle
        online_qas, target_as = self.online_net(s, legal_move, action, hid)
        with torch.no_grad():
            target_qas, _ = self.target_net(s, legal_move, target_as, hid)

        terminal = terminal.float()
        bootstrap = bootstrap.float()

        errs = []
        for i in range(max_seq_len):
            target_i = i + self.multi_step
            target_qa = 0
            if target_i < max_seq_len:
                target_qa = target_qas[target_i]
            bootstrap_qa = (self.gamma ** self.multi_step) * target_qa
            target = reward[i] + bootstrap[i] * bootstrap_qa

            # sanity check
            should_padding = i >= seq_len
            if i > 0:
                is_padding = (terminal[i] + terminal[i - 1] == 2).float()
                assert (is_padding.long() == should_padding.long()).all()

            err = (target.detach() - online_qas[i]) * (1 - should_padding.float())
            errs.append(err)
        return torch.stack(errs, 0)

    def loss(self, batch):
        err = self._err(
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
        )
        loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        # sum over seq dim
        loss = loss.sum(0)
        seq_len, batchsize, num_player, _ = batch.obs["s"].size()
        p = err.abs()
        priority = self.aggregate_priority(p, batch.seq_len)
        return loss, priority
