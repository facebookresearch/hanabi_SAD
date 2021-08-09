# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
from typing import Dict, Tuple
import pprint
import argparse

import torch
import torch.nn as nn

lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
import utils


class LSTMNet(torch.jit.ScriptModule):
    def __init__(
        self,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        hand_size,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_lstm_layer = num_lstm_layer

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            self.hid_dim, self.hid_dim, num_layers=self.num_lstm_layer
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h0 = obs["h0"].transpose(0, 1).contiguous()
        c0 = obs["c0"].transpose(0, 1).contiguous()

        s = obs["s"].unsqueeze(0)
        assert s.size(2) == self.in_dim

        x = self.net(s)
        o, (h, c) = self.lstm(x, (h0, c0))
        a = self.fc_a(o).squeeze(0)

        return {
            "a": a,
            "h0": h.transpose(0, 1).contiguous(),
            "c0": c.transpose(0, 1).contiguous(),
        }


## main program ##
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()


device = "cuda"
state_dict = torch.load(args.model)
in_dim = state_dict["net.0.weight"].size()[1]
out_dim = state_dict["fc_a.weight"].size()[0]

print("after loading model")
search_model = LSTMNet(device, in_dim, 512, out_dim, 2, 5)
utils.load_weight(search_model, args.model, device)
save_path = args.model.rsplit(".", 1)[0] + ".sparta"
print("saving model to:", save_path)
torch.jit.save(search_model, save_path)
