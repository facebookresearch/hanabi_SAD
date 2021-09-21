import os
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Tuple, Dict


@torch.jit.script
def duel(v: torch.Tensor, a: torch.Tensor, legal_move: torch.Tensor) -> torch.Tensor:
    assert a.size() == legal_move.size()
    assert legal_move.dim() == 3  # seq, batch, dim
    legal_a = a * legal_move
    q = v + legal_a - legal_a.mean(2, keepdim=True)
    return q


class PublicLSTMNet(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "out_dim", "num_lstm_layer"]

    def __init__(self, device, in_dim, hid_dim, out_dim, num_lstm_layer):
        super().__init__()
        # for backward compatibility
        if isinstance(in_dim, int):
            assert in_dim == 783
            self.in_dim = in_dim
            self.priv_in_dim = in_dim - 125
            self.publ_in_dim = in_dim - 2 * 125
        else:
            self.in_dim = in_dim
            self.priv_in_dim = in_dim[1]
            self.publ_in_dim = in_dim[2]

        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_ff_layer = 1
        self.num_lstm_layer = num_lstm_layer

        self.priv_net = nn.Sequential(
            nn.Linear(self.priv_in_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
        )

        ff_layers = [nn.Linear(self.publ_in_dim, self.hid_dim), nn.ReLU()]
        for i in range(1, self.num_ff_layer):
            ff_layers.append(nn.Linear(self.hid_dim, self.hid_dim))
            ff_layers.append(nn.ReLU())
        self.publ_net = nn.Sequential(*ff_layers)

        self.lstm = nn.LSTM(
            self.hid_dim,
            self.hid_dim,
            num_layers=self.num_lstm_layer,
        ).to(device)
        self.lstm.flatten_parameters()

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.out_dim)

        # for aux task
        self.pred_1st = nn.Linear(self.hid_dim, 5 * 3)
        # self.pred_2nd = nn.Linear(self.hid_dim, 5 * 3)
        # self.pred_t = nn.Linear(self.hid_dim, 1)

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        shape = (self.num_lstm_layer, batchsize, self.hid_dim)
        hid = {"h0": torch.zeros(*shape), "c0": torch.zeros(*shape)}
        return hid

    @torch.jit.script_method
    def act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert priv_s.dim() == 2, "dim should be 2, [batch, dim], get %d" % priv_s.dim()

        bsize = hid["h0"].size(0)
        # hid size: [batch, num_player, num_layer, dim]
        # -> [num_layer, batch x num_player, dim]
        hid = {
            "h0": hid["h0"].flatten(0, 1).transpose(0, 1).contiguous(),
            "c0": hid["c0"].flatten(0, 1).transpose(0, 1).contiguous(),
        }

        priv_s = priv_s.unsqueeze(0)
        publ_s = publ_s.unsqueeze(0)

        x = self.publ_net(publ_s)
        publ_o, (h, c) = self.lstm(x, (hid["h0"], hid["c0"]))

        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        a = a.squeeze(0)

        # hid size: [num_layer, batch x num_player, dim]
        # -> [batch, num_player, num_layer, dim]
        hid_shape = (
            bsize,
            -1,
            self.num_lstm_layer,
            self.hid_dim,
        )
        h = h.transpose(0, 1).view(hid_shape)
        c = c.transpose(0, 1).view(hid_shape)

        hid = {"h0": h, "c0": c}
        return a, {"h0": h, "c0": c}

    @torch.jit.script_method
    def forward(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        action: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            priv_s.dim() == 3 or priv_s.dim() == 2
        ), "dim = 3/2, [seq_len(optional), batch, dim]"

        one_step = False
        if priv_s.dim() == 2:
            priv_s = priv_s.unsqueeze(0)
            publ_s = publ_s.unsqueeze(0)
            legal_move = legal_move.unsqueeze(0)
            action = action.unsqueeze(0)
            one_step = True

        x = self.publ_net(publ_s)
        if len(hid) == 0:
            publ_o, _ = self.lstm(x)
        else:
            publ_o, _ = self.lstm(x, (hid["h0"], hid["c0"]))
        priv_o = self.priv_net(priv_s)
        o = priv_o * publ_o
        a = self.fc_a(o)
        v = self.fc_v(o)
        q = duel(v, a, legal_move)

        # q: [seq_len, batch, num_action]
        # action: [seq_len, batch]
        qa = q.gather(2, action.unsqueeze(2)).squeeze(2)

        assert q.size() == legal_move.size()
        legal_q = (1 + q - q.min()) * legal_move
        # greedy_action: [seq_len, batch]
        greedy_action = legal_q.argmax(2).detach()

        if one_step:
            qa = qa.squeeze(0)
            greedy_action = greedy_action.squeeze(0)
            o = o.squeeze(0)
            q = q.squeeze(0)
        return qa, greedy_action, q, o

    def pred_loss_1st(self, lstm_o, target, hand_slot_mask, seq_len):
        return cross_entropy(self.pred_1st, lstm_o, target, hand_slot_mask, seq_len)


class R2D2Agent(torch.jit.ScriptModule):
    __constants__ = [
        "vdn",
        "multi_step",
        "gamma",
        "eta",
    ]

    def __init__(
        self,
        vdn,
        multi_step,
        gamma,
        eta,
        device,
        in_dim,
        hid_dim,
        out_dim,
        num_lstm_layer,
        greedy=False,
        nhead=None,
        nlayer=None,
        max_len=None
    ):
        super().__init__()
        self.online_net = PublicLSTMNet(
            device, in_dim, hid_dim, out_dim, num_lstm_layer
        ).to(device)
        self.target_net = PublicLSTMNet(
            device, in_dim, hid_dim, out_dim, num_lstm_layer
        ).to(device)

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.vdn = vdn
        self.multi_step = multi_step
        self.gamma = gamma
        self.eta = eta
        self.num_lstm_layer = num_lstm_layer
        self.greedy = greedy
        self.nhead = nhead
        self.nlayer = nlayer
        self.max_len = max_len

    @torch.jit.script_method
    def get_h0(self, batchsize: int) -> Dict[str, torch.Tensor]:
        return self.online_net.get_h0(batchsize)

    def clone(self, device, overwrite=None):
        if overwrite is None:
            overwrite = {}
        cloned = type(self)(
            overwrite.get("vdn", self.vdn),
            self.multi_step,
            self.gamma,
            self.eta,
            device,
            self.online_net.in_dim,
            self.online_net.hid_dim,
            self.online_net.out_dim,
            self.num_lstm_layer,
            self.greedy,
            nhead=self.nhead,
            nlayer=self.nlayer,
            max_len=self.max_len,
        )
        cloned.load_state_dict(self.state_dict())
        return cloned.to(device)

    def sync_target_with_online(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def greedy_act(
        self,
        priv_s: torch.Tensor,
        publ_s: torch.Tensor,
        legal_move: torch.Tensor,
        hid: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        adv, new_hid = self.online_net.act(priv_s, publ_s, hid)
        legal_adv = (1 + adv - adv.min()) * legal_move
        greedy_action = legal_adv.argmax(1).detach()
        return greedy_action, new_hid

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Acts on the given obs, with eps-greedy policy.
        output: {'a' : actions}, a long Tensor of shape
            [batchsize] or [batchsize, num_player]
        """
        priv_s = obs["priv_s"]
        priv_s = priv_s.squeeze(1)
        # assume input is the SAD version, we will do a hacky conversion
        assert priv_s.size(1) == 838
        priv_s = priv_s[:, :783]  # remove greedy action
        priv_s = priv_s[:, 125:]  # remove my hand (zero)
        publ_s = priv_s[:, 125:]  # remove partner's hand (non-zero)

        legal_move = obs["legal_move"].squeeze(1)
        if "eps" in obs:
            eps = obs["eps"].flatten(0, 1)
        else:
            eps = torch.zeros((priv_s.size(0),), device=priv_s.device)

        bsize, num_player = priv_s.size()[0], 1

        hid = {"h0": obs["h0"], "c0": obs["c0"]}

        greedy_action, new_hid = self.greedy_act(priv_s, publ_s, legal_move, hid)
        reply = {}

        random_action = legal_move.multinomial(1).squeeze(1)
        rand = torch.rand(greedy_action.size(), device=greedy_action.device)
        assert rand.size() == eps.size()
        rand = (rand < eps).long()

        if self.greedy:
            action = greedy_action
        else:
            action = (greedy_action * (1 - rand) + random_action * rand).detach().long()

        reply["a"] = action.unsqueeze(1).detach().cpu()
        reply["greedy_a"] = action.unsqueeze(1).detach().cpu()
        reply["h0"] = new_hid["h0"].detach().cpu()
        reply["c0"] =  new_hid["c0"].detach().cpu()
        return reply


obl_model = R2D2Agent(
    False,
    1,
    0.999,
    0.9,
    "cuda:0",
    (783, 658, 533),
    512,
    21,
    2,
)


def load_obl_model(model_file=None):
    if model_file is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_file = os.path.join(root, 'models', 'obl', 'obl.pthw')

    state_dict = torch.load(model_file)
    if "core_ffn.1.weight" in state_dict:
        state_dict.pop("core_ffn.1.weight")
        state_dict.pop("core_ffn.1.bias")
        state_dict.pop("core_ffn.3.weight")
        state_dict.pop("core_ffn.3.bias")
        state_dict.pop("pred_2nd.weight")
        state_dict.pop("pred_2nd.bias")
        state_dict.pop("pred_t.weight")
        state_dict.pop("pred_t.bias")

    obl_model.online_net.load_state_dict(state_dict)
    obl_model.sync_target_with_online()
    return obl_model
