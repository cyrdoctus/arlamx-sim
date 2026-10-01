"""Integer-only INT8 actor for SB3 PPO MLPs (per-channel int8 weights, int8 activations, int32 accumulators, tanh LUT)."""
from __future__ import annotations

import numpy as np


def layers(model):
    """[(W, b)] of the deterministic actor: mlp_extractor.policy_net (Linear + Tanh) then action_net."""
    import torch.nn as nn
    lin = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [model.policy.action_net]
    return [(m.weight.detach().cpu().numpy().astype(np.float64), m.bias.detach().cpu().numpy().astype(np.float64)) for m in lin]


def fp32(L, x):
    h = np.asarray(x, np.float32)
    for i, (W, b) in enumerate(L):
        h = (W.astype(np.float32) @ h + b.astype(np.float32)).astype(np.float32)
        if i < len(L) - 1:
            h = np.tanh(h)
    return np.clip(h, -1.0, 1.0)


class Int8Actor:
    """Symmetric quantization (Jacob et al. 2018, CVPR): x_q = round(x / s), per-feature input scales folded into
    layer 1, per-output-channel weight scales, hidden activations tanh-LUT to int8 at 1/127 (CMSIS-NN q7 style)."""

    def __init__(self, L, calib, pct=99.99):
        X = np.asarray(calib, np.float64)
        self.s_in = np.maximum(np.percentile(np.abs(X), pct, axis=0), 1e-6) / 127.0
        self.q = []
        h = X
        for i, (W, b) in enumerate(L):
            Wf = W * self.s_in[None, :] if i == 0 else W / 127.0
            s_w = np.maximum(np.abs(Wf).max(axis=1), 1e-12) / 127.0
            Wq = np.clip(np.round(Wf / s_w[:, None]), -127, 127).astype(np.int8)
            bq = np.round(b / s_w).astype(np.int32)
            pre = h @ W.T + b
            s_pre = max(np.percentile(np.abs(pre), pct), 1e-6) / 127.0 if i < len(L) - 1 else None
            self.q.append((Wq, bq, s_w, s_pre))
            h = np.tanh(pre)
        self.luts = [None if s is None else np.clip(np.round(np.tanh((np.arange(256) - 128) * s) * 127), -127, 127).astype(np.int8)
                     for (_, _, _, s) in self.q]

    def bytes(self):
        return int(sum(W.size + 4 * b.size + 4 * s.size + (256 if l is not None else 0)
                       for (W, b, s, _), l in zip(self.q, self.luts)) + 4 * self.s_in.size)

    def __call__(self, x):
        xq = np.clip(np.round(np.asarray(x, np.float64) / self.s_in), -127, 127).astype(np.int32)
        for i, (Wq, bq, s_w, s_pre) in enumerate(self.q):
            acc = Wq.astype(np.int32) @ xq + bq                       # int32 MAC
            if s_pre is None:
                return np.clip(acc * s_w, -1.0, 1.0).astype(np.float32)
            pq = np.clip(np.round(acc * (s_w / s_pre)), -128, 127).astype(np.int32)   # requantize
            xq = self.luts[i][pq + 128].astype(np.int32)              # tanh LUT -> int8 at 1/127
        return None
