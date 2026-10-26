"""Backbone-side audio input embeddings — SEPARATE from the depth decoder's table.

Why separate: the depth table lives at depth d_model
and inside the depth FSDP group; reusing it for backbone inputs both breaks width
and reaches a sharded parameter without its all-gather hook. This module is owned
by the TTSModel root, at backbone width, with one extra learned row for the
position AFTER an audio span ends (the <aeos> input — class 2048 exists only as a
cb0 TARGET; as an INPUT it is this dedicated vector, never an embedding lookup).
"""
import math
import torch
import torch.nn as nn

from ..tokens import N_Q, N_CODES, AEOS_CLASS


class AudioInputEmbed(nn.Module):
    def __init__(self, d_backbone, n_q=N_Q, n_codes=N_CODES):
        super().__init__()
        self.n_q, self.n_codes = n_q, n_codes
        self.table = nn.Embedding(n_q * n_codes, d_backbone)
        self.aeos_in = nn.Parameter(torch.zeros(d_backbone))
        self.norm = nn.RMSNorm(d_backbone)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.normal_(self.aeos_in, std=0.02)

    def frames(self, codes, q_use):
        """codes (N, n_q) in [0, n_codes); use first q_use levels (level-dropout
        Q is sampled BEFORE the backbone sees the frame)."""
        if torch.any((codes[:, :q_use] < 0) | (codes[:, :q_use] >= self.n_codes)):
            raise ValueError("audio code outside [0, n_codes) reached the input "
                             "table — reader validation failed upstream")
        lvl = torch.arange(q_use, device=codes.device)
        e = self.table(codes[:, :q_use] + lvl * self.n_codes)
        return self.norm(e.sum(1) / math.sqrt(q_use))

    def aeos(self, n):
        return self.norm(self.aeos_in)[None, :].expand(n, -1)
