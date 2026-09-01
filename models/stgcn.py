"""
models/stgcn.py — Lightweight STGCN++ classifier for federated skeleton HAR.

Architecture mirrors pyskl's STGCN backbone (``gcn_adaptive='init'``,
``tcn_type='mstcn'``) but is self-contained (no mmcv / pyskl dependency)
so it can run inside any Flower client without installing the full stack.

Input tensor : (N, M, T, V, C)  with C=3, T=100, V=17, M=2  (pyskl convention)
Output tensor: (N, num_classes)  logits
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════ COCO-17 graph ═══════════════════════════════════

COCO_INWARD = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6),
    (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2),
]
COCO_NUM_JOINTS = 17
COCO_CENTER = 0  # nose


def _build_adjacency(num_node: int, edges: list, center: int) -> np.ndarray:
    """Build the 3-subset spatial adjacency used by ST-GCN.

    Returns A of shape (3, V, V) — [self-loops, inward, outward],
    each row-normalised.
    """

    def _normalize(mx: np.ndarray) -> np.ndarray:
        d = mx.sum(axis=0)
        d[d == 0] = 1
        return mx / d

    I = np.eye(num_node, dtype=np.float32)
    inward = np.zeros((num_node, num_node), dtype=np.float32)
    outward = np.zeros((num_node, num_node), dtype=np.float32)

    # BFS hop distance from center
    hop = np.full(num_node, 1e9, dtype=int)
    hop[center] = 0
    adj_list = {i: [] for i in range(num_node)}
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    queue = [center]
    while queue:
        cur = queue.pop(0)
        for nb in adj_list[cur]:
            if hop[nb] > hop[cur] + 1:
                hop[nb] = hop[cur] + 1
                queue.append(nb)

    for u, v in edges:
        if hop[u] < hop[v]:
            inward[v, u] = 1
        else:
            outward[u, v] = 1

    A = np.stack([_normalize(I), _normalize(inward), _normalize(outward)])
    return A  # (3, V, V)


def get_coco_adjacency() -> np.ndarray:
    return _build_adjacency(COCO_NUM_JOINTS, COCO_INWARD, COCO_CENTER)


# ═══════════════════════════ NTU-25 graph ════════════════════════════════════

NTU_INWARD = [
    (0, 1), (1, 20), (20, 2), (2, 3),           # spine
    (20, 4), (4, 5), (5, 6), (6, 7),             # left arm
    (7, 21), (7, 22),                             # left hand
    (20, 8), (8, 9), (9, 10), (10, 11),           # right arm
    (11, 23), (11, 24),                           # right hand
    (0, 12), (12, 13), (13, 14), (14, 15),        # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),        # right leg
]
NTU_NUM_JOINTS = 25
NTU_CENTER = 1  # spine base


def get_ntu_adjacency() -> np.ndarray:
    return _build_adjacency(NTU_NUM_JOINTS, NTU_INWARD, NTU_CENTER)


def get_adjacency(graph: str = 'coco') -> tuple:
    """Return (adjacency_matrix, num_joints) for the given graph type."""
    if graph == 'coco':
        return get_coco_adjacency(), COCO_NUM_JOINTS
    elif graph in ('ntu', 'ntu25', 'nturgb+d'):
        return get_ntu_adjacency(), NTU_NUM_JOINTS
    else:
        raise ValueError(f"Unknown graph type: {graph}")


# ═══════════════════════════ building blocks ═════════════════════════════════

class MsTCN(nn.Module):
    """Multi-scale temporal convolution (lighter variant of pyskl mstcn)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Tuple[int, ...] = (1, 2, 3, 4),
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid = out_channels // (len(dilations) + 2)  # +2 for maxpool + 1×1
        rem = out_channels - mid * (len(dilations) + 2)

        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, mid, 1),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, mid, (3, 1), stride=(stride, 1),
                          padding=(d, 0), dilation=(d, 1)),
                nn.BatchNorm2d(mid),
            ))
        # Max-pool branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(mid),
        ))
        # 1×1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, mid + rem, 1),
            nn.BatchNorm2d(mid + rem),
        ))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [br(x) for br in self.branches]
        # Align temporal dim (in case maxpool rounded differently)
        min_t = min(o.size(2) for o in outs)
        outs = [o[:, :, :min_t, :] for o in outs]
        return self.drop(torch.cat(outs, dim=1))


class UnitGCN(nn.Module):
    """Graph convolution unit with learnable adaptive adjacency (``init`` mode)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,  # (K, V, V)
        adaptive: bool = True,
        with_res: bool = True,
    ):
        super().__init__()
        K = A.size(0)
        self.num_subsets = K
        self.register_buffer("A", A)

        self.conv = nn.ModuleList()
        for _ in range(K):
            self.conv.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(A.clone())
        else:
            self.PA = None

        self.bn = nn.BatchNorm2d(out_channels)

        if with_res:
            if in_channels != out_channels:
                self.res = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.res = nn.Identity()
        else:
            self.res = None

        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        adj = self.A + self.PA if self.PA is not None else self.A
        out = 0
        for k in range(self.num_subsets):
            # Graph multiply:  (N, C_out, T, V)
            z = self.conv[k](x)  # (N, C_out, T, V)
            out = out + torch.einsum("nctv,vw->nctw", z, adj[k])

        out = self.bn(out)
        if self.res is not None:
            out = out + self.res(x)
        return F.relu(out, inplace=True)


class STGCNBlock(nn.Module):
    """Single spatial-temporal GCN block: GCN → MsTCN → residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        adaptive: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A,
                           adaptive=adaptive, with_res=True)
        self.tcn = MsTCN(out_channels, out_channels,
                         stride=stride, dropout=dropout)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.AvgPool2d((stride, 1)) if stride > 1 else nn.Identity(),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x), inplace=True)


# ═══════════════════════════ full model ══════════════════════════════════════

class STGCN(nn.Module):
    """Lightweight STGCN++ classifier for COCO-17 2D skeleton sequences.

    Parameters
    ----------
    num_classes : int
        Number of action classes.
    in_channels : int
        Coordinate channels (2 for x,y only; 3 for x,y,score — Demo 2 compat.).
    base_channels : int
        Width of the first GCN layer.
    num_stages : int
        Number of ST-GCN blocks (default 6, lighter than pyskl's 10).
    inflate_stages : list[int]
        Block indices where channel width doubles.
    down_stages : list[int]
        Block indices where temporal stride = 2.
    num_person : int
        Max persons (M dimension); averaged at the end.  Demo 2 uses M=2.
    adaptive : bool
        Use learnable adjacency (STGCN++ feature).
    dropout : float
        Dropout in temporal convolutions and classifier.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        base_channels: int = 64,
        num_stages: int = 6,
        inflate_stages: List[int] | None = None,
        down_stages: List[int] | None = None,
        num_person: int = 2,
        adaptive: bool = True,
        dropout: float = 0.0,
        graph: str = 'coco',
    ):
        super().__init__()
        if inflate_stages is None:
            inflate_stages = [3, 5]   # double at stages 3 and 5
        if down_stages is None:
            down_stages = [3, 5]

        self.num_person = num_person
        self.num_classes = num_classes

        adj_matrix, num_joints = get_adjacency(graph)
        self.num_joints = num_joints
        A = torch.tensor(adj_matrix, dtype=torch.float32)

        # Data batch-norm over (C*V)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        # Build GCN blocks
        channels = [base_channels]
        for i in range(1, num_stages):
            ch = channels[-1] * 2 if i in inflate_stages else channels[-1]
            channels.append(ch)

        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(num_stages):
            c_out = channels[i]
            stride = 2 if i in down_stages else 1
            self.blocks.append(
                STGCNBlock(c_in, c_out, A, stride=stride,
                           adaptive=adaptive, dropout=dropout))
            c_in = c_out

        self.out_channels = channels[-1]

        # Classification head  (mimics pyskl GCNHead)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

        self._init_fc()

    def _init_fc(self):
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / self.num_classes))
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, M, T, V, C)  — pyskl convention

        Returns
        -------
        logits : Tensor of shape (N, num_classes)
        """
        N, M, T, V, C = x.shape

        # Merge person dim into batch  →  (N*M, C, T, V)
        x = x.permute(0, 1, 4, 2, 3).contiguous().view(N * M, C, T, V)

        # Data BN: reshape to (N*M, C*V, T) → BN → back
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        # Now (N*M, C, T, V)

        for block in self.blocks:
            x = block(x)

        # Pool over T and V → (N*M, C_out, 1, 1)
        x = self.pool(x)
        x = x.view(N, M, self.out_channels)
        x = x.mean(dim=1)  # average over persons

        x = self.drop(x)
        return self.fc(x)


# ═══════════════════════════ helpers ═════════════════════════════════════════

def build_model(num_classes: int = 10, in_channels: int = 3, graph: str = 'coco', **kwargs) -> STGCN:
    """Convenience constructor matching the FL demo defaults."""
    return STGCN(num_classes=num_classes, in_channels=in_channels, graph=graph, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# quick smoke test
if __name__ == "__main__":
    model = build_model(num_classes=10, in_channels=3)
    print(f"STGCN  |  params: {count_parameters(model):,}")
    # (N, M, T, V, C) — pyskl convention
    dummy = torch.randn(2, 2, 100, 17, 3)
    out = model(dummy)
    print(f"Input {dummy.shape} → Output {out.shape}")
    assert out.shape == (2, 10)
    print("OK ✓")
