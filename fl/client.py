"""
fl/client.py — Flower NumPyClient for skeleton-based HAR.

Each client owns a local NPZ shard produced by ``prepare_clients.py`` and
trains a lightweight STGCN model.  Supports:
  • Standard FedAvg (all parameters shipped to/from server).
  • FedBN / FSAR mode (BatchNorm parameters kept local; only "global" params
    are exchanged).  Toggled via ``local_bn`` flag.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar

# Allow importing sibling package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.stgcn import build_model  # noqa: E402


# ───────────────────── helpers ──────────────────────────────────────────────

def _is_bn_param(name: str) -> bool:
    """Return True if a parameter/buffer name belongs to a BatchNorm layer."""
    return "bn" in name.lower() or "data_bn" in name


def _split_param_names(model: nn.Module) -> Tuple[List[str], List[str]]:
    """Split state-dict keys into (global, local/BN)."""
    global_keys, local_keys = [], []
    for name in model.state_dict().keys():
        if _is_bn_param(name):
            local_keys.append(name)
        else:
            global_keys.append(name)
    return global_keys, local_keys


def _load_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


# ───────────────────── Flower NumPyClient ────────────────────────────────────

class SkeletonClient(fl.client.NumPyClient):
    """Flower client wrapping local STGCN training on one data shard."""

    def __init__(
        self,
        cid: int,
        data_path: str | Path,
        num_classes: int = 10,
        in_channels: int = 2,
        batch_size: int = 64,
        epochs_per_round: int = 1,
        lr: float = 0.01,
        device: str = "cpu",
        local_bn: bool = False,
        mu: float = 0.0,
        model_kwargs: dict | None = None,
    ):
        super().__init__()
        self.cid = cid
        self.device = torch.device(device)
        self.local_bn = local_bn
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.lr = lr
        self.mu = mu

        # Build model
        kw = model_kwargs or {}
        self.model = build_model(num_classes=num_classes,
                                 in_channels=in_channels, **kw).to(self.device)
        self.global_keys, self.local_keys = _split_param_names(self.model)

        # Load data
        x, y = _load_npz(data_path)
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 drop_last=False)

        self.criterion = nn.CrossEntropyLoss()

    # ── Flower interface ──

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return parameters that will be aggregated by the server."""
        sd = self.model.state_dict()
        if self.local_bn:
            return [sd[k].cpu().numpy() for k in self.global_keys]
        return [v.cpu().numpy() for v in sd.values()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Receive aggregated parameters from the server."""
        sd = self.model.state_dict()
        if self.local_bn:
            for k, v in zip(self.global_keys, parameters):
                sd[k] = torch.tensor(v)
        else:
            keys = list(sd.keys())
            for k, v in zip(keys, parameters):
                sd[k] = torch.tensor(v)
        self.model.load_state_dict(sd)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                    momentum=0.9, weight_decay=1e-4)

        # FedProx: snapshot global parameters for proximal term
        global_params = None
        if self.mu > 0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        total_loss, correct, total = 0.0, 0, 0
        for _ in range(self.epochs_per_round):
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                # FedProx proximal term: mu/2 * ||w - w_global||^2
                if self.mu > 0 and global_params is not None:
                    prox = sum(
                        torch.sum((p - gp) ** 2)
                        for p, gp in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (self.mu / 2) * prox

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

        metrics = {
            "train_loss": total_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            "cid": self.cid,
        }
        return self.get_parameters(config={}), total, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in self.loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                total_loss += self.criterion(logits, yb).item() * yb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)

        return (
            total_loss / max(total, 1),
            total,
            {"accuracy": correct / max(total, 1), "cid": self.cid},
        )

    # ── descriptor for clustering extension ──

    def get_descriptor(self) -> np.ndarray:
        """Return a vector descriptor for client clustering.

        Uses the last FC-layer weights flattened (small and informative).
        """
        return self.model.fc.weight.detach().cpu().numpy().flatten()
