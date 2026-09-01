"""
fl/strategy_fsar.py — FSAR-inspired Flower strategy.

Three operating modes controlled by ``mode``:

1. ``fedavg``   — vanilla FedAvg (all params aggregated).
2. ``fedbn``    — FedBN: BatchNorm params are local; only "global" params
                  are sent / received / aggregated.
3. ``cluster``  — FedBN + periodic client clustering.  Every ``cluster_every``
                  rounds the server groups clients into ``num_clusters``
                  clusters using KMeans on client descriptors; each cluster
                  maintains its own global model.

All three modes log per-round metrics:
    global_acc, mean_client_acc, worst_client_acc
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger("fl.strategy_fsar")


# ─────────────────────── helpers ────────────────────────────────────────────

def _weighted_average(results: list[Tuple[int, np.ndarray]]) -> list[np.ndarray]:
    """Weighted (by num samples) average of parameter arrays."""
    total = sum(n for n, _ in results)
    if total == 0:
        return results[0][1]
    avg = []
    for i in range(len(results[0][1])):
        acc = np.zeros_like(results[0][1][i], dtype=np.float64)
        for n, params in results:
            acc += params[i].astype(np.float64) * n
        avg.append((acc / total).astype(results[0][1][i].dtype))
    return avg


def _kmeans(descriptors: np.ndarray, k: int, seed: int = 0,
            max_iter: int = 50) -> np.ndarray:
    """Minimal KMeans (avoid sklearn dependency)."""
    rng = np.random.RandomState(seed)
    n = len(descriptors)
    idx = rng.choice(n, k, replace=False)
    centres = descriptors[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(descriptors[:, None] - centres[None, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(k):
            members = descriptors[labels == c]
            if len(members) > 0:
                centres[c] = members.mean(axis=0)
    return labels


# ─────────────────────── main strategy ──────────────────────────────────────

class FSARStrategy(Strategy):
    """Flower Strategy supporting FedAvg / FedBN / Clustered modes."""

    def __init__(
        self,
        *,
        mode: str = "fedbn",           # "fedavg" | "fedbn" | "cluster"
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        num_clusters: int = 2,
        cluster_every: int = 5,        # re-cluster every N rounds
    ):
        assert mode in ("fedavg", "fedbn", "cluster")
        self.mode = mode
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters

        # Clustering state
        self.num_clusters = num_clusters
        self.cluster_every = cluster_every
        self.client_cluster: Dict[str, int] = {}       # cid → cluster_id
        self.cluster_params: Dict[int, Parameters] = {}  # cluster_id → params
        self.round_descriptors: Dict[str, np.ndarray] = {}

        # Logging
        self.history: list[dict] = []

        # Track last aggregated ndarrays for post-training model loading
        self.last_aggregated_ndarrays: Optional[list[np.ndarray]] = None

    # ── initialise ──

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.initial_parameters

    # ── configure fit ──

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[Tuple[ClientProxy, FitIns]]:
        num_clients = client_manager.num_available()
        sample_size = max(int(num_clients * self.fraction_fit),
                          self.min_fit_clients)
        clients = client_manager.sample(
            num_clients=min(sample_size, num_clients), min_num_clients=self.min_fit_clients
        )

        fit_configs: list[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            cid = client.cid
            if self.mode == "cluster" and cid in self.client_cluster:
                cluster_id = self.client_cluster[cid]
                params = self.cluster_params.get(cluster_id, parameters)
            else:
                params = parameters
            config: Dict[str, Scalar] = {"server_round": server_round}
            fit_configs.append((client, FitIns(params, config)))
        return fit_configs

    # ── aggregate fit ──

    def aggregate_fit(
        self,
        server_round: int,
        results: list[Tuple[ClientProxy, FitRes]],
        failures: list[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Collect per-client info
        client_accs = []
        for client, fit_res in results:
            acc = fit_res.metrics.get("train_acc", 0.0)
            client_accs.append(acc)

        # ── Clustering branch ──
        if self.mode == "cluster" and server_round % self.cluster_every == 0:
            # Collect descriptors (sent in metrics by the client)
            descriptors, cids_order = [], []
            for client, fit_res in results:
                desc = fit_res.metrics.get("descriptor", None)
                if desc is not None:
                    descriptors.append(np.frombuffer(
                        desc.encode("latin-1") if isinstance(desc, str) else desc,
                        dtype=np.float32))
                    cids_order.append(client.cid)

            if len(descriptors) >= self.num_clusters:
                desc_mat = np.stack(descriptors)
                labels = _kmeans(desc_mat, self.num_clusters)
                self.client_cluster = {cid: int(lab) for cid, lab
                                       in zip(cids_order, labels)}
                logger.info("Round %d — cluster assignment: %s",
                            server_round, self.client_cluster)

        # ── Aggregate (per-cluster or global) ──
        if self.mode == "cluster" and self.client_cluster:
            cluster_results: Dict[int, list] = defaultdict(list)
            for client, fit_res in results:
                cid = client.cid
                cluster_id = self.client_cluster.get(cid, 0)
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                cluster_results[cluster_id].append(
                    (fit_res.num_examples, ndarrays))

            for cluster_id, cr in cluster_results.items():
                agg = _weighted_average(cr)
                self.cluster_params[cluster_id] = ndarrays_to_parameters(agg)

            # Return first cluster params as "global" (Flower needs one)
            main_params = self.cluster_params.get(0, ndarrays_to_parameters(
                parameters_to_ndarrays(results[0][1].parameters)))
        else:
            # Standard weighted average
            weighted = [(fit_res.num_examples,
                         parameters_to_ndarrays(fit_res.parameters))
                        for _, fit_res in results]
            agg = _weighted_average(weighted)
            main_params = ndarrays_to_parameters(agg)

        # Store raw ndarrays for post-training model loading
        self.last_aggregated_ndarrays = parameters_to_ndarrays(main_params)

        metrics: Dict[str, Scalar] = {
            "mean_train_acc": float(np.mean(client_accs)) if client_accs else 0.0,
            "worst_train_acc": float(np.min(client_accs)) if client_accs else 0.0,
        }
        return main_params, metrics

    # ── configure evaluate ──

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate <= 0:
            return []
        num_clients = client_manager.num_available()
        sample_size = max(int(num_clients * self.fraction_evaluate),
                          self.min_evaluate_clients)
        clients = client_manager.sample(
            num_clients=min(sample_size, num_clients),
            min_num_clients=self.min_evaluate_clients)

        configs = []
        for client in clients:
            cid = client.cid
            if self.mode == "cluster" and cid in self.client_cluster:
                cluster_id = self.client_cluster[cid]
                params = self.cluster_params.get(cluster_id, parameters)
            else:
                params = parameters
            configs.append((client, EvaluateIns(params, {"server_round": server_round})))
        return configs

    # ── aggregate evaluate ──

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[Tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_loss, total_n = 0.0, 0
        accs = []
        for _, eval_res in results:
            total_loss += eval_res.loss * eval_res.num_examples
            total_n += eval_res.num_examples
            accs.append(eval_res.metrics.get("accuracy", 0.0))

        global_loss = total_loss / max(total_n, 1)
        metrics: Dict[str, Scalar] = {
            "global_acc": float(np.mean(accs)),
            "mean_client_acc": float(np.mean(accs)),
            "worst_client_acc": float(np.min(accs)),
        }

        self.history.append({
            "round": server_round,
            **metrics,
            "global_loss": global_loss,
        })
        logger.info(
            "Round %3d | global_acc %.4f | mean_acc %.4f | worst_acc %.4f",
            server_round, metrics["global_acc"],
            metrics["mean_client_acc"], metrics["worst_client_acc"],
        )

        return global_loss, metrics

    # ── evaluate (centralized, optional) ──

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # We rely on federated evaluation; skip centralized.
        return None
