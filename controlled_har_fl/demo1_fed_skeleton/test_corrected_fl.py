"""Corrected FL invariants; no assertions about research accuracy."""

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import corrected_fl as fl
from .models.stgcn import build_model
from .prepare_corrected_fl import assign_clients, class_contract, prepare


def toy_model():
    # BN intentionally has only a numeric path, reproducing the old filter's blind spot.
    return nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))


def toy_shard(offset=0., labels=(0, 1, 0, 1), prefix="train", subject_offset=0):
    return fl.Shard(torch.tensor([[1., 0., 2.], [2., 1., -1.], [0., 3., 1.], [-1., 1., 2.]]) + offset,
                    torch.tensor(labels), np.array([f"{prefix}-{i}" for i in range(4)]),
                    np.arange(4) + subject_offset)


class CorrectedFLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_backbone_batchnorm_state_is_identified(self):
        model = build_model(num_classes=15, dropout=.3)
        modules = [(name, module) for name, module in model.named_modules()
                   if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))]
        keys = fl.batchnorm_keys(model)
        self.assertEqual(len(modules), 79)
        self.assertEqual(len(keys), 395)
        self.assertIn("blocks.0.tcn.branches.0.1.weight", keys)
        self.assertIn("blocks.0.gcn.res.1.running_var", keys)
        self.assertNotIn("blocks.0.gcn.conv.0.weight", keys)
        self.assertTrue(keys <= model.state_dict().keys())

    def test_receive_preserves_each_clients_complete_bn_state(self):
        server = toy_model()
        clients = [copy.deepcopy(server), copy.deepcopy(server)]
        keys = fl.batchnorm_keys(server)
        for cid, model in enumerate(clients):
            with torch.no_grad():
                for key, tensor in model.state_dict().items():
                    if key in keys:
                        tensor.fill_(cid + 2)
        before = [fl.state_copy(model) for model in clients]
        shared = fl.state_copy(server)
        shared["1.weight"].fill_(19)
        for cid, model in enumerate(clients):
            fl.receive_shared(model, shared, local_bn=True)
            self.assertTrue(torch.equal(model.state_dict()["1.weight"], shared["1.weight"]))
            for key in keys:
                self.assertTrue(torch.equal(model.state_dict()[key], before[cid][key]))
        self.assertFalse(torch.equal(clients[0][0].running_mean, clients[1][0].running_mean))

    def test_aggregation_uses_training_size_and_excludes_bn(self):
        base = {"weight": torch.tensor([0.]), "0.running_mean": torch.tensor([7.]),
                "0.num_batches_tracked": torch.tensor(0)}
        first = {"weight": torch.tensor([1.]), "0.running_mean": torch.tensor([2.]),
                 "0.num_batches_tracked": torch.tensor(2)}
        second = {"weight": torch.tensor([9.]), "0.running_mean": torch.tensor([8.]),
                  "0.num_batches_tracked": torch.tensor(5)}
        excluded = {"0.running_mean", "0.num_batches_tracked"}
        updates = [{key: value for key, value in state.items() if key not in excluded}
                   for state in (first, second)]
        result = fl.aggregate(base, updates, [3, 1], excluded)
        self.assertEqual(result["weight"].item(), 3.)
        self.assertEqual(result["0.running_mean"].item(), 7.)
        ordinary = fl.aggregate(base, [first, second], [3, 1])
        self.assertEqual(ordinary["0.running_mean"].item(), 3.5)
        self.assertEqual(ordinary["0.num_batches_tracked"].item(), 5)

    def test_local_update_does_not_mutate_other_client(self):
        clients = [toy_model(), toy_model()]
        before = fl.state_copy(clients[1])
        fl.train_client(clients[0], toy_shard(), device="cpu", batch_size=4,
                        epochs=1, lr=.01, seed=42)
        for key, tensor in clients[1].state_dict().items():
            self.assertTrue(torch.equal(tensor, before[key]))

    def test_heldout_scoring_never_updates_parameters_or_bn(self):
        model = toy_model()
        before = fl.state_copy(model)
        heldout = toy_shard(offset=100., prefix="test", subject_offset=100)
        matrix, logits = fl.score_client(model, heldout, device="cpu", batch_size=4, num_classes=2)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertEqual(matrix.sum(), len(heldout))
        np.testing.assert_array_equal(matrix.sum(1), np.bincount(heldout.y.numpy(), minlength=2))
        np.testing.assert_array_equal(matrix.sum(0), np.bincount(logits.argmax(1), minlength=2))

    def test_fedbn_persists_between_rounds_and_test_does_not_select_weights(self):
        train = [toy_shard(), toy_shard(offset=8., prefix="client1", subject_offset=10)]
        test = [toy_shard(prefix="test0", subject_offset=100),
                toy_shard(prefix="test1", subject_offset=200)]
        kwargs = dict(method="fedbn", rounds=2, local_epochs=1, batch_size=4,
                      lr=.01, seed=42, device="cpu", num_classes=2)
        incoming = []
        original = fl.train_client

        def record(model, shard, **settings):
            incoming.append(fl.state_copy(model))
            return original(model, shard, **settings)

        with patch.object(fl, "train_client", side_effect=record):
            result, predictions, checkpoint = fl.run_method(toy_model, train, test, **kwargs)
        self.assertEqual(incoming[0]["0.num_batches_tracked"].item(), 0)
        self.assertEqual(incoming[2]["0.num_batches_tracked"].item(), 1)
        self.assertEqual(incoming[3]["0.num_batches_tracked"].item(), 1)
        self.assertFalse(torch.equal(incoming[2]["0.running_mean"], incoming[3]["0.running_mean"]))
        changed_test = [copy.deepcopy(shard) for shard in test]
        for shard in changed_test:
            shard.y[:] = 1
        _, _, changed_checkpoint = fl.run_method(toy_model, train, changed_test, **kwargs)
        for original_client, changed_client in zip(checkpoint["clients"], changed_checkpoint["clients"]):
            for key in original_client:
                self.assertTrue(torch.equal(original_client[key], changed_client[key]))
        self.assertEqual(result["heldout_sample_weighted"]["n"], 8)
        self.assertEqual(set(predictions["sequence_ids"]), {sid for s in test for sid in s.sequence_ids})
        self.assertNotIn("test_acc", result["history_training_only"][0])

    def test_partition_covers_each_example_once_with_shared_frozen_proportions(self):
        train_y = np.tile(np.arange(15), 20)
        test_y = np.tile(np.arange(15), 10)
        first, proportions = assign_clients(train_y, test_y, num_clients=5, num_classes=15,
                                              alpha=.5, seed=42)
        second, _ = assign_clients(train_y, test_y, num_clients=5, num_classes=15, alpha=.5, seed=42)
        for labels, assignments, rerun in zip((train_y, test_y), first, second):
            self.assertEqual(len(assignments), len(labels))
            self.assertEqual(set(assignments), set(range(5)))
            np.testing.assert_array_equal(assignments, rerun)
        np.testing.assert_allclose(proportions.sum(1), 1.)

    def test_medical_contracts_do_not_use_first_ten_of_fifteen(self):
        mapping, names = class_contract(10)
        self.assertEqual(list(mapping), [42, 41, 43, 44, 45, 46, 47, 8, 7, 58])
        self.assertEqual(names[2], "touch head")
        self.assertEqual(names[6], "nausea/vomiting")
        self.assertEqual(class_contract(15)[1][2], "nausea/vomiting")

    def test_raw_annotation_preparer_preserves_ids_and_rejects_subject_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mapping, _ = class_contract(15)
            annotations, split_ids = [], {"xsub_train": [], "xsub_val": []}
            for split, subject, repeat in [("xsub_train", 1, 2), ("xsub_val", 2, 1)]:
                for source_label in mapping:
                    for repetition in range(repeat):
                        sid = f"S001C001P{subject:03d}R{repetition + 1:03d}A{source_label + 1:03d}"
                        split_ids[split].append(sid)
                        annotations.append({"frame_dir": sid, "label": source_label,
                                            "keypoint": np.full((1, 4, 17, 2), 100, np.float32),
                                            "keypoint_score": np.ones((1, 4, 17), np.float32),
                                            "img_shape": (1080, 1920)})
            source = root / "annotations.pkl"
            source.write_bytes(pickle.dumps({"split": split_ids, "annotations": annotations}))
            prepare(output_dir=root / "prepared", annotation=source, num_clients=2)
            protocol, train, test = fl.load_protocol(root / "prepared")
            self.assertEqual(protocol["source"]["sha256"], fl.sha256(source))
            self.assertEqual(set(np.concatenate([s.sequence_ids for s in train])), set(split_ids["xsub_train"]))
            self.assertEqual(set(np.concatenate([s.sequence_ids for s in test])), set(split_ids["xsub_val"]))
            self.assertEqual(set(np.concatenate([s.subject_ids for s in train])), {1})
            self.assertEqual(set(np.concatenate([s.subject_ids for s in test])), {2})
            # Different sequence IDs are insufficient if the source performer overlaps.
            old_id = split_ids["xsub_val"][0]
            new_id = old_id.replace("P002", "P001").replace("R001", "R009")
            split_ids["xsub_val"][0] = new_id
            next(a for a in annotations if a["frame_dir"] == old_id)["frame_dir"] = new_id
            source.write_bytes(pickle.dumps({"split": split_ids, "annotations": annotations}))
            with self.assertRaisesRegex(ValueError, "performer identities overlap"):
                prepare(output_dir=root / "invalid", annotation=source, num_clients=2)

    def test_protocol_rejects_overlap_and_legacy_one_channel_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            prepare(output_dir=path, smoke=True, num_clients=2)
            protocol, train, test = fl.load_protocol(path)
            self.assertEqual(protocol["purpose"], "synthetic_smoke")
            self.assertEqual(sum(map(len, train)), 30)
            self.assertEqual(sum(map(len, test)), 15)
            with np.load(path / "test.npz") as data:
                arrays = {key: data[key] for key in data.files}
            arrays["subject_ids"][0] = int(train[0].subject_ids[0])
            np.savez_compressed(path / "test.npz", **arrays)
            protocol["files"]["test"]["sha256"] = fl.sha256(path / "test.npz")
            (path / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                fl.load_protocol(path)
            protocol["sample_shape"][-1] = 1
            (path / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "C=3"):
                fl.load_protocol(path)


if __name__ == "__main__":
    unittest.main()
