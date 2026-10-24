"""Research-critical controls for the development-only runner."""

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
from . import development_fl as dev
from .prepare_corrected_fl import class_contract
from .prepare_development_fl import freeze_split, prepare
from .test_corrected_fl import toy_model, toy_shard


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def shards(self):
        train = [toy_shard(prefix="train0"), toy_shard(3., prefix="train1")]
        heldout = [toy_shard(.5, prefix="dev0", subject_offset=20),
                   toy_shard(3.5, prefix="dev1", subject_offset=20)]
        return train, heldout

    def run_method(self, method, **extra):
        train, heldout = self.shards()
        return dev.run_method(toy_model, train, heldout, method=method, rounds=2, local_epochs=1,
                              batch_size=2, lr=.01, seed=42, device="cpu", num_classes=2, **extra)

    def test_zero_mu_is_exactly_fedavg(self):
        a, artifacts_a = self.run_method("fedavg")
        b, artifacts_b = self.run_method("fedprox", proximal_mu=0.)
        self.assertEqual(a["initial_state_sha256"], b["initial_state_sha256"])
        for endpoint in artifacts_a:
            np.testing.assert_array_equal(artifacts_a[endpoint][0]["logits"], artifacts_b[endpoint][0]["logits"])
            self.assertEqual(dev.state_hash(artifacts_a[endpoint][1]["server"]),
                             dev.state_hash(artifacts_b[endpoint][1]["server"]))

    def test_proximal_gradient_matches_manual_two_step_reference(self):
        torch.manual_seed(2)
        actual = nn.Linear(3, 2, bias=False)
        expected = copy.deepcopy(actual)
        shard = toy_shard()
        reference = fl.state_copy(actual)
        fl.train_client(actual, shard, device="cpu", batch_size=4, epochs=2, lr=.1, seed=3,
                        momentum=0., weight_decay=0., proximal_mu=2., reference_state=reference)
        for _ in range(2):
            loss = nn.functional.cross_entropy(expected(shard.x), shard.y)
            gradient, = torch.autograd.grad(loss, (expected.weight,))
            with torch.no_grad():
                expected.weight -= .1 * (gradient + 2. * (expected.weight - reference["weight"]))
        torch.testing.assert_close(actual.weight, expected.weight)
        self.assertFalse(reference["weight"].requires_grad)

    def test_fedbn_survives_rounds_and_excludes_all_bn_payload(self):
        report, artifacts = self.run_method("fedbn")
        clients = artifacts["final"][1]["clients"]
        for state in clients:
            self.assertEqual(state["0.num_batches_tracked"].item(), 4)
        self.assertFalse(torch.equal(clients[0]["0.running_mean"], clients[1]["0.running_mean"]))
        model = toy_model()
        shared = {k: v for k, v in model.state_dict().items() if k not in fl.batchnorm_keys(model)}
        self.assertEqual(report["logical_upload_bytes"], 2 * 2 * dev.state_bytes(shared))
        self.assertEqual(report["logical_download_bytes"], report["logical_upload_bytes"])

    def test_all_methods_same_initialization_examples_and_heldout_support(self):
        reports = [self.run_method(method)[0] for method in dev.METHODS]
        self.assertEqual(len({r["initial_state_sha256"] for r in reports}), 1)
        for report in reports:
            self.assertEqual(report["training_examples_processed"], 16)
            self.assertEqual(report["final_development"]["pooled"]["n"], 8)
            self.assertEqual(report["optimizer_steps"], 8)
            if report["method"] in {"centralized", "local"}:
                self.assertEqual(report["logical_upload_bytes"], 0)

    def test_development_labels_cannot_change_final_weights(self):
        _, original = self.run_method("fedbn")
        train, heldout = self.shards()
        for shard in heldout:
            shard.y[:] = 0
        _, changed = dev.run_method(toy_model, train, heldout, method="fedbn", rounds=2, local_epochs=1,
                                    batch_size=2, lr=.01, seed=42, device="cpu", num_classes=2)
        for a, b in zip(original["final"][1]["clients"], changed["final"][1]["clients"]):
            self.assertEqual(dev.state_hash(a), dev.state_hash(b))

    def test_selection_uses_development_and_preserves_earlier_state(self):
        evaluate = dev.evaluate
        count = 0

        def forced(*args, **kwargs):
            nonlocal count
            metrics, predictions = evaluate(*args, **kwargs)
            count += 1
            metrics["pooled"]["macro_f1_all_declared_classes"] = 1. / count
            return metrics, predictions

        with patch.object(dev, "evaluate", side_effect=forced):
            result, artifacts = self.run_method("fedbn")
        self.assertEqual(result["selected_round"], 1)
        self.assertEqual(artifacts["best_development"][1]["clients"][0]["0.num_batches_tracked"].item(), 2)
        self.assertEqual(artifacts["final"][1]["clients"][0]["0.num_batches_tracked"].item(), 4)

    def test_split_freezes_whole_performers_deterministically(self):
        records = [{"frame_dir": f"S001C001P{s:03d}R001A{label:03d}"}
                   for s in range(1, 7) for label in range(1, 4)]
        first = freeze_split(records, split_seed=123, development_subjects=2)
        second = freeze_split(list(reversed(records)), split_seed=123, development_subjects=2)
        self.assertEqual(first, second)
        train_subjects = {a["frame_dir"][8:12] for a in first["train"]}
        dev_subjects = {a["frame_dir"][8:12] for a in first["development"]}
        self.assertFalse(train_subjects & dev_subjects)

    def test_overlapping_performers_or_sequences_rejected(self):
        train, heldout = self.shards()
        heldout[0].subject_ids[0] = train[0].subject_ids[0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            dev.validate_shards(train, heldout)
        train, heldout = self.shards()
        train[1].sequence_ids = train[0].sequence_ids.copy()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            dev.validate_shards(train, heldout)

    def test_preparation_loading_and_hash_tamper_control(self):
        annotations, splits = [], {"xsub_train": [], "xsub_val": []}
        for subject in range(1, 6):
            for label in class_contract(15)[0]:
                sid = f"S001C001P{subject:03d}R001A{label + 1:03d}"
                splits["xsub_val" if subject == 5 else "xsub_train"].append(sid)
                annotations.append({"frame_dir": sid, "label": label,
                                    "keypoint": np.ones((1, 4, 17, 2), dtype=np.float32),
                                    "keypoint_score": np.ones((1, 4, 17), dtype=np.float32)})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pkl"
            source.write_bytes(pickle.dumps({"annotations": annotations, "split": splits}))
            output = root / "prepared"
            prepare(annotation=source, output_dir=output, train_per_class=2,
                    development_per_class=1, development_subjects=1, num_clients=2)
            protocol, train, heldout = dev.load_development(output)
            self.assertEqual(sum(map(len, train)), 30)
            self.assertEqual(sum(map(len, heldout)), 15)
            self.assertFalse((output / "test.npz").exists())
            self.assertFalse(protocol["publication_accuracy_claim_allowed"])
            with (output / "development.npz").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                dev.load_development(output)


if __name__ == "__main__":
    unittest.main()
