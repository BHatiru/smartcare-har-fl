"""Interrupted training must preserve the experiment, including personalized BN."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import development_fl as dev
from .test_corrected_fl import toy_model, toy_shard


class Interrupted(Exception):
    pass


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        return ([toy_shard(prefix="train0"), toy_shard(3., prefix="train1")],
                [toy_shard(.5, prefix="dev0", subject_offset=20),
                 toy_shard(3.5, prefix="dev1", subject_offset=20)])

    def run_method(self, method="fedbn", **extra):
        settings = dict(method=method, rounds=3, local_epochs=1, batch_size=2,
                        lr=.01, seed=42, device="cpu", num_classes=2)
        settings.update(extra)
        return dev.run_method(toy_model, *self.inputs(), **settings)

    def test_serialized_round_resume_is_identical_for_all_methods(self):
        for method in dev.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "resume.pt"
                original, expected = self.run_method(method)

                def interrupt(state):
                    if state["current"]["round"] == 1:
                        torch.save(state, path)
                        raise Interrupted()

                with self.assertRaises(Interrupted):
                    self.run_method(method, on_checkpoint=interrupt)
                saved = torch.load(path, map_location="cpu", weights_only=True)
                actual, restored = self.run_method(method, resume_state=saved)
                for key in ("selected_round", "best_development", "final_development",
                            "training_examples_processed", "optimizer_steps", "logical_upload_bytes",
                            "logical_download_bytes", "initial_state_sha256"):
                    self.assertEqual(original[key], actual[key])
                for endpoint in expected:
                    for key in expected[endpoint][0]:
                        np.testing.assert_array_equal(expected[endpoint][0][key], restored[endpoint][0][key])
                    for first, second in zip(expected[endpoint][1]["clients"], restored[endpoint][1]["clients"]):
                        self.assertEqual(dev.state_hash(first), dev.state_hash(second))

    def test_resume_rejects_changed_configuration_and_data(self):
        saved = []
        self.run_method(rounds=1, on_checkpoint=lambda state: saved.append(copy.deepcopy(state)))
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_method(resume_state=saved[-1], batch_size=4)
        train, heldout = self.inputs()
        train[0].x[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "configuration"):
            dev.run_method(toy_model, train, heldout, method="fedbn", rounds=3, local_epochs=1,
                           batch_size=2, lr=.01, seed=42, device="cpu", num_classes=2,
                           resume_state=saved[-1])

    def test_midround_interruption_restarts_from_last_completed_round(self):
        saved = []

        def interrupt(progress):
            if progress["round"] == 2 and progress["client"] == 1:
                raise Interrupted()

        with self.assertRaises(Interrupted):
            self.run_method(on_checkpoint=lambda state: saved.append(copy.deepcopy(state)), on_progress=interrupt)
        self.assertEqual(saved[-1]["current"]["round"], 1)
        expected, endpoints = self.run_method()
        actual, resumed = self.run_method(resume_state=saved[-1])
        self.assertEqual(expected["final_development"], actual["final_development"])
        for a, b in zip(endpoints["final"][1]["clients"], resumed["final"][1]["clients"]):
            self.assertEqual(dev.state_hash(a), dev.state_hash(b))

    def test_completed_resume_does_not_train_again(self):
        saved = []
        expected, _ = self.run_method(rounds=1, on_checkpoint=lambda state: saved.append(copy.deepcopy(state)))
        actual, _ = self.run_method(rounds=1, resume_state=saved[-1],
                                    on_progress=lambda _: self.fail("Completed round trained again"))
        self.assertEqual(expected["final_development"], actual["final_development"])
        self.assertEqual(actual["training_examples_processed"], 8)


if __name__ == "__main__":
    unittest.main()
