import copy
import unittest
import lab

class ExperimentTests(unittest.TestCase):
    def test_split_reproducibility_and_leakage(self):
        labels = ["card_arrival", "cash_withdrawal"]
        train = [{"text": f"train {label} {i}", "label": label} for label in range(2) for i in range(8)]
        test = [{"text": f"test {label} {i}", "label": label} for label in range(2) for i in range(3)]
        train += [dict(test[0]), {"text": "  TEST 0 0 ", "label": 0}]
        config = dict(seed=42, train_per_label=3, validation_per_label=2, test_per_label=2)
        splits, removed = lab.make_splits(train, test, labels, config)
        self.assertEqual(removed, 2)
        self.assertEqual(splits, lab.make_splits(train, test, labels, config)[0])
        self.assertEqual({k:len(v) for k,v in splits.items()}, {"train":6, "validation":4, "test":4})
        lab.validate_splits(splits, labels)
        splits["test"][0] = splits["train"][0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            lab.validate_splits(splits, labels)

    def test_negative_results_and_incompatible_runs(self):
        rows = [{"id":"test:1", "text":"synthetic test", "label":"card_arrival", "prediction":"card_arrival"}]
        base = dict(status="evaluated", mode="base", manifest={"data_sha256":"same"},
                    generation={"do_sample":False}, rows=rows)
        tuned = copy.deepcopy(base)
        tuned["mode"] = "adapter"
        tuned["rows"][0]["prediction"] = "An explanation instead of a label"
        before, after, delta = lab.compare_runs(base, tuned)
        self.assertEqual(delta, -1)
        self.assertEqual(after["invalid_output_rate"], 1)
        tuned["manifest"]["data_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Different"):
            lab.compare_runs(base, tuned)

    def test_missing_results_cannot_become_success(self):
        with self.assertRaises(ValueError):
            lab.compare_runs({"status":"not_run"}, {"status":"not_run"})
        with self.assertRaises(ValueError):
            lab.scores([], [])

if __name__ == "__main__":
    unittest.main()
