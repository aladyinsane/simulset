import contextlib
import csv
import io
import json
import os
import random
import tempfile
import unittest

from .. import generate_dataset, profile_dataset, validate_dataset
from ..profiling import classify


class TestEmptyClassification(unittest.TestCase):
    def test_all_missing_markers_are_empty(self):
        for markers in (["", "", ""], ["NA", "n/a", "NULL", "None", ".", " "]):
            cls = classify.classify_column("x", markers)
            self.assertEqual(cls.dtype, classify.EMPTY)
            self.assertEqual(cls.n_non_null, 0)

    def test_warning_names_column(self):
        with self.assertLogs("simulset.profiling.classify", level="WARNING") as cm:
            classify.classify_column("notes", ["", "NA"])
        self.assertIn("notes", cm.output[0])

    def test_one_value_is_not_empty(self):
        cls = classify.classify_column("x", ["", "", "7", ""])
        self.assertEqual(cls.dtype, classify.NUMERIC)
        self.assertEqual(cls.n_non_null, 1)


class TestEmptyEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        rng = random.Random(2)
        self.rows = [
            [f"{i:07d}", rng.randint(18, 90), "", "NA" if i % 2 else "N/A",
             "5" if i == 3 else ""]
            for i in range(300)
        ]
        with open(self.p("real.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["member_id", "age", "blank", "markers", "one_value"])
            w.writerows(self.rows)

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def _read(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_profile_generate_validate_with_leak_check(self):
        self.assertEqual(profile_dataset.main(
            ["--input", self.p("real.csv"), "--output", self.p("profile.json")]), 0)
        with open(self.p("profile.json"), encoding="utf-8") as f:
            cols = json.load(f)["columns"]
        for name in ("blank", "markers"):
            self.assertEqual(cols[name], {"dtype": "empty", "missing_rate": 1.0})
        self.assertEqual(cols["one_value"]["dtype"], "numeric")

        self.assertEqual(generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "100", "--seed", "1",
            "--use-real-values", "blank", "--real-csv", self.p("real.csv"),
            "--output", self.p("sim.csv"),
        ]), 0)
        sim = self._read(self.p("sim.csv"))
        self.assertEqual(len(sim), 100)
        self.assertTrue(all(r["blank"] == "" and r["markers"] == "" for r in sim))

        for extra in ([], ["--real-csv", self.p("real.csv"), "--leak-check-columns", "markers"]):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = validate_dataset.main([
                    "--profile", self.p("profile.json"), "--simulated", self.p("sim.csv"),
                    "--report", self.p("report.json"), *extra,
                ])
            self.assertEqual(rc, 0)
        self.assertTrue(out.getvalue().rstrip().endswith("LEAK CHECK: PASS"))
        with open(self.p("report.json"), encoding="utf-8") as f:
            leak = json.load(f)["leak_check"]["columns"]["markers"]
        self.assertEqual((leak["checked"], leak["matching"]), (0, 0))

    def test_one_value_column_still_generates(self):
        self.assertEqual(profile_dataset.main(
            ["--input", self.p("real.csv"), "--output", self.p("profile.json")]), 0)
        self.assertEqual(generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "50", "--seed", "1",
            "--output", self.p("sim.csv"),
        ]), 0)
        sim = self._read(self.p("sim.csv"))
        self.assertTrue(all(r["one_value"] in ("", "5") for r in sim))


if __name__ == "__main__":
    unittest.main()
