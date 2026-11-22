import contextlib
import csv
import io
import json
import os
import random
import tempfile
import unittest

from .. import generate_dataset, profile_dataset, validate_dataset


class TestLeakCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        rng = random.Random(11)
        n = 3000
        ids = rng.sample(range(10_000_000), n)
        self.rows = [
            [f"{ids[i]:08d}", f"{rng.randrange(1000, 4500):05d}", rng.randint(18, 90),
             f"{rng.uniform(0, 1e8):.2f}"]
            for i in range(n)
        ]
        with open(self.p("real.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["member_id", "zip", "age", "amount"])
            w.writerows(self.rows)
        self.assertEqual(profile_dataset.main(
            ["--input", self.p("real.csv"), "--output", self.p("profile.json")]), 0)
        self.assertEqual(generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "3000", "--seed", "4",
            "--real-csv", self.p("real.csv"), "--output", self.p("sim.csv"),
        ]), 0)

    def p(self, name):
        return os.path.join(self.tmp.name, name)

    def _plant(self, column, value):
        with open(self.p("sim.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rows[1][rows[0].index(column)] = value
        with open(self.p("sim.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    def _validate(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = validate_dataset.main([
                "--profile", self.p("profile.json"), "--simulated", self.p("sim.csv"),
                "--report", self.p("report.json"), "--log-file", self.p("validate.log"),
                *extra,
            ])
        return rc, out.getvalue()

    def _leak_args(self, *extra):
        return ["--real-csv", self.p("real.csv"), *extra]

    def test_clean_file_passes(self):
        rc, out = self._validate(*self._leak_args())
        self.assertEqual(rc, 0)
        self.assertTrue(out.rstrip().endswith("LEAK CHECK: PASS"))
        with open(self.p("report.json"), encoding="utf-8") as f:
            leak = json.load(f)["leak_check"]
        self.assertTrue(leak["passed"])
        self.assertEqual(set(leak["columns"]), {"member_id", "zip"})
        self.assertEqual(leak["columns"]["member_id"]["checked"], 3000)
        self.assertEqual(leak["columns"]["member_id"]["matching"], 0)

    def test_planted_real_id_fails(self):
        self._plant("member_id", f" {self.rows[0][0]} ")  # also checks trimming
        rc, out = self._validate(*self._leak_args())
        self.assertEqual(rc, 1)
        self.assertTrue(out.rstrip().endswith("LEAK CHECK: FAIL (1 columns)"))
        with open(self.p("report.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["leak_check"]["columns"]["member_id"]["matching"], 1)

    def test_allow_real_values_passes(self):
        self._plant("member_id", self.rows[0][0])
        rc, out = self._validate(*self._leak_args("--allow-real-values", "member_id"))
        self.assertEqual(rc, 0)
        self.assertTrue(out.rstrip().endswith("LEAK CHECK: PASS"))

    def test_leak_check_columns_catches_numeric(self):
        rc, _ = self._validate(*self._leak_args("--leak-check-columns", "amount"))
        self.assertEqual(rc, 0)
        self._plant("amount", self.rows[0][3])
        rc, out = self._validate(*self._leak_args("--leak-check-columns", "amount"))
        self.assertEqual(rc, 1)
        # without naming the column, a numeric column is not checked
        rc, _ = self._validate(*self._leak_args())
        self.assertEqual(rc, 0)

    def test_no_real_value_in_stdout_log_or_report(self):
        planted_id, planted_amount = self.rows[0][0], self.rows[0][3]
        self._plant("member_id", planted_id)
        self._plant("amount", planted_amount)
        rc, out = self._validate(*self._leak_args("--leak-check-columns", "amount"))
        self.assertEqual(rc, 1)
        with open(self.p("validate.log"), encoding="utf-8") as f:
            log_text = f.read()
        with open(self.p("report.json"), encoding="utf-8") as f:
            report_text = f.read()
        for name, text in (("stdout", out), ("log", log_text), ("report", report_text)):
            self.assertNotIn(planted_id, text, name)
            self.assertNotIn(planted_amount, text, name)
            for r in self.rows[:300]:
                self.assertNotIn(r[0], text, name)
                self.assertNotIn(r[3], text, name)

    def test_without_real_csv_unchanged(self):
        rc, out = self._validate()
        self.assertEqual(rc, 0)
        self.assertNotIn("LEAK CHECK", out)
        self.assertNotIn("Leak check", out)
        with open(self.p("report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertNotIn("leak_check", report)
        self.assertIn("min", report["columns"]["amount"])
        self.assertIn("max", report["columns"]["amount"])

    def test_unknown_column_gives_clear_error(self):
        rc, out = self._validate(*self._leak_args("--leak-check-columns", "nope"))
        self.assertEqual(rc, 1)
        with open(self.p("validate.log"), encoding="utf-8") as f:
            self.assertIn("nope", f.read())

    def test_column_missing_from_real_csv_gives_clear_error(self):
        with open(self.p("short.csv"), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["member_id"], ["00000001"]])
        rc, _ = self._validate("--real-csv", self.p("short.csv"))
        self.assertEqual(rc, 1)
        with open(self.p("validate.log"), encoding="utf-8") as f:
            self.assertIn("zip", f.read())

    def test_allow_without_real_csv_is_an_error(self):
        rc, _ = self._validate("--allow-real-values", "member_id")
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
