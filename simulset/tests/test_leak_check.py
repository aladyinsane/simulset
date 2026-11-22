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

    def _plant(self, column, value, row=1):
        with open(self.p("sim.csv"), newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rows[row][rows[0].index(column)] = value
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

    def _add_column(self, filename, name, values):
        path = self.p(filename)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rows[0].append(name)
        for r, v in zip(rows[1:], values):
            r.append(v)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

    def _leak_json(self):
        with open(self.p("report.json"), encoding="utf-8") as f:
            return json.load(f)["leak_check"]

    def test_unprofiled_column_with_real_values_fails(self):
        self._add_column("real.csv", "ssn", [f"{900000000 + i}" for i in range(3000)])
        self._add_column("sim.csv", "ssn", [f"{900000000 + i}" for i in range(2000)] + [""] * 1000)
        rc, out = self._validate(*self._leak_args())
        self.assertEqual(rc, 1)
        self.assertTrue(out.rstrip().endswith("LEAK CHECK: FAIL (1 columns)"))
        entry = self._leak_json()["columns"]["ssn"]
        self.assertEqual((entry["dtype"], entry["matching"]), ("unprofiled", 2000))

    def test_leak_check_columns_accepts_unprofiled_column_in_both_csvs(self):
        self._add_column("real.csv", "ssn", ["111"] * 3000)
        self._add_column("sim.csv", "ssn", ["111"] * 3000)
        rc, _ = self._validate(*self._leak_args("--leak-check-columns", "ssn"))
        self.assertEqual(rc, 1)

    def test_simulated_column_not_in_real_csv_fails_unless_allowed(self):
        self._add_column("sim.csv", "extra", ["x"] * 3000)
        rc, out = self._validate(*self._leak_args())
        self.assertEqual(rc, 1)
        self.assertIn("not in real.csv, not checked", out)
        rc, out = self._validate(*self._leak_args("--allow-real-values", "extra"))
        self.assertEqual(rc, 0)

    def test_dropped_leading_zero_matches(self):
        real_id = next(r[0] for r in self.rows if r[0].startswith("0"))
        self._plant("member_id", real_id.lstrip("0"))
        rc, _ = self._validate(*self._leak_args())
        self.assertEqual(rc, 1)
        self.assertEqual(self._leak_json()["columns"]["member_id"]["matching"], 1)

    def test_case_only_difference_matches(self):
        self._add_column("real.csv", "code", ["AbC123"] * 3000)
        self._add_column("sim.csv", "code", ["abc123"] * 3000)
        rc, _ = self._validate(*self._leak_args("--leak-check-columns", "code"))
        self.assertEqual(rc, 1)

    def test_match_keys(self):
        self.assertTrue(set(validate_dataset.match_keys("12345678")) & set(validate_dataset.match_keys("012345678")))
        self.assertTrue(set(validate_dataset.match_keys(" Ab ")) & set(validate_dataset.match_keys("aB")))
        self.assertFalse(set(validate_dataset.match_keys("12345678")) & set(validate_dataset.match_keys("12345679")))
        self.assertTrue(set(validate_dataset.match_keys("0")) & set(validate_dataset.match_keys("000")))

    def test_redaction_keeps_real_label_and_min_max_out_of_output(self):
        amounts = sorted(float(r[3]) for r in self.rows)
        real_max = next(r[3] for r in self.rows if float(r[3]) == amounts[-1])
        real_min = next(r[3] for r in self.rows if float(r[3]) == amounts[0])
        real_zip = self.rows[0][1]
        self._plant("amount", real_max, row=1)
        self._plant("amount", real_min, row=2)
        self._plant("zip", real_zip, row=1)
        rc, out = self._validate(*self._leak_args("--allow-real-values", "zip"))
        with open(self.p("validate.log"), encoding="utf-8") as f:
            log_text = f.read()
        with open(self.p("report.json"), encoding="utf-8") as f:
            report_text = f.read()
        for name, text in (("stdout", out), ("log", log_text), ("report", report_text)):
            for v in (real_max, real_min, str(float(real_max)), str(float(real_min))):
                self.assertNotIn(v, text, name)
            self.assertNotIn(f"'{real_zip}'", text, name)
            self.assertNotIn(f'"{real_zip}"', text, name)


if __name__ == "__main__":
    unittest.main()
