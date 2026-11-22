import csv
import json
import os
import random
import tempfile
import unittest

from .. import generate_dataset, profile_dataset
from ..profiling import classify


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _read_col(path, col):
    with open(path, newline="", encoding="utf-8") as f:
        return [r[col] for r in csv.DictReader(f)]


class TestIntegerCodeClassification(unittest.TestCase):
    def test_unique_digit_ids_with_leading_zeros_are_identifier(self):
        ids = [f"{i:08d}" for i in range(0, 5000, 7)]
        cls = classify.classify_column("member_id", ids)
        self.assertEqual(cls.dtype, classify.IDENTIFIER)
        self.assertTrue(all(len(v) == 8 for v in cls.non_null_values))

    def test_near_unique_short_ids_are_identifier(self):
        ids = [str(i) for i in range(1, 200)] + ["5"]
        self.assertEqual(classify.classify_column("id", ids).dtype, classify.IDENTIFIER)

    def test_repeating_five_digit_codes_are_categorical(self):
        rng = random.Random(0)
        zips = [f"{rng.choice(range(10000, 10300)):05d}" for _ in range(3000)]
        cls = classify.classify_column("zip", zips)
        self.assertEqual(cls.dtype, classify.CATEGORICAL)

    def test_repeating_codes_with_leading_zero_are_categorical(self):
        cls = classify.classify_column("c", ["01", "02", "01", "02", "03", "01"] * 10)
        self.assertEqual(cls.dtype, classify.CATEGORICAL)

    def test_whitespace_is_trimmed(self):
        cls = classify.classify_column("zip", [" 02134", "02134 ", "10001"] * 5)
        self.assertEqual(cls.dtype, classify.CATEGORICAL)
        self.assertIn("02134", cls.non_null_values)

    def test_counts_ages_and_amounts_stay_numeric(self):
        rng = random.Random(1)
        counts = [str(rng.randint(0, 9)) for _ in range(500)]
        ages = [str(rng.randint(18, 90)) for _ in range(500)]
        amounts = [str(rng.randint(1, 2000)) for _ in range(2000)]  # about 63% distinct
        for name, vals in (("count", counts), ("age", ages), ("amount", amounts)):
            self.assertEqual(classify.classify_column(name, vals).dtype, classify.NUMERIC, name)

    def test_tiny_tables_stay_numeric(self):
        ages = [str(a) for a in (31, 45, 27, 52, 38, 64, 29, 41, 57, 33, 48, 36, 25, 61, 44, 39, 50, 28, 35, 42)]
        self.assertEqual(len(set(ages)), 20)
        self.assertEqual(classify.classify_column("age", ages).dtype, classify.NUMERIC)

    def test_near_unique_amounts_are_identifier_unless_treated_as_number(self):
        amounts = [str(a) for a in random.Random(5).sample(range(1, 100000), 1000)]
        self.assertEqual(classify.classify_column("amount", amounts).dtype, classify.IDENTIFIER)
        self.assertEqual(classify.classify_column("amount", amounts, force_numeric=True).dtype,
                         classify.NUMERIC)

    def test_dirty_value_in_id_column_keeps_identifier(self):
        ids = [f"{i:09d}" for i in range(1, 400, 2)] + ["UNKNOWN"]
        cls = classify.classify_column("id", ids)
        self.assertEqual(cls.dtype, classify.IDENTIFIER)
        self.assertNotIn("UNKNOWN", cls.non_null_values)
        self.assertTrue(all(len(v) == 9 for v in cls.non_null_values))

    def test_zip_plus_four_in_zip_column_keeps_categorical(self):
        zips = ["02134", "10001", "60601"] * 100 + ["10001-1234"]
        cls = classify.classify_column("zip", zips)
        self.assertEqual(cls.dtype, classify.CATEGORICAL)
        self.assertIn("02134", cls.non_null_values)

    def test_decimals_and_negatives_stay_numeric(self):
        self.assertEqual(classify.classify_column("x", ["12345.5", "23456.5", "12345.5"]).dtype,
                         classify.NUMERIC)
        self.assertEqual(classify.classify_column("x", ["-12345", "-23456", "-12345"]).dtype,
                         classify.NUMERIC)

    def test_force_numeric(self):
        ids = [f"{i:08d}" for i in range(100)]
        cls = classify.classify_column("n", ids, force_numeric=True)
        self.assertEqual(cls.dtype, classify.NUMERIC)

    def test_warning_names_column_type_and_option(self):
        with self.assertLogs("simulset.profiling.classify", level="WARNING") as cm:
            classify.classify_column("member_id", [f"{i:08d}" for i in range(100)])
        msg = cm.output[0]
        self.assertIn("member_id", msg)
        self.assertIn("identifier", msg)
        self.assertIn("--treat-as-number", msg)


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def p(self, name):
        return os.path.join(self.dir, name)

    def _profile(self, extra=()):
        rc = profile_dataset.main(
            ["--input", self.p("real.csv"), "--output", self.p("profile.json"), *extra]
        )
        self.assertEqual(rc, 0)
        with open(self.p("profile.json"), encoding="utf-8") as f:
            return json.load(f)["columns"]

    def _real_file(self):
        rng = random.Random(3)
        n = 3000
        member_ids = rng.sample(range(10_000_000), n)
        rows = [
            [f"{member_ids[i]:08d}", f"{rng.randrange(1000, 4500):05d}",
             rng.randint(18, 90), rng.randint(1, 20000)]
            for i in range(n)
        ]
        _write_csv(self.p("real.csv"), ["member_id", "zip", "age", "amount"], rows)
        return rows

    def _small_space_file(self):
        # 90 of the 100 two-digit codes are real, so unchecked generation would
        # hit a real value about 90% of the time.
        real = [f"{i:02d}" for i in range(90)]
        _write_csv(self.p("real.csv"), ["code"], [[v] for v in real])
        return real

    def test_profile_types(self):
        self._real_file()
        cols = self._profile()
        self.assertEqual(cols["member_id"]["dtype"], "identifier")
        self.assertEqual(cols["zip"]["dtype"], "categorical")
        self.assertEqual(cols["age"]["dtype"], "numeric")
        self.assertEqual(cols["amount"]["dtype"], "numeric")

    def test_leading_zeros_survive_in_generated_output(self):
        rows = self._real_file()
        self.assertTrue(any(r[0].startswith("0") for r in rows))
        self._profile()
        generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "2000", "--seed", "1",
            "--real-csv", self.p("real.csv"), "--output", self.p("sim.csv"),
        ])
        out = _read_col(self.p("sim.csv"), "member_id")
        self.assertTrue(all(len(v) == 8 and v.isdigit() for v in out))
        self.assertTrue(any(v.startswith("0") for v in out))

    def test_treat_as_number_cli(self):
        self._real_file()
        cols = self._profile(["--treat-as-number", "member_id,zip"])
        self.assertEqual(cols["member_id"]["dtype"], "numeric")
        self.assertEqual(cols["zip"]["dtype"], "numeric")

    def test_zip_output_is_masked_labels_with_real_proportions(self):
        rows = self._real_file()
        self._profile()
        rc = generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "5000", "--seed", "1",
            "--real-csv", self.p("real.csv"), "--output", self.p("sim.csv"),
        ])
        self.assertEqual(rc, 0)
        out = _read_col(self.p("sim.csv"), "zip")
        self.assertTrue(all(v.startswith("zip_val") for v in out))
        real_zips = [r[1] for r in rows]
        self.assertFalse(set(real_zips) & set(out))
        top_share = max(real_zips.count(z) for z in set(real_zips)) / len(real_zips)
        self.assertAlmostEqual(out.count("zip_val1") / len(out), top_share, delta=0.01)

    def test_no_synthetic_id_equals_real_id_in_small_pattern_space(self):
        real = self._small_space_file()
        cols = self._profile()
        self.assertEqual(cols["code"]["dtype"], "identifier")
        rc = generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "500", "--seed", "2",
            "--real-csv", self.p("real.csv"), "--output", self.p("sim.csv"),
        ])
        self.assertEqual(rc, 0)
        out = _read_col(self.p("sim.csv"), "code")
        self.assertEqual(len(out), 500)
        self.assertFalse(set(out) & set(real))

    def test_without_rejection_small_space_would_collide(self):
        # Guards the test above: proves the setup really would collide.
        from ..generation import identifiers
        real = set(self._small_space_file())
        prof = {"pattern_template": [("class", "digit", 2)], "length_stats": {"min": 2, "max": 2}}
        out = identifiers.generate("code", prof, 500, random.Random(2))
        self.assertTrue(set(out) & real)

    def test_debug_log_has_no_identifier_samples(self):
        self._small_space_file()
        self._profile()
        generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "50", "--seed", "2",
            "--output", self.p("sim.csv"),
            "--log-file", self.p("gen.log"), "--log-level", "DEBUG",
        ])
        with open(self.p("gen.log"), encoding="utf-8") as f:
            self.assertNotIn("sample=", f.read())

    def test_real_values_not_in_profile_or_log(self):
        self._small_space_file()
        self._profile()
        generate_dataset.main([
            "--profile", self.p("profile.json"), "--n-records", "50", "--seed", "2",
            "--real-csv", self.p("real.csv"), "--output", self.p("sim.csv"),
            "--log-file", self.p("gen.log"), "--log-level", "INFO",
        ])
        with open(self.p("profile.json"), encoding="utf-8") as f:
            profile_text = f.read()
        self.assertNotIn("real_values", profile_text)
        with open(self.p("gen.log"), encoding="utf-8") as f:
            log_text = f.read()
        # no real value ('00'..'89') quoted as a value in the log
        for i in range(90):
            self.assertNotIn(f"'{i:02d}'", log_text)

    def test_warning_without_real_csv(self):
        self._small_space_file()
        self._profile()
        with self.assertLogs("simulset.generate_dataset", level="WARNING") as cm:
            generate_dataset.main([
                "--profile", self.p("profile.json"), "--n-records", "20", "--seed", "2",
                "--output", self.p("sim.csv"),
            ])
        hits = [m for m in cm.output if "not prevented" in m]
        self.assertEqual(len(hits), 1)
        self.assertIn("code", hits[0])


if __name__ == "__main__":
    unittest.main()
