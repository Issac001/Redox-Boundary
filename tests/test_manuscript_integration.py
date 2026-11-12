"""Portable input validation, clean-run protection, and manuscript verification."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_manuscript as run
from shared.comparison_io import prepare_output_directory


class ManuscriptIntegrationTests(unittest.TestCase):
    def test_config_cannot_silently_change_scientific_definitions_or_labels(self):
        cfg = json.loads(run.DEFAULT_CONFIG.read_text())
        cfg["seed"] = 13
        cfg["n_replicates"] = 10
        run.validate_config(cfg)
        cfg["model"]["center_bounds"] = [25, 85]
        with self.assertRaisesRegex(ValueError, "fixed model"):
            run.validate_config(cfg)

    def test_default_inputs_have_exact_manuscript_profiles(self):
        cfg = json.loads(run.DEFAULT_CONFIG.read_text())
        data, meta, y, z = run.load_observations(run.DEFAULT_DATA / "eh_observations.csv", cfg)
        self.assertEqual(y.shape, (223, 10))
        self.assertEqual(len(data), 2230)
        self.assertEqual(meta.groupby("Case").size().to_dict(), {"3.1": 77, "3.2": 78, "3.3": 68})
        self.assertEqual(z[-1], 95)
        self.assertEqual(cfg["n_replicates"], 500)

    def test_inputs_and_configuration_match_recorded_provenance(self):
        provenance = json.loads((run.DEFAULT_DATA / "provenance.json").read_text())
        for name, record in provenance["input_files"].items():
            self.assertEqual(run.sha256(run.DEFAULT_DATA / name), record["sha256"])
        self.assertEqual(run.sha256(run.DEFAULT_CONFIG), provenance["configuration_sha256"])
        self.assertEqual(run.sha256(run.DEFAULT_DATA / "expected_results.json"), provenance["expected_results"]["sha256"])

    def test_incomplete_or_duplicate_profiles_are_rejected(self):
        cfg = json.loads(run.DEFAULT_CONFIG.read_text())
        data = pd.read_csv(run.DEFAULT_DATA / "eh_observations.csv")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            for invalid in (data.iloc[1:], pd.concat([data, data.iloc[:1]])):
                invalid.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "Incomplete|Duplicate"):
                    run.load_observations(path, cfg)

    def test_existing_results_and_symlink_targets_remain_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            output.mkdir()
            sentinel = output / "saved.csv"
            sentinel.write_text("unchanged")
            link = root / "alias"
            link.symlink_to(output, target_is_directory=True)
            for path in (output, link, sentinel):
                with self.assertRaises(FileExistsError):
                    prepare_output_directory(path)
            with patch.object(run, "load_observations") as load:
                with self.assertRaises(FileExistsError):
                    run.main(["--output", str(output), "--no-figures"])
                load.assert_not_called()
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_reference_checks_detect_changed_numerical_result(self):
        expected = json.loads((run.DEFAULT_DATA / "expected_results.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name, records in expected.items():
                pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
            self.assertEqual(len(run.verify_reference(output, run.DEFAULT_DATA / "expected_results.json")), 5)
            path = output / "simulation_four_scenarios.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "rmse"] += 1
            frame.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                run.verify_reference(output, run.DEFAULT_DATA / "expected_results.json")


if __name__ == "__main__":
    unittest.main()
