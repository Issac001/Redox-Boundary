"""Pooled-fit mathematics and portable figure output contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import plot_manuscript_figures as plots
from shared.manuscript_figures_data import crossings, prepare_figure_data


def synthetic_observations():
    depths = np.array([10., 20., 30., 40., 50., 60., 70., 80., 90., 95.])
    truth = -300. + 700. * expit((59. - depths) / 3.)
    return pd.concat([
        pd.DataFrame({"Case": case, "Time": 1., "profile_id": f"{case}_1",
                      "Depth": depths, "Eh": truth + offset})
        for case, offset in zip(["3.1", "3.2", "3.3"], [-5., 0., 5.])
    ], ignore_index=True)


class FigureDataTests(unittest.TestCase):
    def test_pooled_fit_uses_equal_observation_weights_and_full_residuals(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_figure_data(synthetic_observations(), directory)
            np.testing.assert_allclose(list(result["parameters"].values()),
                                       [-300., 700., 59., 3.], rtol=0, atol=1e-9)
            self.assertAlmostEqual(result["rmse"], np.sqrt(50. / 3.), places=9)
            self.assertEqual(result["n_observations"], 30)
            curve = pd.read_csv(Path(directory) / "pooled_sigmoid_predictions.csv")
            self.assertEqual(len(curve), 1001)
            np.testing.assert_allclose(curve.iloc[[0, -1]].depth_cm, [10., 95.])
            counts = pd.read_csv(Path(directory) / "threshold_summary.csv")
            self.assertEqual(len(counts), 9)
            np.testing.assert_allclose(counts[["no_crossing_percent", "one_crossing_percent",
                                               "multiple_crossing_percent"]].sum(axis=1), 100.)

    def test_unequal_depth_coverage_is_not_silently_reweighted(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "identical depths"):
                prepare_figure_data(synthetic_observations().iloc[1:], directory)

    def test_crossings_retain_current_strict_interpolation_convention(self):
        np.testing.assert_allclose(crossings([400., -200., 100.], [10., 20., 40.], 0.),
                                   [10. + 20. / 3., 20. + 40. / 3.])
        self.assertEqual(len(crossings([10., 0., -10.], [10., 20., 30.], 0.)), 0)

    def test_pooled_manuscript_regression(self):
        path = ROOT / "data/manuscript/eh_observations.csv"
        if not path.exists():
            self.skipTest("Manuscript input CSV is not installed")
        with tempfile.TemporaryDirectory() as directory:
            result = prepare_figure_data(pd.read_csv(path, dtype={"Case": str}), directory)
            np.testing.assert_allclose(list(result["parameters"].values()),
                                       [-301.5710249792289, 702.3320865153725, 59., 1.],
                                       rtol=0, atol=1e-8)
            self.assertAlmostEqual(result["rmse"], 159.5060660790135, places=9)
            self.assertAlmostEqual(result["r2"], 0.8165642642829194, places=12)


class PortableFigureTests(unittest.TestCase):
    def test_external_input_and_output_paths_render_with_explicit_font(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir, image_dir = root / "results", root / "independent_figures"
            observations = synthetic_observations()
            observation_path = root / "observations.csv"
            observations.to_csv(observation_path, index=False)
            prepare_figure_data(observations, result_dir / "numeric")
            audit = plots._render(result_dir, observation_path, image_dir,
                                  "DejaVu Serif", ["figure2_1_pooled"])
            for extension in ("png", "pdf", "svg"):
                self.assertTrue((image_dir / f"figure2_1_pooled.{extension}").is_file())
            self.assertEqual(audit["figures"]["figure2_1_pooled"]["font"], "DejaVu Serif")
            self.assertEqual(audit["figures"]["figure2_1_pooled"]["text_color"], "black")
            saved = json.loads((image_dir / "figure_style_audit.json").read_text())
            self.assertEqual(saved["inputs_sha256"], audit["inputs_sha256"])
            self.assertIn("../observations.csv", saved["inputs_sha256"])

    def test_missing_font_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "--font-family"):
            plots.style("Unavailable_Manuscript_Font_83947")


if __name__ == "__main__":
    unittest.main()
