"""Checks of the manuscript's observation, pairing and weighting policies."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from shared.manuscript_analysis import (
    METHODS, diagnose_centers, fit_chemical_centers, match_chemical_centers,
    summarize_lodo, summarize_simulations,
)


class ChemistryTests(unittest.TestCase):
    def rows(self):
        z = np.array([40., 50., 60., 70., 80., 90.])
        q = expit((60. - z) / 8.)
        return pd.DataFrame(dict(Case="3.1", Time=1., Depth=z,
                                 NO3=np.expm1(3. * q), NH4=np.expm1(3. * (1 - q)),
                                 Fe=np.nan, Mn=0.))

    def test_analyte_specific_missingness_zeros_and_outlet_exclusion(self):
        data = self.rows()
        data.loc[5, "NO3"] = 0.
        data.loc[0, "NH4"] = np.nan
        data.loc[:3, "Fe"] = [0., 1., 2., 3.]  # Four depths are insufficient.
        outlet = pd.DataFrame(dict(Case=["3.1"], Time=[1.], Depth=[95.],
                                   NO3=[1e9], NH4=[1e9], Fe=[1e9], Mn=[1e9]))
        result = fit_chemical_centers(pd.concat([data, outlet], ignore_index=True)).set_index("analyte")
        self.assertEqual(set(result.index), {"NO3", "NH4", "Mn"})
        self.assertEqual(result.loc["NO3", "n_depths"], 6)
        self.assertEqual(result.loc["NH4", "n_depths"], 5)
        self.assertTrue(result.observed_depth_max_cm.eq(90).all())
        self.assertFalse(result.loc["Mn", "identifiable"])
        self.assertTrue(np.isnan(result.loc["Mn", "b"]))
        # Opposite concentration directions locate the same synthetic center.
        pure = fit_chemical_centers(self.rows()).set_index("analyte")
        np.testing.assert_allclose(pure.loc[["NO3", "NH4"], "b"], [60., 60.])
        np.testing.assert_allclose(pure.loc[["NO3", "NH4"], "w"], [8., 8.])

    def test_one_sided_finite_center_is_retained_and_flagged(self):
        data = self.rows().iloc[1:].copy()
        data["NO3"] = np.expm1(3. * expit((30. - data.Depth) / 18.))
        row = fit_chemical_centers(data).set_index("analyte").loc["NO3"]
        self.assertAlmostEqual(row.b, 30.)
        self.assertTrue(row.identifiable)
        self.assertTrue(row.extrapolated)
        self.assertFalse(row.center_within_observed_range)
        self.assertEqual(row.n_shallower_observations, 0)
        self.assertEqual(row.n_deeper_observations, 5)

    def test_negative_infinite_and_duplicate_records_raise(self):
        for bad in [-.1, np.inf]:
            data = self.rows()
            data.loc[0, "NO3"] = bad
            with self.assertRaises(ValueError):
                fit_chemical_centers(data)
        data = self.rows()
        with self.assertRaises(ValueError):
            fit_chemical_centers(pd.concat([data, data.iloc[[0]]]))


class PairingAndDiagnosticTests(unittest.TestCase):
    def test_exact_time_tolerance_does_not_become_nearest_time_matching(self):
        chemical = pd.DataFrame(dict(case=["3.1"] * 3, time_day=[1. + 5e-10, 1. + 2e-9, 2.],
                                     analyte=["NO3"] * 3, b=[50., 51., 52.],
                                     n_shallower_observations=[2, 2, 0], n_deeper_observations=[3, 3, 5]))
        eh = pd.DataFrame(dict(Case=["3.1", "3.1"], Time=[1., 2.], center=[60., 62.],
                               profile_id=["a", "b"], diagnostic_pass_current=[True, False]))
        matches, summary = match_chemical_centers(chemical, eh)
        np.testing.assert_array_equal(matches.exact_time_match_count, [1, 0, 1])
        self.assertAlmostEqual(matches.chemical_minus_eh_cm.iloc[0], -10.)
        self.assertTrue(np.isnan(matches.chemical_minus_eh_cm.iloc[1]))
        # A failed Eh diagnostic and a one-sided chemical fit are retained.
        self.assertAlmostEqual(matches.chemical_minus_eh_cm.iloc[2], -10.)
        self.assertEqual(summary.loc[summary.subset.eq("all"), "n_matched"].iloc[0], 2)
        self.assertEqual(summary.loc[summary.subset.eq("two_sided_only"), "n_matched"].iloc[0], 1)
        with self.assertRaises(ValueError):
            match_chemical_centers(chemical, pd.concat([eh, eh.iloc[[0]]], ignore_index=True))

    def test_three_diagnostics_are_joint_and_width_edge_is_separate(self):
        centers = pd.DataFrame(dict(method=["sigmoid"] * 5,
                                    center=[50., 20., 50., 50., np.nan],
                                    amplitude=[100., 100., 99., 100., 100.],
                                    lower=[40., 20., 40., 39.9, 40.],
                                    upper=[60., 40., 60., 60., 60.], width=[1., 8., 8., 8., 8.]))
        result = diagnose_centers(centers)
        np.testing.assert_array_equal(result.diagnostic_pass_current, [True, False, False, False, False])
        self.assertTrue(result.width_edge.iloc[0])
        self.assertEqual(len(result), 5)


class AggregationTests(unittest.TestCase):
    def test_lodo_equal_profiles_not_equal_rows_or_cases(self):
        records = []
        # Unequal counts make a pooled-row implementation distinguishable from
        # equal within-profile depth means, even though the paper has 10 each.
        for method, multiplier in zip(METHODS, [1., 2., 3.]):
            records.append(dict(Case="3.1", profile_id="a", method=method, depth_cm=10.,
                                observed=0., predicted=2. * multiplier, region="endpoints"))
            for depth in [10., 20., 30.]:
                records.append(dict(Case="3.2", profile_id="b", method=method, depth_cm=depth,
                                    observed=0., predicted=6. * multiplier,
                                    region="endpoints" if depth == 10. else "interior"))
        predictions = pd.DataFrame(records)
        regions, depths, winners, wins = summarize_lodo(predictions)
        row = regions.loc[regions.Case.eq("ALL") & regions.method.eq("sigmoid") & regions.region.eq("all")].iloc[0]
        self.assertAlmostEqual(row.rmse_mv, np.sqrt((4. + 36.) / 2))
        self.assertEqual(row.n_profiles, 2)
        self.assertEqual(row.n_predictions, 4)
        self.assertTrue(winners.winner.eq("sigmoid").all())
        # Equal method scores must not silently award the first method a win.
        predictions["predicted"] = 1.
        _, _, winners, wins = summarize_lodo(predictions)
        self.assertTrue(winners.tie_count.eq(3).all())
        self.assertTrue(winners.winner.eq("tie").all())
        self.assertTrue(wins.wins.eq(0).all())

    def test_simulation_coverage_uses_all_replicates_and_includes_endpoints(self):
        scenarios = ["logistic_w8_a800_s25", "step_a800_s25", "erf_w8_a800_s25", "gompertz_w8_a800_s25"]
        rows = []
        for scenario in scenarios:
            for method in METHODS:
                rows.extend([dict(scenario=scenario, method=method, replicate=0, true_center=50., center=52., lower=50., upper=54.),
                             dict(scenario=scenario, method=method, replicate=1, true_center=50., center=48., lower=46., upper=49.)])
        result = summarize_simulations(pd.DataFrame(rows))
        self.assertEqual(len(result), 12)
        np.testing.assert_allclose(result.rmse, 2.)
        np.testing.assert_allclose(result.coverage_percent, 50.)
        np.testing.assert_allclose(result.coverage_mcse_pp, 100 * np.sqrt(.5 * .5 / 2))
        np.testing.assert_allclose(result.median_interval_length, 3.5)
        bad = pd.DataFrame(rows).drop(index=0)
        with self.assertRaises(ValueError):
            summarize_simulations(bad)
        bad = pd.DataFrame(rows)
        bad.loc[0, "center"] = np.nan
        with self.assertRaises(ValueError):
            summarize_simulations(bad)


if __name__ == "__main__":
    unittest.main()
