"""Validate the four truth shapes, reproducibility, and unconditional coverage."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from shared.comparison_simulation import scenario_catalog, generate_scenario, scenario_seed, summarize_fit

Z = np.r_[np.arange(10, 91, 10), 95.]


class SimulationTests(unittest.TestCase):
    def test_only_four_manuscript_scenarios_and_reproducible_streams(self):
        catalog = scenario_catalog()
        self.assertEqual([s['id'] for s in catalog], [
            'logistic_w8_a800_s25', 'step_a800_s25',
            'erf_w8_a800_s25', 'gompertz_w8_a800_s25'])
        for spec in catalog:
            seed = scenario_seed(42, spec['id'])
            a = generate_scenario(spec, 8, seed, Z)
            b = generate_scenario(dict(spec), 8, seed, Z)
            np.testing.assert_array_equal(a['observed'], b['observed'])
            self.assertTrue(np.isfinite(a['observed']).all())
            self.assertTrue(np.isfinite(a['true_center']).all())
            # Independent check of draw order, not the production generator.
            rng = np.random.default_rng(seed)
            np.testing.assert_array_equal(a['true_center'], rng.uniform(30., 80., 8))
            np.testing.assert_allclose(a['observed']-a['mean'],
                                       25.*rng.normal(size=(8, len(Z))), atol=1e-12)

    def test_seed_is_independent_of_scenario_order(self):
        catalog = scenario_catalog()
        forwards = {s['id']: scenario_seed(20261018, s['id']) for s in catalog}
        backwards = {s['id']: scenario_seed(20261018, s['id']) for s in catalog[::-1]}
        self.assertEqual(forwards, backwards)
        self.assertEqual(len(set(forwards.values())), 4)

    def test_smooth_shapes_have_equal_10_90_span(self):
        expected = 2*np.log(9.)*8.
        for spec in scenario_catalog():
            truth = generate_scenario(spec, 1, 11, Z)
            scale = truth['shape_scale']
            if spec['family'] == 'step':
                self.assertEqual(truth['transition_span_10_90'], 0.)
                continue
            if spec['family'] == 'logistic':
                span = 2*np.log(9.)*scale
            elif spec['family'] == 'erf':
                span = (ndtri(.9)-ndtri(.1))*scale
            else:
                span = (np.log(-np.log(.1))-np.log(-np.log(.9)))*scale
            self.assertAlmostEqual(span, expected)
            self.assertAlmostEqual(truth['transition_span_10_90'], expected)

    def test_gompertz_target_is_gradient_maximum_not_half_height(self):
        spec = next(s for s in scenario_catalog() if s['family'] == 'gompertz')
        truth = generate_scenario(spec, 1, 1729, Z)
        c, s = truth['true_center'][0], truth['shape_scale']
        grid = np.linspace(c-3*s, c+3*s, 6001)
        q = np.exp(-np.exp((grid-c)/s))
        self.assertLess(abs(grid[np.argmax(-np.gradient(q, grid))]-c), .002*s)
        self.assertAlmostEqual(np.exp(-np.exp(0)), 1/np.e)
        self.assertNotAlmostEqual(np.exp(-np.exp(0)), .5)

    def test_missing_estimates_are_not_removed_from_coverage(self):
        spec = scenario_catalog()[0]
        truth = generate_scenario(spec, 4, 3, Z)
        true = truth['true_center']
        center = true.copy()
        center[3] = np.nan
        lo, hi = true-1, true+1
        lo[1], hi[1] = true[1]+2, true[1]+3
        lo[3] = hi[3] = np.nan
        # Diagnostic flags must not select which replicates enter evaluation.
        fit = dict(center=center, lower=lo, upper=hi, prediction=truth['mean'],
                   diagnostic_pass=np.zeros(4, dtype=bool))
        result = summarize_fit(spec, 'test', fit, truth, Z)
        self.assertEqual(result['n'], 4)
        self.assertEqual(result['coverage_all'], .5)
        self.assertEqual(result['coverage_percent'], 50.)
        self.assertAlmostEqual(result['coverage_mcse_pp'], 25.)
        self.assertEqual(result['estimate_availability'], .75)
        self.assertEqual(result['n_valid_estimate'], 3)
        self.assertEqual(result['n_valid_interval'], 3)
        self.assertEqual(result['rmse'], 0.)
        self.assertEqual(result['median_interval_length'], 2.)

    def test_interval_endpoints_count_as_covered(self):
        spec = scenario_catalog()[0]
        truth = generate_scenario(spec, 3, 3, Z)
        true = truth['true_center']
        fit = dict(center=true, lower=true, upper=true)
        summary = summarize_fit(spec, 'test', fit, truth, Z)
        self.assertEqual(summary['coverage_all'], 1.)
        self.assertEqual(summary['coverage_all_mcse'], 0.)
        self.assertEqual(summary['median_interval_length'], 0.)


if __name__ == '__main__':
    unittest.main()
