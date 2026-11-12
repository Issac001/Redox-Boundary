"""Numerical and design checks independent of the production closed forms.

Run from the project root: python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import dblquad
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from shared.comparison_models import (admissible_gaps, fit_models, _ordered_segment_posterior,
                    _mixture_quantile)


class OrderedGaussianEvidenceTests(unittest.TestCase):
    def test_evidence_and_posterior_against_direct_integration(self):
        # Integrate the original likelihood times the normalized ordered prior.
        # This does not reuse marginal-evidence or truncated-moment formulas.
        y = np.array([2.0, 1.5, -0.5, -1.0])
        sigma, prior_sd = 1.0, 2.0
        normalizer = (2 * np.pi * sigma ** 2) ** (-len(y) / 2)
        def density(lower, upper):
            residual = np.sum((y[:2] - upper) ** 2) + np.sum((y[2:] - lower) ** 2)
            return (2 * norm.pdf(upper, scale=prior_sd) * norm.pdf(lower, scale=prior_sd)
                    * normalizer * np.exp(-0.5 * residual / sigma ** 2))
        integrate = lambda f: dblquad(f, -12, 12, lambda u: -12, lambda u: u,
                                     epsabs=2e-10, epsrel=2e-8)[0]
        reference = integrate(density)
        expected_u = integrate(lambda l, u: u * density(l, u)) / reference
        expected_l = integrate(lambda l, u: l * density(l, u)) / reference
        log_evidence, upper, lower = _ordered_segment_posterior(y[None, :], [2], sigma, 0.0, prior_sd)
        np.testing.assert_allclose(np.exp(log_evidence[0, 0]), reference, rtol=2e-7)
        np.testing.assert_allclose([upper[0, 0], lower[0, 0]], [expected_u, expected_l], rtol=2e-7)

    def test_negative_contrast_is_numerically_stable(self):
        y = np.array([[-10000.0, -10000.0, 10000.0, 10000.0]])
        log_evidence, upper, lower = _ordered_segment_posterior(y, [2], 1.0, 0.0, 500.0)
        self.assertTrue(np.isfinite(log_evidence).all())
        self.assertTrue(np.isfinite(upper).all() and np.isfinite(lower).all())
        self.assertGreater(float(upper[0, 0] - lower[0, 0]), 0.0)


class CommonModelTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 100.])
        self.cfg = {"center_bounds": [15., 90.], "grid_step": .2,
                    "width_grid": [1., 2., 3., 5., 8., 12., 18.], "sigma": 25.}

    def test_exact_sigmoid_reconstruction_and_current_method_set(self):
        z = self.z
        center = 44.0
        y = -150. + 500. * expit((center-z) / 5.)
        result = fit_models(y, z, self.cfg)
        sigmoid = result["sigmoid"]
        self.assertAlmostEqual(sigmoid["center"][0], center, places=7)
        np.testing.assert_allclose(sigmoid["prediction"][0], y, atol=1e-8)
        self.assertTrue(sigmoid["diagnostic_pass"][0])
        self.assertEqual(set(result), {"sigmoid", "step", "bayes_step"})
        self.assertNotIn("diagnostic_pass", result["step"])
        self.assertNotIn("diagnostic_pass", result["bayes_step"])

    def test_null_and_ascending_profiles_do_not_get_frequentist_centers(self):
        y = np.vstack([np.full(len(self.z), 200.), 2. * self.z - 100.])
        result = fit_models(y, self.z, self.cfg)
        for name in ["sigmoid", "step"]:
            self.assertTrue(np.isnan(result[name]["center"]).all())
        self.assertFalse(result["sigmoid"]["diagnostic_pass"].any())
        bayes = result["bayes_step"]
        self.assertTrue(np.isfinite(bayes["center"]).all())

    def test_single_gap_retains_gap_uncertainty_and_integrates_predictions(self):
        z = np.array([0., 10., 30., 50.])
        y = np.array([300., 310., -100., -110.])
        cfg = {"center_bounds": [0., 50.], "sigma": 5., "grid_step": .5,
               "width_grid": [1., 3., 8.]}
        query = np.array([10., 15., 20., 25., 30.])
        out = fit_models(y, z, cfg, query_depth=query)["bayes_step"]
        np.testing.assert_allclose(out["gap_probability"], [[1.]])
        # Uniform location on the complete 20-unit unobserved gap.
        np.testing.assert_allclose([out["center"][0], out["lower"][0], out["upper"][0]], [20., 10.5, 29.5])
        p = out["prediction"][0]
        np.testing.assert_allclose(p, np.linspace(p[0], p[-1], len(query)), atol=1e-10)
        self.assertGreater(p[0], p[-1])
        step = fit_models(y, z, cfg)["step"]
        np.testing.assert_allclose([step["lower"][0], step["upper"][0]], [10., 30.])

    def test_gap_prior_is_uniform_in_physical_depth_when_likelihood_flat(self):
        # Very weak likelihood makes the posterior approach the depth-uniform
        # prior despite unequal gap lengths and unequal counts in each segment.
        z = np.array([0., 1., 2., 8., 9., 10.])
        cfg = {"center_bounds": [0., 10.], "sigma": 1e8,
               "prior_sd": 1., "grid_step": .5, "width_grid": [1., 3.]}
        out = fit_models(np.zeros(len(z)), z, cfg)["bayes_step"]
        np.testing.assert_allclose(out["gap_probability"][0], [1/8, 6/8, 1/8], atol=1e-12)
        np.testing.assert_allclose([out["center"][0], out["lower"][0], out["upper"][0]], [5., 1.2, 8.8], atol=1e-10)

    def test_equal_tail_quantiles_have_expected_probability(self):
        probability = np.array([[.1, .6, .3], [.9, .05, .05]])
        lower, upper = np.array([1., 3., 10.]), np.array([3., 10., 20.])
        for q in [.025, .5, .975]:
            quantile = _mixture_quantile(probability, lower, upper, q)
            cdf = np.sum(probability * np.clip((quantile[:, None] - lower) / (upper-lower), 0, 1), axis=1)
            np.testing.assert_allclose(cdf, q, atol=1e-13)

    def test_uneven_depth_and_leave_one_depth_share_candidate_domain(self):
        z = np.array([0., 5., 17., 30., 41., 56., 70., 89., 100., 120.])
        y = 500. * expit((49.-z)/5.)
        cfg = {**self.cfg, "center_bounds": [12., 95.]}
        for holdout in range(len(z)):
            keep = np.arange(len(z)) != holdout
            _, lo, hi = admissible_gaps(z[keep], cfg)
            out = fit_models(y[keep], z[keep], cfg, query_depth=z[[holdout]])
            for name in ["sigmoid", "step", "bayes_step"]:
                model = out[name]
                self.assertEqual(model["prediction"].shape, (1, 1))
                self.assertTrue(np.isfinite(model["prediction"]).all())
                self.assertTrue(lo[0] <= model["center"][0] <= hi[-1])
                self.assertTrue(lo[0] <= model["lower"][0] <= model["upper"][0] <= hi[-1])
            np.testing.assert_allclose(out["bayes_step"]["gap_probability"].sum(axis=1), 1.)

    def test_sigmoid_diagnostic_rejects_small_amplitude_and_boundary_center(self):
        # Keep exact, high-quality curve fits while changing one criterion.
        z = np.r_[np.arange(10., 91., 10.), 95.]
        y = np.vstack([300. * expit((55. - z) / 3.),
                       99. * expit((55. - z) / 3.),
                       800. * expit((20. - z) / 3.),
                       800. * expit((90. - z) / 3.)])
        sigmoid = fit_models(y, z)["sigmoid"]
        np.testing.assert_array_equal(sigmoid["diagnostic_pass"], [True, False, False, False])

    def test_sigmoid_diagnostic_rejects_wide_interval(self):
        z = np.r_[np.arange(10., 91., 10.), 95.]
        y = 800. * expit((55. - z) / 5.)
        sigmoid = fit_models(y, z, {"sigma": 1000.})["sigmoid"]
        self.assertGreater(sigmoid["amplitude"][0], 100.)
        self.assertGreater(sigmoid["upper"][0] - sigmoid["lower"][0], 20.)
        self.assertFalse(sigmoid["diagnostic_pass"][0])

    def test_step_midpoint_tie_is_assigned_to_deep_plateau(self):
        z = np.array([0., 10., 30., 50.])
        y = np.array([300., 300., -100., -100.])
        out = fit_models(y, z, {"center_bounds": [0., 50.]},
                         query_depth=[19.9, 20., 20.1])["step"]
        np.testing.assert_allclose(out["prediction"], [[300., -100., -100.]])

    def test_invalid_data_is_rejected_not_silently_interpolated(self):
        with self.assertRaises(ValueError):
            fit_models(np.r_[np.nan, np.zeros(len(self.z)-1)], self.z, self.cfg)
        with self.assertRaises(ValueError):
            fit_models(np.zeros(len(self.z)), self.z[::-1], self.cfg)
        with self.assertRaises(ValueError):
            fit_models(np.zeros(len(self.z)), self.z, {"center_bounds": [150., 160.]})


if __name__ == "__main__":
    unittest.main()
