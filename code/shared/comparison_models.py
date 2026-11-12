"""Common profile models for the reproducible boundary comparison.

All methods use the same observed depths and the same admissible boundary
support. No interpolated observations are introduced. The existing shared
``boundary_model.py`` supplies the manuscript's unchanged sigmoid core.

The two frequentist intervals are profile-SSE working sets, not guaranteed
confidence intervals. The Bayesian interval is conditional on exactly one
descending step and includes uniform uncertainty within each unsampled gap.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import erfcx, log_ndtr, logsumexp
from scipy.stats import chi2

from .boundary_model import TransitionGrid


DEFAULT_CONFIG = {
    "center_bounds": [20.0, 90.0], "grid_step": 0.1,
    "width_grid": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0],
    "min_side": 2, "sigma": 25.0, "level": 0.95,
    "prior_mean": 0.0, "prior_sd": 500.0,
}


def admissible_gaps(z, config=None):
    """Return split counts and clipped physical gaps, with >= min_side points.

    A split count k assigns the first k observations to the shallow plateau.
    Endpoints are locations in physical depth units, not candidate grid indices.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("z must be finite, one-dimensional, and strictly increasing")
    min_side = int(cfg["min_side"])
    if min_side != cfg["min_side"] or min_side < 1 or len(z) < max(3, 2 * min_side):
        raise ValueError("min_side must be a positive integer supported by the data")
    bounds = np.asarray(cfg["center_bounds"], dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("center_bounds must be two finite increasing values")
    split = np.arange(min_side, len(z) - min_side + 1)
    lower = np.maximum(z[split - 1], bounds[0])
    upper = np.minimum(z[split], bounds[1])
    valid = upper > lower
    if not np.any(valid):
        raise ValueError("No admissible physical boundary interval remains")
    return split[valid], lower[valid], upper[valid]


@lru_cache(maxsize=32)
def _transition_grid(depth, low, high, step, widths):
    # Keep the requested step and include the shared upper endpoint exactly.
    grid = low + np.arange(int(np.floor((high - low) / step)) + 1) * step
    grid = grid[grid <= high + 1e-10]
    if high - grid[-1] > 1e-9:
        grid = np.r_[grid, high]
    else:
        grid[-1] = high
    return TransitionGrid(np.asarray(depth), b_grid=grid, w_grid=np.asarray(widths))


def _gap_index(center, lower, upper):
    """Zero-based admissible gap containing the point; observed ties go deeper."""
    center = np.asarray(center)
    index = np.searchsorted(upper, center, side="right")
    index = np.minimum(index, len(lower) - 1)
    valid = np.isfinite(center) & (center >= lower[0]) & (center <= upper[-1])
    return np.where(valid, index, -1)


def _normal_mean_evidence(y, sigma, prior_mean, prior_sd):
    """Analytic Gaussian marginal likelihood and unconstrained mean posterior."""
    n = y.shape[1]
    mean = y.mean(axis=1)
    within = np.sum((y - mean[:, None]) ** 2, axis=1)
    variance, prior_variance = sigma ** 2, prior_sd ** 2
    posterior_variance = 1.0 / (1.0 / prior_variance + n / variance)
    posterior_mean = posterior_variance * (prior_mean / prior_variance + n * mean / variance)
    log_evidence = (-0.5 * n * np.log(2 * np.pi * variance)
                    -0.5 * np.log1p(n * prior_variance / variance)
                    -0.5 * (within / variance
                            + n * (mean - prior_mean) ** 2 / (variance + n * prior_variance)))
    return log_evidence, posterior_mean, posterior_variance


def _inverse_mills(a):
    """phi(a)/Phi(a), including strongly ascending observations without overflow."""
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    negative = a < -5.0
    out[negative] = np.sqrt(2.0 / np.pi) / erfcx(-a[negative] / np.sqrt(2.0))
    out[~negative] = np.exp(-0.5 * a[~negative] ** 2
                           -0.5 * np.log(2 * np.pi) - log_ndtr(a[~negative]))
    return out


def _ordered_segment_posterior(y, split, sigma, prior_mean, prior_sd):
    """Evidence and means under independent normal priors conditioned on U > L.

    The ordering prior has probability 1/2 because both original priors are
    identical. Consequently the evidence correction is 2*P(U>L | data),
    not just the posterior ordering probability.
    """
    evidence, upper_mean, lower_mean = [], [], []
    for k in split:
        log_u, mean_u, var_u = _normal_mean_evidence(y[:, :k], sigma, prior_mean, prior_sd)
        log_l, mean_l, var_l = _normal_mean_evidence(y[:, k:], sigma, prior_mean, prior_sd)
        scale = np.sqrt(var_u + var_l)
        a = (mean_u - mean_l) / scale
        shift = _inverse_mills(a) / scale
        evidence.append(log_u + log_l + np.log(2.0) + log_ndtr(a))
        upper_mean.append(mean_u + var_u * shift)
        lower_mean.append(mean_l - var_l * shift)
    return np.column_stack(evidence), np.column_stack(upper_mean), np.column_stack(lower_mean)


def _mixture_quantile(probability, lower, upper, q):
    """Quantile of ordered, disjoint uniform components (one per physical gap)."""
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.0
    index = np.argmax(cumulative >= q, axis=1)
    row = np.arange(len(probability))
    previous = np.column_stack([np.zeros(len(probability)), cumulative[:, :-1]])
    fraction = (q - previous[row, index]) / probability[row, index]
    return lower[index] + np.clip(fraction, 0.0, 1.0) * (upper[index] - lower[index])


def _base_result(n, prediction, sse):
    return {
        "center": np.full(n, np.nan), "lower": np.full(n, np.nan),
        "upper": np.full(n, np.nan), "prediction": prediction, "sse": sse,
        "amplitude": np.full(n, np.nan), "width": np.full(n, np.nan),
        "width_edge": np.zeros(n, dtype=bool),
        "center_edge": np.zeros(n, dtype=bool), "interval_components": np.zeros(n, dtype=int),
        "center_map": np.full(n, np.nan), "center_median": np.full(n, np.nan),
        "gap_index": np.full(n, -1, dtype=int), "interval_kind": "none",
    }


def fit_models(y, z, config=None, query_depth=None):
    """Fit the three manuscript models, retaining a leading profile dimension throughout.

    ``y`` is (profile, observed_depth) or a single profile. Missing depths must be
    removed explicitly in both y and z before calling, including leave-depth-out
    fits. Predictions are returned at query_depth or, by default, observed z.

    ``center_map`` for bayes_step is the midpoint of the largest-posterior-mass
    gap; on uneven designs this differs from a highest-density gap. Its main
    center is the posterior mean. Bayesian SSE evaluates the posterior-mean
    prediction and is not an MLE objective.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    z = np.asarray(z, dtype=float)
    split, gap_lower, gap_upper = admissible_gaps(z, cfg)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[None, :]
    if y.ndim != 2 or y.shape[1] != len(z) or not len(y) or not np.isfinite(y).all():
        raise ValueError("y must be finite with shape (n_profiles, len(z))")
    query = z if query_depth is None else np.atleast_1d(np.asarray(query_depth, dtype=float))
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query_depth must be a finite one-dimensional array")
    sigma, level = float(cfg["sigma"]), float(cfg["level"])
    prior_sd = float(cfg["prior_sd"])
    prior_mean, step_size = float(cfg["prior_mean"]), float(cfg["grid_step"])
    if not all(np.isfinite([sigma, prior_sd, prior_mean, step_size])):
        raise ValueError("Scale and prior parameters must be finite")
    if sigma <= 0 or prior_sd <= 0 or step_size <= 0 or not 0 < level < 1:
        raise ValueError("Positive scales and probabilities strictly between 0 and 1 are required")
    n_profiles, n_depth = y.shape
    row = np.arange(n_profiles)
    midpoint = (gap_lower + gap_upper) / 2.0
    tolerance = 1e-9 * np.maximum(1.0, np.max(np.abs(y), axis=1))
    y_mean = y.mean(axis=1)
    # The original sigmoid estimator and working interval are reused unchanged.
    grid = _transition_grid(tuple(z), float(gap_lower[0]), float(gap_upper[-1]),
                            step_size, tuple(cfg["width_grid"]))
    fitted = grid.profile_interval(y, sigma=sigma, level=level)
    sigmoid = _base_result(n_profiles, grid.predict(fitted, query), fitted["sse"])
    sigmoid.update(center=fitted["b"], lower=fitted["low"], upper=fitted["high"],
                   amplitude=fitted["amplitude"], width=fitted["w"],
                   width_edge=fitted["width_grid_edge"] & fitted["identifiable"],
                   center_edge=fitted["b_grid_edge"] & fitted["identifiable"],
                   interval_components=np.where(fitted["identifiable"], fitted["support_components"], 0),
                   center_map=fitted["b"].copy(), center_median=fitted["b"].copy(),
                   interval_kind="profile_sse_grid_hull",
                   interval_truncated=fitted["grid_truncated"], identifiable=fitted["identifiable"])
    # The manuscript's three joint diagnostic criteria concern sigmoid only.
    sigmoid["diagnostic_pass"] = (np.isfinite(fitted["b"])
                                  & (fitted["amplitude"] >= 100.0)
                                  & ((fitted["high"] - fitted["low"]) <= 20.0)
                                  & (fitted["b"] > gap_lower[0])
                                  & (fitted["b"] < gap_upper[-1]))

    # Descending least-squares step: violating candidates reduce to one mean.
    shallow = np.column_stack([y[:, :k].mean(axis=1) for k in split])
    deep = np.column_stack([y[:, k:].mean(axis=1) for k in split])
    descending = shallow > deep
    shallow = np.where(descending, shallow, y_mean[:, None])
    deep = np.where(descending, deep, y_mean[:, None])
    step_sse = np.column_stack([
        np.sum((y[:, :k] - shallow[:, j, None]) ** 2, axis=1)
        + np.sum((y[:, k:] - deep[:, j, None]) ** 2, axis=1)
        for j, k in enumerate(split)])
    selected = np.argmin(step_sse, axis=1)
    chosen_u, chosen_l = shallow[row, selected], deep[row, selected]
    amplitude = chosen_u - chosen_l
    identified = amplitude > tolerance
    center = np.where(identified, midpoint[selected], np.nan)
    prediction = np.where(query[None, :] < midpoint[selected, None], chosen_u[:, None], chosen_l[:, None])
    step = _base_result(n_profiles, prediction, step_sse[row, selected])
    supported = step_sse <= step["sse"][:, None] + sigma ** 2 * chi2.ppf(level, 1)
    low_index = np.argmax(supported, axis=1)
    high_index = len(split) - 1 - np.argmax(supported[:, ::-1], axis=1)
    low = np.where(identified, gap_lower[low_index], np.nan)
    high = np.where(identified, gap_upper[high_index], np.nan)
    components = supported[:, 0].astype(int) + np.sum(supported[:, 1:] & ~supported[:, :-1], axis=1)
    step.update(center=center, lower=low, upper=high, amplitude=amplitude,
                center_edge=identified & ((selected == 0) | (selected == len(split) - 1)),
                interval_components=np.where(identified, components, 0),
                center_map=center.copy(), center_median=center.copy(), identifiable=identified,
                interval_kind="profile_sse_full_gap_hull",
                gap_index=np.where(identified, selected, -1),
                interval_truncated=supported[:, 0] | supported[:, -1])

    # Bayesian single step with a proper descending prior, fully marginalized.
    log_evidence, post_u, post_l = _ordered_segment_posterior(y, split, sigma, prior_mean, prior_sd)
    gap_length = gap_upper - gap_lower
    log_prior = np.log(gap_length / gap_length.sum())
    log_joint = log_evidence + log_prior[None, :]
    log_k1 = logsumexp(log_joint, axis=1)
    probability = np.exp(log_joint - log_k1[:, None])
    probability /= probability.sum(axis=1, keepdims=True)

    def predict_bayes(depth):
        # At a query inside a gap, integrate the unknown boundary location.
        prob_shallow = np.clip((gap_upper[:, None] - depth[None, :]) / gap_length[:, None], 0.0, 1.0)
        return np.sum(probability * post_l, axis=1)[:, None] + (probability * (post_u - post_l)) @ prob_shallow

    bayes_prediction = predict_bayes(query)
    bayes_sse = np.sum((y - predict_bayes(z)) ** 2, axis=1)
    bayes = _base_result(n_profiles, bayes_prediction, bayes_sse)
    tail = (1.0 - level) / 2.0
    low = _mixture_quantile(probability, gap_lower, gap_upper, tail)
    high = _mixture_quantile(probability, gap_lower, gap_upper, 1.0 - tail)
    map_index = np.argmax(probability, axis=1)
    amplitude = np.sum(probability * (post_u - post_l), axis=1)
    bayes.update(center=probability @ midpoint, lower=low, upper=high, amplitude=amplitude,
                 center_map=midpoint[map_index], center_median=_mixture_quantile(probability, gap_lower, gap_upper, 0.5),
                 center_edge=(map_index == 0) | (map_index == len(split) - 1),
                 interval_components=np.ones(n_profiles, dtype=int),
                 gap_probability=probability,
                 gap_lower=gap_lower.copy(), gap_upper=gap_upper.copy(), map_gap_index=map_index,
                 plateau_upper_by_gap=post_u, plateau_lower_by_gap=post_l,
                 log_evidence_k1=log_k1,
                 interval_kind="conditional_k1_equal_tail_credible")
    result = {"sigmoid": sigmoid, "step": step, "bayes_step": bayes}
    for method in result.values():
        method["support_lower"] = float(gap_lower[0])
        method["support_upper"] = float(gap_upper[-1])
        method["gap_index"] = _gap_index(method["center"], gap_lower, gap_upper)
        method["interval_width"] = method["upper"] - method["lower"]
    return result
