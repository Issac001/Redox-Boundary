"""Direct one-transition profile models with reusable, vectorized grid fitting.

The fitted location is an *Eh transition*, not a named reaction front. The
sigmoid assumes an oxidized upper plateau and reduced lower plateau. Its width
parameter is a logistic scale: the 10--90% transition spans 2*log(9)*w.
No chemical observations, time smoothing, or fixed Eh threshold enter fitting.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import chi2


def _depth(values):
    z = np.asarray(values, dtype=float)
    if z.ndim != 1 or len(z) < 3 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("depth must contain >=3 finite, strictly increasing values")
    return z


def _profiles(y, depth):
    values = np.asarray(y, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(depth) or len(values) == 0:
        raise ValueError("y must have shape (n_profiles, n_depths), or (n_depths,)")
    if not np.isfinite(values).all():
        raise ValueError("Missing depths require an explicit subset and a new grid")
    return values


class TransitionGrid:
    """Variable-projection fit of a + A*expit((b-z)/w), constrained A >= 0.

    Candidate selection minimizes SSE. All fit outputs keep a leading profile
    dimension, even for a one-dimensional input. `identifiable` only
    rules out numerically zero amplitude/contrast; it is not a significance test.
    """

    def __init__(self, depth, b_grid=None, w_grid=None):
        self.depth = _depth(depth)
        self.b_grid = np.asarray(np.arange(15., 91.) if b_grid is None else b_grid, dtype=float)
        self.w_grid = np.asarray([1., 2., 3., 5., 8., 12., 18.] if w_grid is None else w_grid, dtype=float)
        for name, grid in (("b", self.b_grid), ("w", self.w_grid)):
            if grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
                raise ValueError(f"{name}_grid must be finite and strictly increasing")
        if np.any(self.w_grid <= 0):
            raise ValueError("w_grid must be positive")
        self._b, self._w = (v.ravel() for v in np.meshgrid(self.b_grid, self.w_grid, indexing="ij"))
        self._s = expit((self._b[:, None] - self.depth) / self._w[:, None])
        self._s_mean = self._s.mean(axis=1)
        self._sc = self._s - self._s_mean[:, None]
        self._ss = np.einsum("gd,gd->g", self._sc, self._sc)
        self._chunk_size = 1024

    def _evaluate(self, y):
        yc = y - y.mean(axis=1, keepdims=True)
        sy = yc @ self._sc.T
        yy = np.einsum("nd,nd->n", yc, yc)[:, None]
        amplitude = np.maximum(sy / np.maximum(self._ss, 1e-14), 0.)
        amplitude[:, self._ss < 1e-14] = 0.
        sse = yy - 2 * amplitude * sy + amplitude ** 2 * self._ss
        return amplitude, np.maximum(sse, 0.)

    def _selected(self, y, evaluated):
        amplitude, sse = evaluated
        pick = np.argmin(sse, axis=1)
        row = np.arange(len(y))
        a = amplitude[row, pick]
        b, w = self._b[pick].copy(), self._w[pick].copy()
        intercept = y.mean(axis=1) - a * self._s_mean[pick]
        tolerance = 1e-9 * np.maximum(1., np.max(np.abs(y), axis=1))
        span = np.ptp(self._s[pick], axis=1)
        identified = (a > tolerance) & (a * span > tolerance)
        prediction = intercept[:, None] + a[:, None] * self._s[pick]
        b_edge = (b == self.b_grid[0]) | (b == self.b_grid[-1])
        w_edge = (w == self.w_grid[0]) | (w == self.w_grid[-1])
        b[~identified], w[~identified] = np.nan, np.nan
        return {
            "b": b, "w": w, "amplitude": a, "intercept": intercept,
            "prediction": prediction, "sse": sse[row, pick],
            "identifiable": identified, "grid_edge": b_edge | w_edge,
            "b_grid_edge": b_edge, "width_grid_edge": w_edge,
            "saturated": span < .1,
            "plateau_unobserved": (self._s[pick].max(axis=1) < .9) | (self._s[pick].min(axis=1) > .1),
        }

    def fit(self, y):
        """Fit all profiles, independently, without pooling observations in time."""
        y = _profiles(y, self.depth)
        pieces = [self._selected(part, self._evaluate(part))
                  for part in (y[i:i + self._chunk_size] for i in range(0, len(y), self._chunk_size))]
        return {key: np.concatenate([part[key] for part in pieces], axis=0) for key in pieces[0]}

    def predict(self, fit, depth=None):
        """Predict at observed or arbitrary depths from the fitted coefficients."""
        z = self.depth if depth is None else np.atleast_1d(np.asarray(depth, dtype=float))
        if z.ndim != 1 or not np.isfinite(z).all():
            raise ValueError("prediction depth must be finite and one-dimensional")
        # Null-amplitude fits deliberately expose NaN b/w; their mean remains valid.
        b = np.nan_to_num(np.atleast_1d(fit["b"]), nan=self.b_grid[0])
        w = np.nan_to_num(np.atleast_1d(fit["w"]), nan=self.w_grid[0])
        return (np.atleast_1d(fit["intercept"])[:, None]
                + np.atleast_1d(fit["amplitude"])[:, None] * expit((b[:, None] - z) / w[:, None]))

    def profile_interval(self, y, sigma=25., level=.95):
        """Grid profile-SSE working interval; this is not a coverage guarantee.

        Use min_w SSE(b,w)-min_b,w SSE <= sigma²*chi2_1(level),
        assuming known independent Gaussian observation errors. The asymptotic
        approximation need not hold with ten depths, bounded amplitude, discrete
        width candidates, model misspecification or spatial dependence.
        Endpoints hitting the search grid are flagged as truncated. Zero-amplitude
        fits return NaN endpoints, not an arbitrary confidence region for b.
        """
        if not np.isfinite(sigma) or sigma <= 0 or not 0 < level < 1:
            raise ValueError("sigma > 0 and 0 < level < 1 are required")
        y = _profiles(y, self.depth)
        threshold = float(sigma ** 2 * chi2.ppf(level, 1))
        pieces = []
        for start in range(0, len(y), self._chunk_size):
            part = y[start:start + self._chunk_size]
            evaluated = self._evaluate(part)
            out = self._selected(part, evaluated)
            prof = evaluated[1].reshape(len(part), len(self.b_grid), len(self.w_grid)).min(axis=2)
            support = prof <= prof.min(axis=1, keepdims=True) + threshold
            low = self.b_grid[np.argmax(support, axis=1)].copy()
            high = self.b_grid[len(self.b_grid) - 1 - np.argmax(support[:, ::-1], axis=1)].copy()
            low[~out["identifiable"]] = np.nan
            high[~out["identifiable"]] = np.nan
            out.update(b_low=low, b_high=high, low=low, high=high,
                       grid_truncated=support[:, 0] | support[:, -1],
                       full_grid_support=support.all(axis=1),
                       profile_sse=prof, support=support,
                       # Multiple disconnected grid intervals must not be mistaken
                       # for a single connected confidence set. low/high is its hull.
                       support_components=(support[:, 0].astype(int)
                                           + np.sum(support[:, 1:] & ~support[:, :-1], axis=1)))
            pieces.append(out)
        result = {key: np.concatenate([part[key] for part in pieces], axis=0) for key in pieces[0]}
        result.update(b_grid=self.b_grid.copy(), threshold_sse=threshold,
                      method="profile_sse")
        return result
