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

    kind='sigmoid_trend' adds c*(z-mean(depth))/85 and fixes a ridge penalty
    c**2 (lambda=1); intercept and amplitude remain unpenalized. Candidate
    selection minimizes SSE plus that penalty. All fit outputs keep a leading
    profile dimension, even for a one-dimensional input. `identifiable` only
    rules out numerically zero amplitude/contrast; it is not a significance test.
    """

    def __init__(self, depth, b_grid=None, w_grid=None, kind="sigmoid"):
        self.depth = _depth(depth)
        self.kind = kind
        if kind not in ("sigmoid", "sigmoid_trend"):
            raise ValueError("kind must be sigmoid or sigmoid_trend")
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
        self.depth_center = float(self.depth.mean())
        self._x = (self.depth - self.depth_center) / 85.
        self._xx = float(self._x @ self._x)
        self._sx = self._sc @ self._x
        self.ridge = 1. if kind == "sigmoid_trend" else 0.
        self._chunk_size = 1024

    def _evaluate(self, y):
        yc = y - y.mean(axis=1, keepdims=True)
        sy = yc @ self._sc.T
        yy = np.einsum("nd,nd->n", yc, yc)[:, None]
        if self.kind == "sigmoid":
            amplitude = np.maximum(sy / np.maximum(self._ss, 1e-14), 0.)
            amplitude[:, self._ss < 1e-14] = 0.
            trend = np.zeros_like(amplitude)
            sse = yy - 2 * amplitude * sy + amplitude ** 2 * self._ss
        else:
            xy = (yc @ self._x)[:, None]
            det = self._ss * (self._xx + self.ridge) - self._sx ** 2
            amplitude = np.maximum((sy * (self._xx + self.ridge) - xy * self._sx)
                                   / np.maximum(det, 1e-14), 0.)
            amplitude[:, det < 1e-14] = 0.
            trend = (xy - amplitude * self._sx) / (self._xx + self.ridge)
            sse = (yy - 2 * amplitude * sy - 2 * trend * xy
                   + amplitude ** 2 * self._ss + 2 * amplitude * trend * self._sx
                   + trend ** 2 * self._xx)
        sse = np.maximum(sse, 0.)
        return amplitude, trend, sse, sse + self.ridge * trend ** 2

    def _selected(self, y, evaluated):
        amplitude, trend, sse, objective = evaluated
        pick = np.argmin(objective, axis=1)
        row = np.arange(len(y))
        a, c = amplitude[row, pick], trend[row, pick]
        b, w = self._b[pick].copy(), self._w[pick].copy()
        intercept = y.mean(axis=1) - a * self._s_mean[pick]
        tolerance = 1e-9 * np.maximum(1., np.max(np.abs(y), axis=1))
        span = np.ptp(self._s[pick], axis=1)
        identified = (a > tolerance) & (a * span > tolerance)
        prediction = intercept[:, None] + a[:, None] * self._s[pick] + c[:, None] * self._x
        b_edge = (b == self.b_grid[0]) | (b == self.b_grid[-1])
        w_edge = (w == self.w_grid[0]) | (w == self.w_grid[-1])
        b[~identified], w[~identified] = np.nan, np.nan
        return {
            "b": b, "w": w, "amplitude": a, "intercept": intercept, "trend": c,
            "prediction": prediction, "sse": sse[row, pick], "objective": objective[row, pick],
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
        """Predict from fitted coefficients, preserving the original trend center."""
        z = self.depth if depth is None else np.atleast_1d(np.asarray(depth, dtype=float))
        if z.ndim != 1 or not np.isfinite(z).all():
            raise ValueError("prediction depth must be finite and one-dimensional")
        # Null-amplitude fits deliberately expose NaN b/w; their mean remains valid.
        b = np.nan_to_num(np.atleast_1d(fit["b"]), nan=self.b_grid[0])
        w = np.nan_to_num(np.atleast_1d(fit["w"]), nan=self.w_grid[0])
        return (np.atleast_1d(fit["intercept"])[:, None]
                + np.atleast_1d(fit["amplitude"])[:, None] * expit((b[:, None] - z) / w[:, None])
                + np.atleast_1d(fit["trend"])[:, None] * (z - self.depth_center) / 85.)

    def profile_interval(self, y, sigma=25., level=.95):
        """Grid profile-SSE working interval; this is not a coverage guarantee.

        For sigmoid, use min_w SSE(b,w)-min_b,w SSE <= sigma²*chi2_1(level),
        assuming known independent Gaussian observation errors. The asymptotic
        approximation need not hold with ten depths, bounded amplitude, discrete
        width candidates, model misspecification or spatial dependence. For trend,
        the penalized objective replaces SSE and the interval is only heuristic.
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
            prof = evaluated[3].reshape(len(part), len(self.b_grid), len(self.w_grid)).min(axis=2)
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
                      method="profile_sse" if self.kind == "sigmoid" else "penalized_profile_heuristic")
        return result


def fit_step(y, depth):
    """One free-level change point, with >=2 observed depths on either side.

    b is the midpoint of the unsampled bracket [lower_gap, upper_gap]; the data
    cannot resolve a location inside that bracket. Both signs of amplitude are
    allowed and nonpositive_amplitude is explicitly flagged. A flat profile has
    no identifiable location and returns NaN b/bracket, but a valid prediction.
    """
    z = _depth(depth)
    if len(z) < 4:
        raise ValueError("step model requires at least four observed depths")
    y = _profiles(y, z)
    n = len(z)
    split = np.arange(2, n - 1)
    summed = np.cumsum(y, axis=1)
    squared = np.cumsum(y ** 2, axis=1)
    left = summed[:, split - 1] / split
    right = (summed[:, -1, None] - summed[:, split - 1]) / (n - split)
    sse = np.maximum(squared[:, -1, None] - left ** 2 * split - right ** 2 * (n - split), 0.)
    chosen = np.argmin(sse, axis=1)
    row = np.arange(len(y))
    k = split[chosen]
    l, r = left[row, chosen], right[row, chosen]
    amplitude = l - r
    tolerance = 1e-9 * np.maximum(1., np.max(np.abs(y), axis=1))
    identified = np.abs(amplitude) > tolerance
    lower, upper = z[k - 1].copy(), z[k].copy()
    b = (lower + upper) / 2
    b[~identified], lower[~identified], upper[~identified] = np.nan, np.nan, np.nan
    return {"b": b, "lower_gap": lower, "upper_gap": upper,
            "left_mean": l, "right_mean": r, "amplitude": amplitude,
            "prediction": np.where(np.arange(n)[None, :] < k[:, None], l[:, None], r[:, None]),
            "sse": sse[row, chosen], "nonpositive_amplitude": amplitude <= tolerance,
            "identifiable": identified}
