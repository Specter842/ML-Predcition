"""Tests for equal forecast accuracy.

Two tests live here, and using the right one matters more than most people
admit.

**Diebold-Mariano** tests whether two forecasts have equal expected loss. It
assumes the two forecasts come from *non-nested* models. The brief requires a
DM p-value against every benchmark, so DM is what the headline table reports.

**Clark-West** exists because the most important comparison in this project —
challenger versus random walk — is usually *nested*: a regression that includes
lagged inflation among its features contains the random walk as a special case.
Under the null that the extra parameters are useless, the larger model still
has to estimate them, so its out-of-sample squared error is biased *upward*.
DM therefore under-rejects: it makes a genuinely better model look
indistinguishable from persistence. Clark & West (2007) correct for exactly
that estimation noise.

Both are reported. Where they disagree, the nesting structure decides which to
believe, and the report labels which comparisons are nested.

Small samples
-------------
The Harvey-Leybourne-Newbold correction is applied by default and the statistic
is referred to a t-distribution rather than a normal. With 24-300 folds this is
not cosmetic — the uncorrected DM statistic over-rejects noticeably at the
lower end of that range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    """Outcome of a forecast-comparison test.

    ``statistic`` positive means the *first* forecast has higher loss, i.e. the
    second one is better. ``p_value`` is two-sided for DM and one-sided for
    Clark-West (which only tests "the larger model is better").
    """

    statistic: float
    p_value: float
    n: int
    test: str
    note: str = ""

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


def _newey_west_variance(d: np.ndarray, lags: int) -> float:
    """Long-run variance of ``d`` with a Bartlett kernel."""
    n = len(d)
    e = d - d.mean()
    gamma0 = float(e @ e) / n
    total = gamma0
    for lag in range(1, max(lags, 0) + 1):
        if lag >= n:
            break
        gamma = float(e[lag:] @ e[:-lag]) / n
        total += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    # A negative estimate is possible with a small sample; fall back to the
    # plain variance rather than returning a nonsense standard error.
    return total if total > 0 else gamma0


def diebold_mariano(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    *,
    horizon: int = 1,
    loss: str = "squared",
    hln: bool = True,
    lags: int | None = None,
) -> TestResult:
    """Diebold-Mariano test of equal predictive accuracy.

    A positive statistic means ``forecast_a`` is *worse* (higher loss) than
    ``forecast_b``. In this project ``forecast_a`` is the benchmark, so a
    positive, significant statistic is what "the model beat the benchmark"
    looks like.
    """
    actual = np.asarray(actual, dtype="float64")
    fa = np.asarray(forecast_a, dtype="float64")
    fb = np.asarray(forecast_b, dtype="float64")

    mask = np.isfinite(actual) & np.isfinite(fa) & np.isfinite(fb)
    actual, fa, fb = actual[mask], fa[mask], fb[mask]
    n = len(actual)
    if n < 8:
        return TestResult(np.nan, np.nan, n, "DM", "too few paired observations")

    ea, eb = actual - fa, actual - fb
    if loss == "squared":
        d = ea**2 - eb**2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    else:
        raise ValueError(f"unknown loss {loss!r}")

    if np.allclose(d, 0):
        return TestResult(0.0, 1.0, n, "DM", "forecasts are identical")

    nw_lags = (horizon - 1) if lags is None else lags
    var = _newey_west_variance(d, nw_lags)
    if var <= 0:
        return TestResult(np.nan, np.nan, n, "DM", "non-positive variance estimate")

    stat = d.mean() / np.sqrt(var / n)

    note = ""
    if hln:
        # Harvey, Leybourne & Newbold (1997) small-sample correction.
        adj = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        stat *= np.sqrt(max(adj, 1e-12))
        note = "HLN-corrected, t reference"

    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return TestResult(float(stat), float(p), n, "DM", note)


def clark_west(
    actual: np.ndarray,
    forecast_restricted: np.ndarray,
    forecast_unrestricted: np.ndarray,
    *,
    horizon: int = 1,
) -> TestResult:
    """Clark-West test for *nested* models.

    ``forecast_restricted`` is the smaller model (e.g. the random walk);
    ``forecast_unrestricted`` nests it. A positive, significant statistic means
    the larger model genuinely improves on the smaller one.

    One-sided by construction: the alternative is only ever "the larger model
    is better", since under the null the extra parameters are zero.
    """
    actual = np.asarray(actual, dtype="float64")
    fr = np.asarray(forecast_restricted, dtype="float64")
    fu = np.asarray(forecast_unrestricted, dtype="float64")

    mask = np.isfinite(actual) & np.isfinite(fr) & np.isfinite(fu)
    actual, fr, fu = actual[mask], fr[mask], fu[mask]
    n = len(actual)
    if n < 8:
        return TestResult(np.nan, np.nan, n, "CW", "too few paired observations")

    # (y-f_r)^2 - (y-f_u)^2 + (f_r-f_u)^2 — the last term removes the upward
    # bias in the larger model's squared error caused by estimating parameters
    # that are zero under the null.
    adjusted = (actual - fr) ** 2 - (actual - fu) ** 2 + (fr - fu) ** 2

    var = _newey_west_variance(adjusted, max(horizon - 1, 0))
    if var <= 0:
        return TestResult(np.nan, np.nan, n, "CW", "non-positive variance estimate")

    stat = adjusted.mean() / np.sqrt(var / n)
    p = 1.0 - stats.norm.cdf(stat)  # one-sided
    return TestResult(float(stat), float(p), n, "CW", "one-sided, nested comparison")
