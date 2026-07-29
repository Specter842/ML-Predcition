"""Vintage-aware FRED/ALFRED puller.

The central object here is :class:`VintageStore`, which answers exactly one
question: *what did series X look like to someone standing on date D?*

How it works
------------
FRED's ``series/observations`` endpoint, given a wide ``realtime_start`` /
``realtime_end`` window, returns the series' full revision history in long
form: one row per (observation period, real-time interval, value). A value that
was never revised produces one row; a value revised four times produces four
rows with adjoining real-time intervals.

That single response is enough to reconstruct *any* vintage locally::

    vintage(D) = rows where realtime_start <= D <= realtime_end

which is why we pull once per series and cache, rather than issuing one request
per (series, as_of_date) pair — the latter would be tens of thousands of calls
for a full backtest.

``realtime_start`` is the date a value first appeared in FRED, i.e. its
publication date. Filtering on it is therefore what enforces the no-leakage
constraint, and it also handles publication lags for free: an observation for
2024-03 with realtime_start 2024-04-10 is simply invisible to any as_of date
before 2024-04-10.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from src.config import SERIES, VINTAGES, SeriesSpec, fred_api_key

API_ROOT = "https://api.stlouisfed.org/fred"

# FRED's earliest/latest sentinel real-time dates. Requesting this full window
# is what makes the response a complete revision history rather than a snapshot.
REALTIME_MIN = "1776-07-04"
REALTIME_MAX = "9999-12-31"

_PAGE_LIMIT = 100_000


class FredError(RuntimeError):
    """A FRED API call failed in a way retrying will not fix."""


@dataclass
class _Fetcher:
    api_key: str
    session: requests.Session
    max_retries: int = 4
    backoff: float = 1.5

    def get(self, endpoint: str, **params) -> dict:
        params = {
            **params,
            "api_key": self.api_key,
            "file_type": "json",
        }
        url = f"{API_ROOT}/{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:  # transport-level
                last_exc = exc
                time.sleep(self.backoff**attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:  # rate limited — always worth retrying
                time.sleep(self.backoff ** (attempt + 1))
                last_exc = FredError("rate limited")
                continue
            # 400s carry a useful message from FRED; surface it rather than retry.
            raise FredError(
                f"FRED {endpoint} returned {resp.status_code} for "
                f"{params.get('series_id', '')}: {resp.text[:400]}"
            )
        raise FredError(f"FRED {endpoint} failed after {self.max_retries} attempts") from last_exc


def _cache_path(series_id: str):
    return VINTAGES / f"{series_id}.parquet"


def fetch_revision_history(
    series_id: str,
    *,
    fetcher: _Fetcher | None = None,
    observation_start: str | None = None,
) -> pd.DataFrame:
    """Download the complete real-time revision history for one series.

    Returns a frame with columns ``[date, realtime_start, realtime_end, value]``
    sorted by (date, realtime_start). ``date`` is the observation period;
    ``realtime_start``/``realtime_end`` bound the window during which that value
    was the published one.
    """
    if fetcher is None:
        fetcher = _Fetcher(fred_api_key(), requests.Session())

    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "series_id": series_id,
            "realtime_start": REALTIME_MIN,
            "realtime_end": REALTIME_MAX,
            "output_type": 1,  # observations by real-time period (long form)
            "limit": _PAGE_LIMIT,
            "offset": offset,
        }
        if observation_start:
            params["observation_start"] = observation_start
        payload = fetcher.get("series/observations", **params)
        batch = payload.get("observations", [])
        rows.extend(batch)
        count = payload.get("count", len(rows))
        offset += len(batch)
        if not batch or offset >= count:
            break

    if not rows:
        raise FredError(f"No observations returned for {series_id}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    # 9999-12-31 overflows pandas' ns-resolution Timestamp; clamp to max.
    df["realtime_end"] = pd.to_datetime(
        df["realtime_end"].replace(REALTIME_MAX, "2262-04-11")
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
    df = df.dropna(subset=["value"])
    df = df[["date", "realtime_start", "realtime_end", "value"]]
    return df.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def download_series(series_id: str, *, force: bool = False, quiet: bool = False) -> pd.DataFrame:
    """Fetch and cache one series' revision history; reuse the cache if present."""
    path = _cache_path(series_id)
    if path.exists() and not force:
        return pd.read_parquet(path)
    df = fetch_revision_history(series_id)
    df.to_parquet(path, index=False)
    if not quiet:
        span = f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}"
        print(f"  {series_id:<12} {len(df):>7,} rows  {span}")
    return df


class VintageStore:
    """Point-in-time access to the cached FRED series.

    All reads go through :meth:`as_of`, which is the only place in the codebase
    permitted to decide what was knowable when.
    """

    def __init__(self, specs: dict[str, SeriesSpec] | None = None, *, force: bool = False):
        self.specs = specs if specs is not None else SERIES
        self._history: dict[str, pd.DataFrame] = {}
        self._arrays_cache: dict[str, tuple[np.ndarray, ...]] = {}
        self._force = force

    @classmethod
    def from_frames(
        cls, frames: dict[str, pd.DataFrame], specs: dict[str, SeriesSpec] | None = None
    ) -> "VintageStore":
        """Build a store from in-memory revision histories.

        Used by the test suite to exercise the point-in-time logic against
        synthetic data with known revision structure — no network, no API key,
        and no dependence on what FRED happens to hold today.
        """
        store = cls(specs if specs is not None else SERIES)
        store._history = {k: v.copy() for k, v in frames.items()}
        return store

    # -- loading ---------------------------------------------------------

    def history(self, series_id: str) -> pd.DataFrame:
        """Full revision history for a series, loaded lazily and memoised."""
        if series_id not in self._history:
            path = _cache_path(series_id)
            if path.exists() and not self._force:
                self._history[series_id] = pd.read_parquet(path)
            else:
                self._history[series_id] = download_series(series_id, force=self._force)
        return self._history[series_id]

    def available(self) -> list[str]:
        return [sid for sid in self.specs if _cache_path(sid).exists()]

    # -- point-in-time reads ---------------------------------------------

    def _arrays(self, series_id: str) -> tuple[np.ndarray, ...]:
        """Revision history as raw numpy arrays, memoised.

        A backtest issues tens of thousands of :meth:`as_of` calls, and pandas
        boolean indexing on a DataFrame dominates the runtime at that volume.
        Comparing datetime64 arrays directly is roughly two orders of magnitude
        faster and returns the identical rows.
        """
        if series_id not in self._arrays_cache:
            h = self.history(series_id)
            self._arrays_cache[series_id] = (
                h["date"].to_numpy(dtype="datetime64[ns]"),
                h["realtime_start"].to_numpy(dtype="datetime64[ns]"),
                h["realtime_end"].to_numpy(dtype="datetime64[ns]"),
                h["value"].to_numpy(dtype="float64"),
            )
        return self._arrays_cache[series_id]

    def as_of(self, series_id: str, as_of_date) -> pd.Series:
        """The series exactly as it was published on ``as_of_date``.

        Applies the series' ``extra_lag_days`` cushion, so the effective
        knowledge date is ``as_of_date - extra_lag_days``.
        """
        as_of_date = pd.Timestamp(as_of_date)
        spec = self.specs.get(series_id)
        if spec is not None and spec.extra_lag_days:
            as_of_date = as_of_date - pd.Timedelta(days=spec.extra_lag_days)

        dates, rt_start, rt_end, values = self._arrays(series_id)
        stamp = np.datetime64(as_of_date, "ns")
        live = (rt_start <= stamp) & (rt_end >= stamp)
        if not live.any():
            return pd.Series(dtype="float64", name=series_id)

        d, v = dates[live], values[live]
        order = np.argsort(d, kind="stable")
        d, v = d[order], v[order]
        # One row per observation date by construction; keep the last if not.
        keep = np.empty(len(d), dtype=bool)
        keep[:-1] = d[:-1] != d[1:]
        keep[-1] = True
        return pd.Series(
            v[keep], index=pd.DatetimeIndex(d[keep], name="date"), name=series_id
        )

    def first_print(self, series_id: str) -> pd.Series:
        """Each observation's *initial* published value, ignoring later revisions."""
        hist = self.history(series_id)
        first = hist.sort_values("realtime_start").drop_duplicates(subset="date", keep="first")
        return first.set_index("date")["value"].sort_index().rename(f"{series_id}_first")

    def release_dates(self, series_id: str) -> pd.Series:
        """The date each observation period was first published.

        Derived from the data rather than a hardcoded BLS calendar, so it stays
        correct through schedule changes and historical quirks.
        """
        hist = self.history(series_id)
        rel = hist.groupby("date")["realtime_start"].min()
        return rel.sort_index().rename(f"{series_id}_release")

    def latest_vintage_date(self, series_id: str) -> pd.Timestamp:
        """Most recent date on which this series was updated."""
        return self.history(series_id)["realtime_start"].max()


def download_all(force: bool = False) -> None:
    """Populate the vintage cache for every registered series."""
    print(f"Caching revision histories into {VINTAGES}")
    failures: list[tuple[str, str]] = []
    for sid in SERIES:
        try:
            download_series(sid, force=force)
        except Exception as exc:  # keep going; report at the end
            print(f"  {sid:<12} FAILED: {exc}")
            failures.append((sid, str(exc)))
    if failures:
        print(f"\n{len(failures)} series failed: {', '.join(s for s, _ in failures)}")
    else:
        print(f"\nAll {len(SERIES)} series cached.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache ALFRED revision histories.")
    ap.add_argument("--all", action="store_true", help="download every registered series")
    ap.add_argument("--series", nargs="*", help="specific series ids")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    if args.all or not args.series:
        download_all(force=args.force)
    else:
        for s in args.series:
            download_series(s, force=args.force)
