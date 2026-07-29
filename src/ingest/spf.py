"""Philadelphia Fed Survey of Professional Forecasters — benchmark only.

SPF is a *competitor forecast*, never a model input. It exists here so the
question "does this model beat the people who do this for a living?" has an
answer.

The frequency mismatch, stated plainly
--------------------------------------
SPF forecasts **quarterly annualised** CPI inflation. Our target is a
**monthly** percent change. These are not the same quantity, and squeezing SPF
into the monthly comparison table would produce a number that looks like a
like-for-like benchmark but isn't.

So SPF is handled two ways, both labelled:

* ``quarterly_benchmark()`` — the honest comparison. Model nowcasts are
  aggregated to a quarterly annualised rate and scored against SPF's
  current-quarter median (``CPI2``) on the same quarters. This is the number
  that belongs in a write-up.
* ``implied_monthly()`` — SPF's quarterly rate converted to the equivalent
  constant monthly rate. Useful for plotting on the same axis as the monthly
  nowcast; explicitly *not* a fair scoring comparison, because SPF was never
  asked for a single month.

Real-time availability
----------------------
The SPF for quarter Q is surveyed and published in the middle month of Q
(February, May, August, November). We treat the survey as knowable from the
end of that middle month, which is a few days later than the actual release —
conservative in the right direction.
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from src.config import RAW

RAW_SPF = RAW / "spf"

_BASE = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files"
)

# variable key -> candidate filenames on the Philly Fed site
SPF_FILES = {
    "cpi": ["median_cpi_level.xlsx", "median_cpi_level.xls"],
    "corecpi": ["median_corecpi_level.xlsx", "median_corecpi_level.xls"],
}

# Middle month of each quarter — when the survey is fielded and published.
_SURVEY_MONTH = {1: 2, 2: 5, 3: 8, 4: 11}


class SPFUnavailable(RuntimeError):
    """The SPF file could not be downloaded or parsed."""


def download(variable: str = "cpi", force: bool = False) -> bytes:
    """Fetch an SPF median-forecast workbook, caching under data/raw/spf/."""
    RAW_SPF.mkdir(parents=True, exist_ok=True)
    cached = sorted(RAW_SPF.glob(f"median_{variable}_level.*"))
    if cached and not force:
        return cached[0].read_bytes()

    errors: list[str] = []
    for name in SPF_FILES.get(variable, []):
        url = f"{_BASE}/{name}"
        try:
            resp = requests.get(url, timeout=90, headers={"User-Agent": "inflation-nowcaster/1.0"})
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        if resp.status_code == 200 and resp.content:
            (RAW_SPF / name).write_bytes(resp.content)
            return resp.content
        errors.append(f"{url}: HTTP {resp.status_code}")

    raise SPFUnavailable(
        f"Could not download SPF '{variable}'. Tried:\n  " + "\n  ".join(errors) +
        "\nDownload the median level file manually from "
        "https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/"
        f"survey-of-professional-forecasters and save it to {RAW_SPF}"
    )


def parse(content: bytes, variable: str = "cpi") -> pd.DataFrame:
    """Parse an SPF workbook into tidy quarterly forecasts.

    Returns ``[survey_quarter, available_from, nowcast, h1, h2, h3, h4]`` where
    ``nowcast`` is the median current-quarter forecast (source column ``CPI2``)
    and ``h1``..``h4`` are the one- to four-quarter-ahead medians. All values
    are annualised percent.
    """
    try:
        raw = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise SPFUnavailable(f"Could not read SPF workbook: {exc}") from exc

    cols = {str(c).strip().upper(): c for c in raw.columns}
    if "YEAR" not in cols or "QUARTER" not in cols:
        raise SPFUnavailable(f"SPF file missing YEAR/QUARTER; saw {list(raw.columns)[:12]}")

    key = variable.upper()
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(raw[cols["YEAR"]], errors="coerce"),
            "quarter": pd.to_numeric(raw[cols["QUARTER"]], errors="coerce"),
        }
    )

    # CPI1 is the prior quarter, CPI2 the current-quarter nowcast, CPI3..CPI6
    # the one- through four-quarter-ahead forecasts.
    horizon_names = {2: "nowcast", 3: "h1", 4: "h2", 5: "h3", 6: "h4"}
    for idx, name in horizon_names.items():
        src = cols.get(f"{key}{idx}")
        if src is not None:
            out[name] = pd.to_numeric(raw[src], errors="coerce")

    if "nowcast" not in out.columns:
        raise SPFUnavailable(f"No {key}2 column in SPF file; saw {list(raw.columns)[:20]}")

    out = out.dropna(subset=["year", "quarter"])
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    out = out[out["quarter"].between(1, 4)]

    out["survey_quarter"] = pd.PeriodIndex(
        year=out["year"], quarter=out["quarter"], freq="Q"
    ).to_timestamp()

    # Published in the middle month of the quarter; usable from that month's end.
    survey_month = out["quarter"].map(_SURVEY_MONTH)
    out["available_from"] = pd.to_datetime(
        {"year": out["year"], "month": survey_month, "day": 1}
    ) + pd.offsets.MonthEnd(0)

    keep = ["survey_quarter", "available_from", "nowcast"] + [
        c for c in ("h1", "h2", "h3", "h4") if c in out.columns
    ]
    return out[keep].sort_values("survey_quarter").reset_index(drop=True)


def load(variable: str = "cpi", force: bool = False) -> pd.DataFrame:
    """Download (or reuse) and parse SPF medians, caching the tidy frame."""
    tidy = RAW_SPF / f"spf_{variable}.parquet"
    if tidy.exists() and not force:
        return pd.read_parquet(tidy)
    table = parse(download(variable, force=force), variable)
    RAW_SPF.mkdir(parents=True, exist_ok=True)
    table.to_parquet(tidy, index=False)
    return table


def implied_monthly(annualised_pct: pd.Series) -> pd.Series:
    """Convert an annualised quarterly rate to the equivalent monthly percent.

    Presentation helper only — see the module docstring. A forecaster who says
    "3% annualised this quarter" has not said "0.247% in March"; they have said
    something about the quarter's average.
    """
    return (((1.0 + annualised_pct / 100.0) ** (1.0 / 12.0)) - 1.0) * 100.0


def realised_quarterly_annualised(target_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly first-print MoM percents into quarterly annualised rates.

    Compounds the three monthly changes in each quarter and annualises, giving
    a quantity directly comparable to SPF's ``CPI2``. Quarters with fewer than
    three observed months are dropped rather than annualised from partial data.
    """
    df = target_table[["target_month", "target"]].copy()
    df["quarter"] = df["target_month"].dt.to_period("Q")
    df["gross"] = 1.0 + df["target"] / 100.0

    grouped = df.groupby("quarter").agg(gross=("gross", "prod"), n=("gross", "size"))
    grouped = grouped[grouped["n"] == 3]
    out = pd.DataFrame(
        {
            "quarter_start": grouped.index.to_timestamp(),
            "realised_annualised": (grouped["gross"] ** 4 - 1.0) * 100.0,
        }
    )
    return out.reset_index(drop=True)


def quarterly_benchmark(
    target_table: pd.DataFrame, variable: str = "cpi", force: bool = False
) -> pd.DataFrame:
    """Join SPF current-quarter medians to realised quarterly annualised inflation.

    Returns ``[quarter_start, available_from, spf_nowcast, realised_annualised]``.
    """
    spf = load(variable, force=force)
    realised = realised_quarterly_annualised(target_table)
    merged = realised.merge(
        spf.rename(columns={"survey_quarter": "quarter_start", "nowcast": "spf_nowcast"})[
            ["quarter_start", "available_from", "spf_nowcast"]
        ],
        on="quarter_start",
        how="inner",
    )
    return merged.dropna(subset=["spf_nowcast"]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and parse SPF median forecasts.")
    ap.add_argument("--variable", default="cpi", choices=sorted(SPF_FILES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t = load(args.variable, force=args.force)
    first = pd.Period(t["survey_quarter"].min(), freq="Q")
    last = pd.Period(t["survey_quarter"].max(), freq="Q")
    print(f"SPF {args.variable}: {len(t):,} surveys, {first} to {last}")
    print(t.tail(6).to_string(index=False))
