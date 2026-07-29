"""Caldara & Iacoviello Geopolitical Risk (GPR) index.

Source: https://www.matteoiacoviello.com/gpr.htm (mirrored at the Federal
Reserve Board). The monthly file carries the benchmark index plus its
decomposition into *threats* and *acts*, which the brief requires be kept
separable rather than blended into one opaque risk feature.

Real-time properties
--------------------
GPR is built by counting geopolitical-risk phrases in newspaper archives for a
given month. It is not revised: the newspaper text for March 2003 is the same
text today as it was in April 2003, so there is no vintage problem of the kind
CPI has. What *does* need care is publication timing — the month-``m`` value is
only mechanically computable once month ``m`` has ended.

We therefore treat the month-``m`` value as knowable from
``month_end + PUBLICATION_LAG_DAYS``. The default of 5 days is a judgement
call: the underlying news archives are available same-day (so a practitioner
running the authors' code could have it almost immediately), but Iacoviello's
posted file updates on a slower, irregular cadence. Five days is the
conservative-but-not-absurd middle. It is a single constant here so the
assumption can be found, argued with, and changed in one place.
"""

from __future__ import annotations

import argparse
import io

import pandas as pd
import requests

from src.config import RAW

# Candidate locations for the monthly export, tried in order. The author has
# moved this file before; the fallbacks save a manual scramble when he does.
GPR_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xlsx",
    "https://www.matteoiacoviello.com/gpr_files/gpr_web_latest.xlsx",
]

RAW_GPR = RAW / "gpr"
PUBLICATION_LAG_DAYS = 5

# Source column -> our name. The file also carries ~40 country columns and the
# historical (GPRH*) variants; we take the benchmark index and its two
# components and leave the rest.
_COLUMN_MAP = {
    "gpr": "gpr",
    "gprt": "gpr_threats",
    "gpra": "gpr_acts",
    "gprh": "gpr_hist",
    "gprht": "gpr_hist_threats",
    "gprha": "gpr_hist_acts",
}


class GPRUnavailable(RuntimeError):
    """The GPR file could not be downloaded or parsed."""


def download(force: bool = False) -> bytes:
    """Fetch the raw monthly GPR workbook, caching it under data/raw/gpr/."""
    RAW_GPR.mkdir(parents=True, exist_ok=True)
    cached = sorted(RAW_GPR.glob("data_gpr_export.*"))
    if cached and not force:
        return cached[0].read_bytes()

    errors: list[str] = []
    for url in GPR_URLS:
        try:
            resp = requests.get(url, timeout=90, headers={"User-Agent": "inflation-nowcaster/1.0"})
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue
        if resp.status_code == 200 and resp.content:
            suffix = ".xlsx" if url.endswith("xlsx") else ".xls"
            (RAW_GPR / f"data_gpr_export{suffix}").write_bytes(resp.content)
            return resp.content
        errors.append(f"{url}: HTTP {resp.status_code}")

    raise GPRUnavailable(
        "Could not download the GPR index. Tried:\n  " + "\n  ".join(errors) +
        f"\nDownload the monthly file manually from https://www.matteoiacoviello.com/gpr.htm "
        f"and save it to {RAW_GPR}\\data_gpr_export.xlsx"
    )


def _read_workbook(content: bytes) -> pd.DataFrame:
    """Read the workbook regardless of whether it is real .xls, .xlsx, or CSV.

    The published '.xls' has at times actually been an .xlsx or even a CSV with
    the wrong extension, so sniff rather than trust.
    """
    attempts: list[str] = []
    for engine in ("openpyxl", "xlrd", None):
        try:
            return pd.read_excel(io.BytesIO(content), engine=engine)
        except Exception as exc:
            attempts.append(f"{engine or 'default'}: {type(exc).__name__}")
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        attempts.append(f"csv: {type(exc).__name__}")
    raise GPRUnavailable("Could not parse the GPR workbook (" + "; ".join(attempts) + ")")


def _find_date(df: pd.DataFrame) -> pd.Series:
    """Locate and normalise the date column.

    The file has used 'month', 'DATE', and a YYYYMM integer across versions.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    for key in ("month", "date", "yearmonth", "obs"):
        if key in lower:
            col = df[lower[key]]
            # YYYYMM integers (e.g. 200303) — parse explicitly, since to_datetime
            # would otherwise read them as nanosecond epochs.
            if pd.api.types.is_numeric_dtype(col) and col.dropna().between(180001, 299912).all():
                return pd.to_datetime(col.astype("Int64").astype(str), format="%Y%m")
            return pd.to_datetime(col, errors="coerce")
    raise GPRUnavailable(f"No recognisable date column in GPR file; saw {list(df.columns)[:12]}")


def parse(content: bytes | None = None) -> pd.DataFrame:
    """Parse the GPR workbook into a tidy monthly frame.

    Returns columns ``[month, available_from, gpr, gpr_threats, gpr_acts, ...]``
    where ``available_from`` is the first date the value may be used.
    """
    if content is None:
        content = download()
    raw = _read_workbook(content)

    dates = _find_date(raw)
    out = pd.DataFrame({"month": dates.dt.to_period("M").dt.to_timestamp()})

    lower = {str(c).strip().lower(): c for c in raw.columns}
    for src, dest in _COLUMN_MAP.items():
        if src in lower:
            out[dest] = pd.to_numeric(raw[lower[src]], errors="coerce")

    if "gpr" not in out.columns:
        raise GPRUnavailable(f"No GPR column found; saw {list(raw.columns)[:20]}")

    out = out.dropna(subset=["month"]).drop_duplicates(subset="month").sort_values("month")

    # The month-m value is computable once month m has ended, plus the lag.
    month_end = out["month"] + pd.offsets.MonthEnd(0)
    out["available_from"] = month_end + pd.Timedelta(days=PUBLICATION_LAG_DAYS)

    return out.reset_index(drop=True)


class GPRStore:
    """Point-in-time access to GPR, mirroring :class:`VintageStore.as_of`."""

    def __init__(self, table: pd.DataFrame | None = None):
        self.table = parse() if table is None else table

    def as_of(self, as_of_date, column: str = "gpr") -> pd.Series:
        """GPR history knowable on ``as_of_date``, indexed by month."""
        as_of_date = pd.Timestamp(as_of_date)
        if column not in self.table.columns:
            return pd.Series(dtype="float64", name=column)
        live = self.table[self.table["available_from"] <= as_of_date]
        return live.set_index("month")[column].dropna().rename(column)

    @property
    def columns(self) -> list[str]:
        return [c for c in self.table.columns if c not in ("month", "available_from")]


def load(force: bool = False) -> pd.DataFrame:
    """Download (or reuse) and parse GPR, caching the tidy frame."""
    tidy_path = RAW_GPR / "gpr_monthly.parquet"
    if tidy_path.exists() and not force:
        return pd.read_parquet(tidy_path)
    table = parse(download(force=force))
    RAW_GPR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(tidy_path, index=False)
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and parse the GPR index.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t = load(force=args.force)
    print(f"GPR: {len(t):,} months, {t['month'].min():%Y-%m} to {t['month'].max():%Y-%m}")
    print(f"Columns: {[c for c in t.columns if c != 'month']}")
    print(t.tail(6).to_string(index=False))
