from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def download_yahoo_ohlcv(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance ticker by ticker."""
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        frame = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            interval="1d",
            threads=False,
        )
        if frame.empty:
            continue

        frame = frame.reset_index()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                (column[0] if column[0] else column[1]).lower().replace(" ", "_") for column in frame.columns
            ]
        else:
            frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        frame["ticker"] = ticker
        frames.append(frame[["date", "ticker", *PRICE_COLUMNS]])

    if not frames:
        raise RuntimeError("Yahoo Finance download returned no data.")

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df = raw_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return raw_df


def clean_and_align_prices(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align common trading dates and fill any residual missing values ticker by ticker."""
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    date_sets = [set(group["date"]) for _, group in df.groupby("ticker")]
    common_dates = sorted(set.intersection(*date_sets))
    aligned = df[df["date"].isin(common_dates)].copy()
    aligned = aligned.sort_values(["ticker", "date"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for ticker, group in aligned.groupby("ticker", sort=True):
        missing_ratio = group[PRICE_COLUMNS].isna().mean().mean()
        summary_rows.append(
            {
                "ticker": ticker,
                "start_date": group["date"].min().date().isoformat(),
                "end_date": group["date"].max().date().isoformat(),
                "num_rows": int(len(group)),
                "missing_ratio": float(missing_ratio),
            }
        )

    aligned[PRICE_COLUMNS] = (
        aligned.groupby("ticker", group_keys=False)[PRICE_COLUMNS].apply(lambda group: group.ffill().bfill())
    )

    summary_df = pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True)
    return aligned, summary_df
