"""Validated CSV inputs for reusable stage/discharge rating-curve workflows.

This module preserves gauged stages, reports excluded input records, and computes
the stage-change covariate exclusively from consecutive daily observations.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


_ALIASES = {
    "date": {"date", "datetime", "timestamp", "time", "measurementdate", "observationdate", "gaugingdate"},
    "stage": {"wl", "waterlevel", "stage", "h", "level", "gageheight", "gaugeheight", "wlevel"},
    "discharge": {"q", "discharge", "flow", "flowrate", "streamflow", "observeddischarge", "qobs"},
    "DL": {"dl", "dangerlevel", "dangerstage", "floodlevel"},
}


def _normalized_header(value: str) -> str:
    # Strip common unit suffixes only: this must not mistake modelled Q columns
    # or metadata descriptions for observed discharge.
    value = str(value).strip().lower().replace("³", "3")
    value = re.sub(r"\s*[\[(]\s*(?:m|meters?|metres?|m3\s*/\s*s|m\^3\s*/\s*s|cms|cumecs?)\s*[\])]\s*$", "", value)
    value = re.sub(r"(?:[_\s]+)(?:m|meters?|metres?|m3s|m3_s|m3/s|cms|cumecs?)$", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def _select_column(frame: pd.DataFrame, role: str, override: str | None, *, required: bool, label: str) -> str | None:
    names = list(frame.columns)
    if override is not None:
        if override in names:
            return override
        matches = [name for name in names if name.casefold() == override.casefold()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"{label}: column override {override!r} is missing or ambiguous. Available columns: {names}")
    matches = [name for name in names if _normalized_header(name) in _ALIASES[role]]
    if len(matches) > 1:
        if role == 'DL':
            raise ValueError(f"{label}: ambiguous optional DL columns {matches}; retain or rename columns so only the intended danger-level column is recognized.")
        raise ValueError(f"{label}: ambiguous {role} columns {matches}; provide an explicit {role} column override.")
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"{label}: cannot identify the {role} column. Provide an explicit column override. Available columns: {names}")
    return None


def _read_csv(path: str | Path, label: str, notes: list[str]) -> pd.DataFrame:
    path = Path(path)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = path.read_text(encoding="cp1252")
        notes.append(f"{label}: decoded CSV as Windows-1252 after UTF-8 decoding failed.")
    if not content.strip():
        raise ValueError(f"{label}: CSV is empty: {path}")
    try:
        dialect = csv.Sniffer().sniff(content[:65536], delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        first_line = next(line for line in content.splitlines() if line.strip())
        delimiter = max((",", ";", "\t", "|"), key=first_line.count)
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    headers = None
    rows: list[list[str]] = []
    source_rows: list[int] = []
    previous_end = 0
    for row in reader:
        source_line = previous_end + 1
        previous_end = reader.line_num
        if not row or all(not value.strip() for value in row):
            continue
        if headers is None:
            headers = [value.strip() or f"Unnamed: {i}" for i, value in enumerate(row)]
            duplicates = sorted({name for name in headers if headers.count(name) > 1})
            if duplicates:
                raise ValueError(f"{label}: duplicate column headers {duplicates}; rename them in the CSV before loading.")
            if "source_row" in headers or "rejection_reason" in headers:
                raise ValueError(f"{label}: source_row and rejection_reason are reserved report column names; rename those input columns.")
            continue
        if len(row) > len(headers):
            raise ValueError(f"{label}: row {source_line} has {len(row)} fields but header has {len(headers)}. Check the delimiter and CSV quoting.")
        rows.append(row + [""] * (len(headers) - len(row)))
        source_rows.append(source_line)
    if headers is None:
        raise ValueError(f"{label}: no CSV header found: {path}")
    result = pd.DataFrame(rows, columns=headers)
    result["source_row"] = source_rows
    return result


def _dates(values: pd.Series, *, date_format: str | None, dayfirst: bool, label: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(values.str.strip().replace("", np.nan), format=date_format or "mixed", dayfirst=dayfirst, errors="coerce")
        # Daily covariates and duplicate detection operate on local calendar days.
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_localize(None)
        return parsed.dt.normalize()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label}: dates could not be interpreted consistently. Use one timezone and provide date_format for ambiguous formats.") from exc


def _numbers(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.str.strip().replace("", np.nan), errors="coerce").astype(float)


def _reject_report(raw: pd.DataFrame, reasons: list[list[str]]) -> pd.DataFrame:
    rejected = raw.loc[[bool(reason) for reason in reasons]].copy()
    rejected["rejection_reason"] = ["; ".join(reason) for reason in reasons if reason]
    return rejected.reset_index(drop=True)


def load_inputs(
    q_path: str | Path,
    daily_path: str | Path | None = None,
    *,
    date_col: str | None = None,
    stage_col: str | None = None,
    discharge_col: str | None = None,
    daily_date_col: str | None = None,
    daily_stage_col: str | None = None,
    date_format: str | None = None,
    dayfirst: bool = False,
) -> dict[str, Any]:
    """Read paired gaugings and optional daily stages without silent row removal.

    Header aliases are case/punctuation insensitive; multiple matches require an
    explicit override. Date parsing defaults to month-first for ambiguous dates;
    specify ``date_format`` (recommended) or ``dayfirst=True`` when appropriate.
    Observations without a date column remain eligible for the static curve.

    Returns canonical ``observations`` (Date, WL, Q, source_row, optional DL,
    dWL, daily_WL), ``daily`` (Date, WL, source_row, dWL) or None, two rejection
    reports containing original records and reasons, ``notes``, and ``columns``.
    A finite, nonnegative Q and finite WL are required. A supplied date column
    must parse. ``dWL`` is NaN for absent daily data, first dates, or day gaps.
    """
    notes: list[str] = []
    raw = _read_csv(q_path, "observations", notes)
    cols = {
        "date": _select_column(raw, "date", date_col, required=False, label="observations"),
        "stage": _select_column(raw, "stage", stage_col, required=True, label="observations"),
        "discharge": _select_column(raw, "discharge", discharge_col, required=True, label="observations"),
        "DL": _select_column(raw, "DL", None, required=False, label="observations"),
    }
    used = [name for name in cols.values() if name is not None]
    if len(set(used)) != len(used):
        raise ValueError("observations: date, stage, discharge, and DL must refer to distinct input columns.")
    obs = pd.DataFrame({"Date": pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]"), "WL": _numbers(raw[cols["stage"]]), "Q": _numbers(raw[cols["discharge"]]), "source_row": raw["source_row"]})
    if cols["date"] is not None:
        obs["Date"] = _dates(raw[cols["date"]], date_format=date_format, dayfirst=dayfirst, label="observations")
    else:
        notes.append("observations: no date column found; time-based validation and daily stage-change matching are unavailable.")
    reasons: list[list[str]] = [[] for _ in range(len(obs))]
    for i in range(len(obs)):
        if not np.isfinite(obs.at[i, "WL"]):
            reasons[i].append("WL must be numeric and finite")
        if not np.isfinite(obs.at[i, "Q"]) or obs.at[i, "Q"] < 0:
            reasons[i].append("Q must be numeric, finite, and nonnegative")
        if cols["date"] is not None and pd.isna(obs.at[i, "Date"]):
            reasons[i].append("Date is missing or invalid")
    rejected_obs = _reject_report(raw, reasons)
    if cols["DL"] is not None:
        obs["DL"] = _numbers(raw[cols["DL"]])
        invalid_dl = ~np.isfinite(obs["DL"])
        if invalid_dl.any():
            notes.append(f"observations: {int(invalid_dl.sum())} missing/nonfinite optional DL values retained as NaN.")
        obs.loc[invalid_dl, "DL"] = np.nan
    obs = obs.loc[[not reason for reason in reasons]].copy()
    duplicated = obs.duplicated(subset=["Date", "WL", "Q"], keep="first")
    if duplicated.any():
        notes.append(f"observations: collapsed {int(duplicated.sum())} exact duplicate Date/WL/Q measurements; retained first source rows.")
    obs = obs.loc[~duplicated].sort_values("Date", kind="stable", na_position="last").reset_index(drop=True)
    if len(rejected_obs):
        notes.append(f"observations: rejected {len(rejected_obs)} input records; see rejected_observations report.")

    daily = None
    daily_cols = None
    rejected_daily = pd.DataFrame(columns=["source_row", "rejection_reason"])
    if daily_path is not None:
        daily_raw = _read_csv(daily_path, "daily", notes)
        daily_cols = {
            "date": _select_column(daily_raw, "date", daily_date_col, required=True, label="daily"),
            "stage": _select_column(daily_raw, "stage", daily_stage_col, required=True, label="daily"),
        }
        if daily_cols["date"] == daily_cols["stage"]:
            raise ValueError("daily: date and stage must refer to distinct input columns.")
        daily = pd.DataFrame({"Date": _dates(daily_raw[daily_cols["date"]], date_format=date_format, dayfirst=dayfirst, label="daily"), "WL": _numbers(daily_raw[daily_cols["stage"]]), "source_row": daily_raw["source_row"]})
        daily_reasons: list[list[str]] = [[] for _ in range(len(daily))]
        for i in range(len(daily)):
            if pd.isna(daily.at[i, "Date"]):
                daily_reasons[i].append("Date is missing or invalid")
            if not np.isfinite(daily.at[i, "WL"]):
                daily_reasons[i].append("WL must be numeric and finite")
        rejected_daily = _reject_report(daily_raw, daily_reasons)
        daily = daily.loc[[not reason for reason in daily_reasons]].copy()
        conflicts = daily.groupby("Date")["WL"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            examples = ", ".join(date.strftime("%Y-%m-%d") for date in conflicts.index[:5])
            raise ValueError(f"daily: conflicting duplicate water levels for {len(conflicts)} dates ({examples}). Resolve or explicitly aggregate these dates before fitting; no automatic averaging is performed.")
        duplicates = daily.duplicated("Date", keep="first")
        if duplicates.any():
            notes.append(f"daily: collapsed {int(duplicates.sum())} duplicate dates with identical WL; retained first source rows.")
        daily = daily.loc[~duplicates].sort_values("Date", kind="stable").reset_index(drop=True)
        consecutive = daily["Date"].diff().eq(pd.Timedelta(days=1))
        daily["dWL"] = daily["WL"].diff().where(consecutive)
        gap_count = max(int((~consecutive).sum()) - (1 if len(daily) else 0), 0)
        if gap_count:
            notes.append(f"daily: {gap_count} calendar-day gaps; dWL after each gap remains NaN.")
        if len(rejected_daily):
            notes.append(f"daily: rejected {len(rejected_daily)} input records; see rejected_daily report.")
        obs = obs.merge(daily[["Date", "WL", "dWL"]].rename(columns={"WL": "daily_WL"}), on="Date", how="left", validate="many_to_one", sort=False)
        mismatch = obs["daily_WL"].notna() & ~np.isclose(obs["WL"], obs["daily_WL"], rtol=0.0, atol=0.001)
        if mismatch.any():
            notes.append(f"observations: {int(mismatch.sum())} gauged WL values differ from same-day daily WL by more than 0.001 stage units; retained gauged WL unchanged.")
        unmatched = obs["daily_WL"].isna().sum()
        if unmatched:
            notes.append(f"observations: {int(unmatched)} records have no valid same-day daily WL match; daily_WL and dWL remain NaN.")
    else:
        obs["daily_WL"] = np.nan
        obs["dWL"] = np.nan
        notes.append("No daily CSV supplied; dWL is unavailable and remains NaN.")
    if (cols["date"] is not None or daily_path is not None) and date_format is None:
        notes.append(f"Dates parsed with dayfirst={dayfirst}; specify date_format for an explicit calendar convention.")
    return {"observations": obs, "daily": daily, "rejected_observations": rejected_obs, "rejected_daily": rejected_daily, "notes": notes, "columns": {"observations": cols, "daily": daily_cols}}
