"""
data_prep.py
============
Loads the LC-HRMS non-targeted analysis (NTA) feature table and the sampling
metadata, merges them on the sample key, and builds every grouping variable
used by the cross-validation schemes.

Design notes
------------
* Peak areas are treated as semi-quantitative intensities. Zeros are genuine
  non-detects, not missing values, and are retained as zeros.
* Three intensity transforms are exposed. Which one is used is decided *inside*
  the inner tuning loop, never by inspecting held-out data.
* Four grouping variables are constructed. The distinction between them is the
  single most important methodological point in this analysis:
    - `event_date`   : the calendar sampling date (10 levels). Each date is one
                       discrete wet-weather sampling occasion with exactly one
                       sample per site. This is the correct unit of replication.
    - `rain_day`     : the "Day of rain" label (Day 1..Day 4, 4 levels). This is
                       the position of the sample *within* a rain episode and is
                       reused across episodes; it is NOT a unique event id.
    - `episode`      : the contiguous rain episode (4 levels), derived from the
                       calendar gaps between sampling dates.
    - `site`         : the classification target (4 levels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IDENT_FILE = "CL_1_2a_2b_identifications.xlsx"
META_FILE = "sample_details_rain_days.xlsx"

# a new rain episode starts when the gap to the previous sampling date exceeds
# this many days
EPISODE_GAP_DAYS = 2


def load_dataset(data_dir="."):
    """Load, merge and return (X_df, y, meta_df, compound_names)."""
    import os

    ident = pd.read_excel(os.path.join(data_dir, IDENT_FILE))
    meta = pd.read_excel(os.path.join(data_dir, META_FILE))
    meta = meta.rename(columns={"Sample Name": "Sample"})

    missing = set(ident["Sample"]) ^ set(meta["Sample"])
    if missing:
        raise ValueError(f"Sample keys do not match between files: {sorted(missing)}")

    df = meta.merge(ident, on="Sample", how="inner", validate="one_to_one")
    df["date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
    df = df.sort_values(["date", "Sampling Point"]).reset_index(drop=True)

    compounds = [c for c in ident.columns if c != "Sample"]
    X = df[compounds].astype(float)

    # ---- grouping variables -------------------------------------------------
    df["site"] = df["Sampling Point"].astype(str)
    df["event_date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["rain_day"] = df["Day of rain"].astype(str)

    uniq = np.sort(df["date"].unique())
    gaps = np.diff(uniq).astype("timedelta64[D]").astype(int)
    ep_of_date, ep = {}, 1
    ep_of_date[uniq[0]] = ep
    for i, g in enumerate(gaps):
        if g > EPISODE_GAP_DAYS:
            ep += 1
        ep_of_date[uniq[i + 1]] = ep
    df["episode"] = df["date"].map(ep_of_date).map(lambda e: f"Episode {e}")

    meta_cols = [
        "Sample", "site", "date", "event_date", "rain_day", "episode",
        "Sampling Point", "Day of rain",
    ]
    return X, df["site"].values, df[meta_cols].copy(), compounds


# --------------------------------------------------------------------------- #
# Intensity transforms
# --------------------------------------------------------------------------- #
def transform_intensities(X, method):
    """
    Apply an intensity transform to a raw peak-area matrix.

    log      : log10(area + 1). Retains absolute intensity information.
    tic_log  : each sample is first scaled to a constant total signal
               (1e6 units) and then log10(x+1) transformed. Removes
               sample-to-sample differences in total injected/ionised mass
               (injection volume, matrix effects, dilution by rain volume) and
               makes the fingerprint compositional.
    presence : binary detect / non-detect matrix. Discards all quantitative
               information and tests whether the *pattern of occurrence* alone
               carries site information.

    Every transform is applied row-wise (per sample) only, so it can be
    computed identically for training and held-out samples without any
    information passing between them.
    """
    X = np.asarray(X, dtype=float)
    if method == "log":
        return np.log10(X + 1.0)
    if method == "tic_log":
        tot = X.sum(axis=1, keepdims=True)
        tot[tot == 0] = 1.0
        return np.log10(X / tot * 1e6 + 1.0)
    if method == "presence":
        return (X > 0).astype(float)
    raise ValueError(f"unknown transform: {method}")


TRANSFORMS = ["log", "tic_log", "presence"]


def detection_summary(X, compounds, meta):
    """Detection frequency overall and per site — used for the chemical inventory."""
    Xv = np.asarray(X, dtype=float)
    det = Xv > 0
    out = pd.DataFrame({
        "compound": compounds,
        "n_detects": det.sum(axis=0),
        "detection_frequency": det.mean(axis=0),
        "median_area_when_detected": [
            np.median(Xv[det[:, j], j]) if det[:, j].any() else 0.0
            for j in range(Xv.shape[1])
        ],
    })
    for s in sorted(pd.unique(meta["site"])):
        m = (meta["site"] == s).values
        out[f"DF_{s}"] = det[m].mean(axis=0)
    return out.sort_values("detection_frequency", ascending=False).reset_index(drop=True)
