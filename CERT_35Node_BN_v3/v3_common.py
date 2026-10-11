"""
v3_common.py - shared loading code for the v3 scripts.

Everything is resolved relative to this folder, so the project can be copied anywhere.

    HERE/data/user_day_raw.csv.gz      20 daily counts per user-day (from cert_build_table.py)
    HERE/data/attack_days.csv       user, day, scenario - one row per malicious day (from cert_labels.py)
    HERE/data/cache/v3_base.pkl     the table with labels + per-user baselines (built once, then reused)
    HERE/results/                   plots, CPT tables, out-of-fold scores
"""
import os
import pandas as pd
import cert_labels as L
import cert_preprocessing as P

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(DATA, "cache", "v3_base.pkl")


def load_base(force: bool = False) -> pd.DataFrame:
    """raw counts -> + labels (attack days) -> + per-user baselines and z-scores. Cached on disk."""
    if os.path.exists(CACHE) and not force:
        return pd.read_pickle(CACHE)
    raw = pd.read_csv(os.path.join(DATA, "user_day_raw.csv.gz"))
    attacks = L.load_attack_days(events_csv=os.path.join(DATA, "attack_days.csv"))
    df = P.add_user_baselines(L.attach_labels(raw, attacks))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_pickle(CACHE)
    return df

