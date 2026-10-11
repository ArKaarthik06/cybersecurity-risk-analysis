"""
cert_labels.py
==============
Ground-truth labelling for CERT r4.2 at *attack-day* granularity.

Why this module exists
----------------------
The original pipeline labelled **every day of every insider user** as malicious
(15,444 "positive" rows).  Only 966 user-days actually contain red-team events
(0.29 % of the table).  ~94 % of the original positives were ordinary days that
are *by construction* indistinguishable from normal behaviour, which caps every
classifier (BN or otherwise) at ROC-AUC ~0.6-0.7.

Ground truth used here
----------------------
answers/insiders.csv                 -> which users are insiders + scenario
answers/r4.2-<s>/r4.2-<s>-<USER>.csv -> the exact malicious events of each insider

A user-day is *malicious* iff the insider (the user named in insiders.csv)
produced at least one event listed in that incident's observables file on that day.

Scenario -> threat hypotheses (multi-label; the 4 threat nodes are not exclusive)
-------------------------------------------------------------------------------
  1  after-hours logon + thumb drive + wikileaks upload   -> DataExfiltration
  2  job-hunting / competitor sites + heavy thumb drive   -> IPTheft
  3  sysadmin keylogger, masquerade as supervisor, panic
     mass-email                                           -> UnauthorizedAccess + ITSabotage
  4,5 do NOT occur in r4.2 (only 1,2,3 exist), so they are not mapped.

NOTE: ITSabotage therefore has supervision from scenario 3 only (10 insiders,
20 attack days).  That is a documented limitation, not something to hide.
"""
import io
import csv
import zipfile
from datetime import datetime

import pandas as pd

SCENARIO_THREATS = {
    1: ("DataExfiltration",),
    2: ("IPTheft",),
    3: ("UnauthorizedAccess", "ITSabotage"),
}
THREAT_NODES = ["DataExfiltration", "UnauthorizedAccess", "ITSabotage", "IPTheft"]


def load_attack_days(zip_path: str = None, events_csv: str = None) -> pd.DataFrame:
    """
    Return a DataFrame [user, day, scenario] with one row per malicious user-day.

    Either read the answer key straight from archive.zip (small files only) or
    from a cached `answers_malicious_events.csv` (columns user,date,scenario,...).
    """
    if zip_path is not None:
        rows = set()
        with zipfile.ZipFile(zip_path) as z:
            ins = list(csv.DictReader(io.StringIO(z.read("answers/insiders.csv").decode())))
            for r in ins:
                if r["dataset"] != "4.2":
                    continue
                insider, scen, fn = r["user"], int(r["scenario"]), r["details"]
                folder = fn.rsplit("-", 1)[0]
                for line in z.read(f"answers/{folder}/{fn}").decode().splitlines():
                    p = line.split(",")
                    if len(p) < 5 or p[3] != insider:   # only the insider's own events
                        continue
                    d = datetime.strptime(p[2], "%m/%d/%Y %H:%M:%S").date().isoformat()
                    rows.add((insider, d, scen))
        return pd.DataFrame(sorted(rows), columns=["user", "day", "scenario"])

    return pd.read_csv(events_csv)[["user", "day", "scenario"]]


def load_insider_windows(zip_path: str) -> pd.DataFrame:
    """[user, scenario, start_day, end_day] from insiders.csv (r4.2 only)."""
    with zipfile.ZipFile(zip_path) as z:
        df = pd.read_csv(io.BytesIO(z.read("answers/insiders.csv")))
    df = df[df["dataset"] == 4.2].copy()
    df["start_day"] = pd.to_datetime(df["start"], format="mixed").dt.date.astype(str)
    df["end_day"] = pd.to_datetime(df["end"], format="mixed").dt.date.astype(str)
    return df[["user", "scenario", "start_day", "end_day"]].reset_index(drop=True)


def attach_labels(user_day: pd.DataFrame, attack_days: pd.DataFrame,
                  windows: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add columns to a user-day table:
      y            : 1 if malicious attack-day
      scenario     : scenario of the *insider user* (0 for normal users)
      is_insider   : user is an insider (any day)  -- used ONLY for stratified splitting
      one 0/1 column per threat node (multi-label supervision)
      risk_label   : 2 = attack day, 1 = inside the insider's campaign window
                     without a logged attack event, 0 = otherwise
    """
    df = user_day.copy()
    # Strip any legacy label columns (whole-user labelling) coming from older caches
    df = df.drop(columns=[c for c in ("is_insider", "threat_type", "scenario", "y") if c in df.columns])
    ad = attack_days.copy()
    ad["y"] = 1
    df = df.merge(ad[["user", "day", "y"]], on=["user", "day"], how="left")
    df["y"] = df["y"].fillna(0).astype(int)

    user_scen = attack_days.groupby("user")["scenario"].first()
    df["scenario"] = df["user"].map(user_scen).fillna(0).astype(int)
    df["is_insider"] = (df["scenario"] > 0).astype(int)

    for t in THREAT_NODES:
        df[t] = 0
    for s, threats in SCENARIO_THREATS.items():
        m = (df["y"] == 1) & (df["scenario"] == s)
        for t in threats:
            df.loc[m, t] = 1

    df["risk_label"] = 0
    if windows is not None:
        w = windows.rename(columns={"scenario": "w_scen"})
        tmp = df[["user", "day"]].merge(w, on="user", how="left")
        inwin = ((tmp["day"] >= tmp["start_day"]) & (tmp["day"] <= tmp["end_day"])).fillna(False)
        df["risk_label"] = inwin.astype(int).values
    df.loc[df["y"] == 1, "risk_label"] = 2
    return df


if __name__ == "__main__":
    # python cert_labels.py path/to/archive.zip
    # Writes data/attack_days.csv and data/insider_windows.csv from the answers/ folder of the zip.
    import os, sys
    if len(sys.argv) < 2:
        sys.exit("usage: python cert_labels.py path/to/archive.zip")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    ad = load_attack_days(zip_path=sys.argv[1])
    ad.to_csv(os.path.join(out, "attack_days.csv"), index=False)
    load_insider_windows(sys.argv[1]).to_csv(os.path.join(out, "insider_windows.csv"), index=False)
    print(f"attack days: {len(ad)}, insiders: {ad.user.nunique()} -> {out}")
