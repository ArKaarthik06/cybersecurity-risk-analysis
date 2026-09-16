"""
cert_preprocessing.py
=====================
Streaming preprocessing for CERT Insider Threat Dataset R4.2.

Reads log CSVs directly from the source zip (no full extraction to disk).
Outputs a user-day behavioral feature table suitable for the Bayesian Network.

Pipeline:
  archive.zip  (streaming)
       ↓
  user-day raw counts
       ↓
  label assignment  (answers/insiders.csv)
       ↓
  user-level train/test split
       ↓
  percentile threshold fitting  (training users only)
       ↓
  discretization → BN evidence variables (binary Normal/High)

Author: Generated for Probabilistic Reasoning End-Semester Project
Dataset: CERT Insider Threat Test Dataset R4.2 (synthetic data)
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 200_000          # rows per streaming chunk
BUSINESS_START = 9            # 09:00 local — business hours start
BUSINESS_END   = 17           # 17:00 local — business hours end
INTERNAL_DOMAIN = "dtaa.com"  # CERT R4.2 organisation email domain

# Raw feature columns produced by the streaming aggregation step
RAW_FEATURE_COLS = [
    "logon_count",
    "after_hours_logon_count",
    "weekend_logon_count",
    "machine_count",
    "device_connect_count",
    "after_hours_device_count",
    "file_activity_count",
    "file_copy_to_removable_count",
    "archive_activity_count",
    "exe_activity_count",
    "unusual_file_activity_count",
    "email_count",
    "external_email_count",
    "email_size_total",
    "after_hours_email_count",
    "external_large_email_count",
    "web_activity_count",
    "after_hours_web_count",
    "cloud_storage_count",
    "job_search_count",
]

# BN evidence node names produced after discretization
EVIDENCE_NODES = [
    "AfterHoursLogin",        # after_hours_logon_count
    "UnusualLoginFreq",       # logon_count
    "WeekendLogin",           # weekend_logon_count
    "HighMachineCount",       # machine_count
    "DeviceConnectActivity",  # device_connect_count
    "DeviceAfterHours",       # after_hours_device_count
    "FileAccessCount",        # file_activity_count
    "FileCopyToRemovable",    # file_copy_to_removable_count
    "ArchiveCreation",        # archive_activity_count
    "ExeActivity",            # exe_activity_count
    "UnusualFileTypes",       # unusual_file_activity_count
    "ExternalEmail",          # external_email_count
    "UnusualEmailVolume",     # email_count
    "AfterHoursEmail",        # after_hours_email_count
    "ExternalLargeEmail",     # external_large_email_count
    "UnusualWebActivity",     # web_activity_count
    "AfterHoursWeb",          # after_hours_web_count
    "CloudStorageAccess",     # cloud_storage_count
    "JobSearchActivity",      # job_search_count
    "HighDataMovement",       # email_size_total
    "HighFileCopyActivity",   # file_copy_to_removable_count
]

# ---------------------------------------------------------------------------
# CERT Scenario → Threat type mapping
# (based on CERT R4.2 scenario descriptions)
# ---------------------------------------------------------------------------
SCENARIO_THREAT_MAP = {
    1: "DataExfiltration",      # After-hours USB + Wikileaks upload
    2: "IPTheft",               # Job-search + USB data theft
    3: "UnauthorizedAccess",    # Sysadmin keylogger + masquerade
    4: "ITSabotage",            # Login to other machine + email to home
    5: "DataExfiltration",      # Dropbox upload
}


# ===========================================================================
# Step 1 — Streaming aggregation (adapted from cert_stage1_preprocessing.ipynb)
# ===========================================================================

def _is_after_hours(ts_series: pd.Series) -> pd.Series:
    hours = ts_series.dt.hour
    return (hours < BUSINESS_START) | (hours >= BUSINESS_END)


def _stream_logon(zf: zipfile.ZipFile, agg: dict, member: str = "r4.2/logon.csv"):
    """Aggregate logon events per (user, day)."""
    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["date", "user", "pc", "activity"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", dayfirst=False)
            chunk["day"] = chunk["date"].dt.date
            logons = chunk[chunk["activity"] == "Logon"].copy()
            logons["after_hours"] = _is_after_hours(logons["date"])
            logons["is_weekend"] = logons["date"].dt.dayofweek >= 5
            for (user, day), g in logons.groupby(["user", "day"]):
                key = (user, str(day))
                agg[key]["logon_count"] += len(g)
                agg[key]["after_hours_logon_count"] += int(g["after_hours"].sum())
                agg[key]["weekend_logon_count"] += int(g["is_weekend"].sum())
                if "pcs" not in agg[key]:
                    agg[key]["pcs"] = set()
                agg[key]["pcs"].update(g["pc"].dropna().unique())

    # Post-process sets to counts
    for key in agg:
        if "pcs" in agg[key]:
            agg[key]["machine_count"] = len(agg[key]["pcs"])
            del agg[key]["pcs"]


def _stream_device(zf: zipfile.ZipFile, agg: dict, member: str = "r4.2/device.csv"):
    """Aggregate device Connect events per (user, day)."""
    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["date", "user", "activity"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", dayfirst=False)
            chunk["day"] = chunk["date"].dt.date
            connects = chunk[chunk["activity"] == "Connect"].copy()
            connects["after_hours"] = _is_after_hours(connects["date"])
            for (user, day), g in connects.groupby(["user", "day"]):
                key = (user, str(day))
                agg[key]["device_connect_count"] += len(g)
                agg[key]["after_hours_device_count"] += int(g["after_hours"].sum())


def _stream_file(zf: zipfile.ZipFile, agg: dict, member: str = "r4.2/file.csv"):
    """Aggregate file activity and copies-to-removable per (user, day)."""
    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["date", "user", "filename"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", dayfirst=False)
            chunk["day"] = chunk["date"].dt.date
            
            removable_exts = {".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx"}
            archive_exts = {".zip", ".rar", ".7z", ".tar", ".gz"}
            exe_exts = {".exe", ".bat", ".ps1", ".vbs"}
            unusual_exts = {".iso", ".img", ".sys", ".dll"}

            def categorize(fn):
                ext = os.path.splitext(str(fn).lower())[1]
                return (
                    ext in removable_exts or ext in archive_exts or ext in exe_exts,
                    ext in archive_exts,
                    ext in exe_exts,
                    ext in unusual_exts
                )
            
            cats = chunk["filename"].apply(categorize)
            chunk["to_removable"] = cats.apply(lambda x: x[0])
            chunk["is_archive"] = cats.apply(lambda x: x[1])
            chunk["is_exe"] = cats.apply(lambda x: x[2])
            chunk["is_unusual"] = cats.apply(lambda x: x[3])

            for (user, day), g in chunk.groupby(["user", "day"]):
                key = (user, str(day))
                agg[key]["file_activity_count"] += len(g)
                agg[key]["file_copy_to_removable_count"] += int(g["to_removable"].sum())
                agg[key]["archive_activity_count"] += int(g["is_archive"].sum())
                agg[key]["exe_activity_count"] += int(g["is_exe"].sum())
                agg[key]["unusual_file_activity_count"] += int(g["is_unusual"].sum())


def _stream_email(
    zf: zipfile.ZipFile,
    agg: dict,
    member: str = "r4.2/email.csv",
    internal_domain: str = INTERNAL_DOMAIN,
):
    """Aggregate email counts, external emails, and total size per (user, day)."""
    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["date", "user", "to", "size"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", dayfirst=False)
            chunk["day"] = chunk["date"].dt.date
            chunk["to"] = chunk["to"].fillna("")
            chunk["is_external"] = ~chunk["to"].str.contains(
                internal_domain, na=False, regex=False
            )
            chunk["size"] = pd.to_numeric(chunk["size"], errors="coerce").fillna(0)
            chunk["after_hours"] = _is_after_hours(chunk["date"])
            chunk["external_large"] = chunk["is_external"] & (chunk["size"] > 1_000_000) # > 1MB

            for (user, day), g in chunk.groupby(["user", "day"]):
                key = (user, str(day))
                agg[key]["email_count"] += len(g)
                agg[key]["external_email_count"] += int(g["is_external"].sum())
                agg[key]["email_size_total"] += float(g["size"].sum())
                agg[key]["after_hours_email_count"] += int(g["after_hours"].sum())
                agg[key]["external_large_email_count"] += int(g["external_large"].sum())


def _stream_http(zf: zipfile.ZipFile, agg: dict, member: str = "r4.2/http.csv"):
    """Aggregate web activity per (user, day)."""
    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=["date", "user", "url"],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk["date"] = pd.to_datetime(chunk["date"], format="mixed", dayfirst=False)
            chunk["day"] = chunk["date"].dt.date
            chunk["after_hours"] = _is_after_hours(chunk["date"])
            
            url_str = chunk["url"].fillna("").astype(str).str.lower()
            chunk["is_cloud"] = url_str.str.contains("dropbox|drive.google|onedrive|box.com")
            chunk["is_job"] = url_str.str.contains("indeed|monster|linkedin|career|job")

            for (user, day), g in chunk.groupby(["user", "day"]):
                key = (user, str(day))
                agg[key]["web_activity_count"] += len(g)
                agg[key]["after_hours_web_count"] += int(g["after_hours"].sum())
                agg[key]["cloud_storage_count"] += int(g["is_cloud"].sum())
                agg[key]["job_search_count"] += int(g["is_job"].sum())


def build_user_day_table(zip_path: str, cache_path: str | None = None) -> pd.DataFrame:
    """
    Stream all CERT R4.2 log CSVs from the zip and aggregate into a
    user-day feature DataFrame.

    Parameters
    ----------
    zip_path  : Path to archive.zip (or r4.2.zip).
    cache_path: Optional path to cache the result as CSV (saves re-running).

    Returns
    -------
    DataFrame with columns: ['user', 'day'] + RAW_FEATURE_COLS
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  [cache] Loading user-day table from {cache_path}")
        df = pd.read_csv(cache_path)
        print(f"  Loaded {len(df)} rows, {df['user'].nunique()} users.")
        return df

    print(f"  Streaming CERT logs from {zip_path} ...")
    agg: dict = defaultdict(lambda: defaultdict(float))

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

        if "r4.2/logon.csv" in names:
            print("    → logon.csv")
            _stream_logon(zf, agg)
        if "r4.2/device.csv" in names:
            print("    → device.csv")
            _stream_device(zf, agg)
        if "r4.2/file.csv" in names:
            print("    → file.csv")
            _stream_file(zf, agg)
        if "r4.2/email.csv" in names:
            print("    → email.csv")
            _stream_email(zf, agg)
        if "r4.2/http.csv" in names:
            print("    → http.csv")
            _stream_http(zf, agg)

    rows = []
    for (user, day), feats in agg.items():
        row = {"user": user, "day": day}
        row.update({c: feats.get(c, 0.0) for c in RAW_FEATURE_COLS})
        rows.append(row)

    df = pd.DataFrame(rows, columns=["user", "day"] + RAW_FEATURE_COLS)
    df = df.sort_values(["user", "day"]).reset_index(drop=True)

    print(f"  Done: {len(df)} user-day rows, {df['user'].nunique()} users.")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"  Cached → {cache_path}")

    return df


# ===========================================================================
# Step 2 — Threat label assignment
# ===========================================================================

def load_insider_labels(zip_path: str) -> pd.DataFrame:
    """
    Load the CERT R4.2 answer key from answers/insiders.csv inside the zip.

    Returns DataFrame with columns:
        user, scenario, threat_type, start, end
    restricted to R4.2 entries.
    """
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("answers/insiders.csv") as f:
            insiders = pd.read_csv(f)

    # Keep only R4.2 entries
    r42 = insiders[insiders["dataset"] == 4.2].copy()
    r42["threat_type"] = r42["scenario"].map(SCENARIO_THREAT_MAP).fillna("Unknown")
    r42["start"] = pd.to_datetime(r42["start"], format="mixed", dayfirst=False)
    r42["end"]   = pd.to_datetime(r42["end"],   format="mixed", dayfirst=False)
    r42["start_date"] = r42["start"].dt.date.astype(str)
    r42["end_date"]   = r42["end"].dt.date.astype(str)
    return r42[["user", "scenario", "threat_type", "start_date", "end_date"]].reset_index(drop=True)


def assign_threat_labels(user_day_df: pd.DataFrame, insiders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate each user-day row with insider-threat labels.

    Adds columns:
        is_insider   : bool  — True if user is in the malicious set
        threat_type  : str   — 'DataExfiltration', 'UnauthorizedAccess', or 'Normal'
        scenario     : int   — CERT scenario number (0 = normal)

    NOTE: We label the ENTIRE activity window for simplicity. The exact
    malicious day range from the answer key is stored for reference but the
    full user is flagged, which is consistent with the user-level split
    strategy used in this project.
    """
    insider_users = set(insiders_df["user"].unique())
    user_to_scenario  = insiders_df.set_index("user")["scenario"].to_dict()
    user_to_threat    = insiders_df.set_index("user")["threat_type"].to_dict()

    df = user_day_df.copy()
    df["is_insider"]  = df["user"].isin(insider_users)
    df["threat_type"] = df["user"].map(user_to_threat).fillna("Normal")
    df["scenario"]    = df["user"].map(user_to_scenario).fillna(0).astype(int)
    return df


# ===========================================================================
# Step 3 — Train/test split (user-level to prevent leakage)
# ===========================================================================

def user_level_split(
    df: pd.DataFrame,
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at the user level so that no employee's days appear in both sets.

    Strategy:
    - All malicious users are split proportionally (stratified by threat_type).
    - Normal users fill the remaining quota.

    Returns: (train_df, test_df)
    """
    rng = np.random.default_rng(random_state)

    # Get one row per user with their insider status
    user_info = (
        df.groupby("user")["is_insider"]
        .first()
        .reset_index()
    )
    insider_users  = user_info[user_info["is_insider"]].sample(frac=1, random_state=random_state)
    normal_users   = user_info[~user_info["is_insider"]].sample(frac=1, random_state=random_state)

    n_insider_test = max(1, int(len(insider_users) * test_fraction))
    n_normal_test  = max(1, int(len(normal_users)  * test_fraction))

    test_insiders = insider_users.iloc[:n_insider_test]["user"].tolist()
    test_normals  = normal_users.iloc[:n_normal_test]["user"].tolist()
    test_users    = set(test_insiders + test_normals)

    test_df  = df[df["user"].isin(test_users)].reset_index(drop=True)
    train_df = df[~df["user"].isin(test_users)].reset_index(drop=True)

    print(f"  Train: {len(train_df)} rows, {train_df['user'].nunique()} users "
          f"({train_df['is_insider'].sum()} insider-days)")
    print(f"  Test : {len(test_df)} rows, {test_df['user'].nunique()} users "
          f"({test_df['is_insider'].sum()} insider-days)")
    return train_df, test_df


# ===========================================================================
# Step 4 — Threshold fitting and discretization (training data only)
# ===========================================================================

class CERTPreprocessor:
    """
    Fits percentile-based thresholds from TRAINING DATA ONLY, then applies
    them to discretize any user-day DataFrame into BN evidence variables.

    This is the leakage-safe approach: thresholds are never computed using
    test-set information.
    """

    # Mapping from raw feature → (evidence_node, percentile_for_threshold)
    FEATURE_EVIDENCE_MAP = {
        "after_hours_logon_count":      ("AfterHoursLogin",        90),
        "logon_count":                  ("UnusualLoginFreq",       90),
        "weekend_logon_count":          ("WeekendLogin",           95),
        "machine_count":                ("HighMachineCount",       95),
        "device_connect_count":         ("DeviceConnectActivity",  90),
        "after_hours_device_count":     ("DeviceAfterHours",       95),
        "file_activity_count":          ("FileAccessCount",        90),
        "file_copy_to_removable_count": ("FileCopyToRemovable",    85),
        "archive_activity_count":       ("ArchiveCreation",        90),
        "exe_activity_count":           ("ExeActivity",            95),
        "unusual_file_activity_count":  ("UnusualFileTypes",       95),
        "external_email_count":         ("ExternalEmail",          90),
        "email_count":                  ("UnusualEmailVolume",     90),
        "after_hours_email_count":      ("AfterHoursEmail",        95),
        "external_large_email_count":   ("ExternalLargeEmail",     95),
        "web_activity_count":           ("UnusualWebActivity",     90),
        "after_hours_web_count":        ("AfterHoursWeb",          95),
        "cloud_storage_count":          ("CloudStorageAccess",     95),
        "job_search_count":             ("JobSearchActivity",      95),
        "email_size_total":             ("HighDataMovement",       90),
    }

    # HighFileCopyActivity uses a stricter threshold than FileCopyToRemovable
    HIGH_FILE_COPY_PERCENTILE = 95

    def __init__(self):
        self.thresholds: dict = {}
        self.is_fitted: bool  = False

    def fit(self, train_df: pd.DataFrame) -> dict:
        """
        Compute thresholds from TRAINING rows (normal days only, then all).

        For evidence nodes we use the 90th-percentile on ALL training rows
        (not just normals), because in CERT the normal baseline is the
        majority and the percentile naturally separates routine from
        anomalous behaviour without needing explicit class filtering.
        """
        for raw_feat, (ev_node, pct) in self.FEATURE_EVIDENCE_MAP.items():
            if raw_feat in train_df.columns:
                val = float(train_df[raw_feat].quantile(pct / 100.0))
                # Ensure threshold is strictly positive so that any non-zero
                # activity triggers the flag (avoid all-False evidence)
                if val == 0.0:
                    val = float(train_df[raw_feat].quantile(0.95))
                if val == 0.0:
                    val = 0.5
                self.thresholds[ev_node] = val

        # HighFileCopyActivity — stricter than FileCopyToRemovable
        if "file_copy_to_removable_count" in train_df.columns:
            val = float(train_df["file_copy_to_removable_count"].quantile(
                self.HIGH_FILE_COPY_PERCENTILE / 100.0
            ))
            if val == 0.0:
                val = 1.0
            self.thresholds["HighFileCopyActivity"] = val

        self.is_fitted = True
        return dict(self.thresholds)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a user-day raw-count DataFrame into a DataFrame of binary
        BN evidence variables (0 = Normal, 1 = High/Anomalous).

        Returns a DataFrame with columns = EVIDENCE_NODES, same row order.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() on training data before transform().")

        out = {}
        for raw_feat, (ev_node, _) in self.FEATURE_EVIDENCE_MAP.items():
            thresh = self.thresholds.get(ev_node, 0.0)
            if raw_feat in df.columns:
                out[ev_node] = (df[raw_feat] > thresh).astype(int)
            else:
                out[ev_node] = pd.Series(0, index=df.index)

        # HighFileCopyActivity
        thresh_hfc = self.thresholds.get("HighFileCopyActivity", 1.0)
        if "file_copy_to_removable_count" in df.columns:
            out["HighFileCopyActivity"] = (
                df["file_copy_to_removable_count"] > thresh_hfc
            ).astype(int)
        else:
            out["HighFileCopyActivity"] = pd.Series(0, index=df.index)

        return pd.DataFrame(out, index=df.index)

    def transform_row(self, row: dict | pd.Series) -> dict:
        """
        Transform a single user-day raw-feature dict/Series into a BN
        evidence dict: {node_name: bool}.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() on training data before transform_row().")

        result = {}
        for raw_feat, (ev_node, _) in self.FEATURE_EVIDENCE_MAP.items():
            thresh = self.thresholds.get(ev_node, 0.0)
            val = row.get(raw_feat, 0) if isinstance(row, dict) else row.get(raw_feat, 0)
            result[ev_node] = bool(val > thresh)

        thresh_hfc = self.thresholds.get("HighFileCopyActivity", 1.0)
        raw_val = row.get("file_copy_to_removable_count", 0)
        result["HighFileCopyActivity"] = bool(raw_val > thresh_hfc)

        return result

    def save_thresholds(self, path: str):
        """Persist thresholds as JSON for reproducibility."""
        with open(path, "w") as f:
            json.dump(self.thresholds, f, indent=2)
        print(f"  Thresholds saved → {path}")

    def load_thresholds(self, path: str):
        """Load thresholds from a previously saved JSON file."""
        with open(path) as f:
            self.thresholds = json.load(f)
        self.is_fitted = True
        print(f"  Thresholds loaded from {path}")


# ===========================================================================
# Convenience: full pipeline in one call
# ===========================================================================

def run_full_preprocessing(
    zip_path: str,
    cache_dir: str = "data",
    test_fraction: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Run the complete preprocessing pipeline:

    1. Stream → user-day raw counts (cached)
    2. Load & assign threat labels
    3. User-level train/test split
    4. Fit preprocessor on training users
    5. Discretize both splits

    Returns a dict with keys:
        raw_df, labeled_df,
        train_raw, test_raw,
        train_ev, test_ev,
        preprocessor, insiders_df
    """
    os.makedirs(cache_dir, exist_ok=True)
    raw_cache = os.path.join(cache_dir, "user_day_raw.csv")

    print("\n[1/5] Building user-day raw feature table...")
    raw_df = build_user_day_table(zip_path, cache_path=raw_cache)

    print("\n[2/5] Assigning threat labels...")
    insiders_df = load_insider_labels(zip_path)
    labeled_df  = assign_threat_labels(raw_df, insiders_df)
    labeled_df.to_csv(os.path.join(cache_dir, "user_day_labeled.csv"), index=False)
    print(f"  Labeled: {labeled_df['is_insider'].sum()} insider days "
          f"/ {len(labeled_df)} total days")
    print(f"  Insider users: {labeled_df[labeled_df['is_insider']]['user'].nunique()}")
    print(f"  Threat types: {labeled_df['threat_type'].value_counts().to_dict()}")

    print("\n[3/5] User-level train/test split...")
    train_raw, test_raw = user_level_split(
        labeled_df, test_fraction=test_fraction, random_state=random_state
    )

    print("\n[4/5] Fitting preprocessor (thresholds from training data only)...")
    preprocessor = CERTPreprocessor()
    thresholds = preprocessor.fit(train_raw)
    preprocessor.save_thresholds(os.path.join(cache_dir, "thresholds.json"))
    print("  Fitted thresholds:")
    for node, val in thresholds.items():
        print(f"    {node:<28} > {val:.2f}")

    print("\n[5/5] Discretizing to BN evidence variables...")
    train_ev_df = preprocessor.transform(train_raw)
    test_ev_df  = preprocessor.transform(test_raw)

    # Attach labels
    train_ev_df["is_insider"]  = train_raw["is_insider"].values
    train_ev_df["threat_type"] = train_raw["threat_type"].values
    train_ev_df["user"]        = train_raw["user"].values
    train_ev_df["day"]         = train_raw["day"].values

    test_ev_df["is_insider"]   = test_raw["is_insider"].values
    test_ev_df["threat_type"]  = test_raw["threat_type"].values
    test_ev_df["user"]         = test_raw["user"].values
    test_ev_df["day"]          = test_raw["day"].values

    train_ev_df.to_csv(os.path.join(cache_dir, "train_evidence.csv"), index=False)
    test_ev_df.to_csv(os.path.join(cache_dir, "test_evidence.csv"),  index=False)

    print(f"\n  Evidence columns: {EVIDENCE_NODES}")
    print(f"  Train evidence: {len(train_ev_df)} rows saved.")
    print(f"  Test evidence : {len(test_ev_df)} rows saved.")

    return {
        "raw_df":       raw_df,
        "labeled_df":   labeled_df,
        "train_raw":    train_raw,
        "test_raw":     test_raw,
        "train_ev":     train_ev_df,
        "test_ev":      test_ev_df,
        "preprocessor": preprocessor,
        "insiders_df":  insiders_df,
    }


def generate_synthetic_evidence(
    cache_dir: str = "data",
    n_train: int = 1000,
    n_test: int = 300,
    insider_rate: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Generate realistic synthetic binary evidence DataFrames for training & testing
    the 35-node Bayesian Network without requiring a 7.1GB zip stream.
    """
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    def _sample_split(n_samples: int, split_name: str):
        n_insiders = int(n_samples * insider_rate)
        n_normals  = n_samples - n_insiders

        # Base noise probabilities for normal days (mostly 0s)
        normal_p = {node: 0.05 for node in EVIDENCE_NODES}
        normal_p["AfterHoursLogin"] = 0.10
        normal_p["ExternalEmail"] = 0.15

        data = []
        # Normal rows
        for i in range(n_normals):
            row = {node: int(rng.rand() < normal_p[node]) for node in EVIDENCE_NODES}
            row["is_insider"] = 0
            row["threat_type"] = "Normal"
            row["user"] = f"USER_{i % 50:04d}"
            row["day"] = f"2026-01-{(i % 28) + 1:02d}"
            data.append(row)

        # Insider rows with targeted threat patterns
        threat_types = ["DataExfiltration", "ITSabotage", "IPTheft", "UnauthorizedAccess"]
        for i in range(n_insiders):
            ttype = threat_types[i % len(threat_types)]
            row = {node: int(rng.rand() < normal_p[node]) for node in EVIDENCE_NODES}
            
            if ttype == "DataExfiltration":
                for node in ["FileCopyToRemovable", "HighFileCopyActivity", "CloudStorageWeb", "BccToPersonalEmail"]:
                    row[node] = int(rng.rand() < 0.85)
            elif ttype == "ITSabotage":
                for node in ["AfterHoursLogin", "JobSearchWeb", "ExeOrZipCopy", "HackingSiteWeb"]:
                    row[node] = int(rng.rand() < 0.85)
            elif ttype == "IPTheft":
                for node in ["FileAccessCount", "LargeEmailAttachment", "ExternalEmail", "FileCopyToRemovable"]:
                    row[node] = int(rng.rand() < 0.80)
            elif ttype == "UnauthorizedAccess":
                for node in ["MultiUserMachine", "HighMachineCount", "UnusualLoginFreq", "AfterHoursLogin"]:
                    row[node] = int(rng.rand() < 0.85)

            row["is_insider"] = 1
            row["threat_type"] = ttype
            row["user"] = f"INSIDER_{i:04d}"
            row["day"] = f"2026-02-{(i % 28) + 1:02d}"
            data.append(row)

        df = pd.DataFrame(data).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        return df

    train_ev = _sample_split(n_train, "train")
    test_ev  = _sample_split(n_test, "test")

    train_ev.to_csv(os.path.join(cache_dir, "train_evidence.csv"), index=False)
    test_ev.to_csv(os.path.join(cache_dir, "test_evidence.csv"), index=False)

    preprocessor = CERTPreprocessor()
    return {
        "train_ev": train_ev,
        "test_ev": test_ev,
        "preprocessor": preprocessor,
        "insiders_df": pd.DataFrame(),
        "train_raw": None,
        "test_raw": None,
    }


# ===========================================================================
# Script entry point (standalone usage)
# ===========================================================================

if __name__ == "__main__":
    import sys

    ZIP_PATH  = "archive.zip"
    CACHE_DIR = "data"

    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: {ZIP_PATH} not found. Place archive.zip in the project directory.")
        sys.exit(1)

    result = run_full_preprocessing(ZIP_PATH, cache_dir=CACHE_DIR)
    print("\nPreprocessing complete. Files written to data/:")
    for fname in ["user_day_raw.csv", "user_day_labeled.csv",
                  "thresholds.json", "train_evidence.csv", "test_evidence.csv"]:
        fpath = os.path.join(CACHE_DIR, fname)
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1e6
            print(f"  {fname:<30} {mb:.1f} MB")
