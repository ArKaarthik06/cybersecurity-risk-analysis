"""
cert_build_table.py
===================
Step 1 of the pipeline: build the user-day count table straight from archive.zip.

This is the streaming aggregation from the original CERT_35Node_BN pipeline, copied here
unchanged so that CERT_35Node_BN_v3 does not depend on any other folder.

Reads five logs inside the zip, 200,000 rows at a time (nothing is extracted to disk):
    r4.2/logon.csv   r4.2/device.csv   r4.2/file.csv   r4.2/email.csv   r4.2/http.csv
and writes one row per (user, day) with 20 count columns -> data/user_day_raw.csv.gz

Usage:
    python cert_build_table.py path/to/archive.zip            # writes data/user_day_raw.csv.gz
"""
import os
import sys
import zipfile
import pandas as pd
from collections import defaultdict

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python cert_build_table.py path/to/archive.zip [out_csv]")
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data", "user_day_raw.csv.gz")
    build_user_day_table(sys.argv[1], cache_path=out)
