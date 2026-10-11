"""
run_pipeline.py - run the whole v3 system from one command.

    python run_pipeline.py                      # use the shipped data/ files (no archive.zip needed)
    python run_pipeline.py --zip archive.zip    # start from the raw archive (slow: streams ~14.5 GB of logs)
    python run_pipeline.py --skip-cv            # only rebuild the app model files (data/model_params.json ...)

Steps
  0. (only with --zip) cert_build_table.py -> data/user_day_raw.csv.gz
                       cert_labels.py      -> data/attack_days.csv, data/insider_windows.csv
  1. test_all.py      checks: variable elimination vs enumeration, decision engine
  2. cv_bn.py         5-fold user-grouped cross-validation -> results/oof_bn.npy and printed metrics
  3. make_results.py  plots + CPT tables -> results/
  4. build_model.py   final fit on all users -> data/model_params.json, flag_rules.json, samples.json
Then start the web app:   python cert_app_server.py     (open http://localhost:5000)
"""
import os
import subprocess
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="path to archive.zip (optional)")
    ap.add_argument("--skip-cv", action="store_true", help="skip steps 1-3")
    a = ap.parse_args()

    if a.zip:
        run("cert_build_table.py", a.zip)
        run("cert_labels.py", a.zip)
        cache = os.path.join(HERE, "data", "cache", "v3_base.pkl")
        if os.path.exists(cache):
            os.remove(cache)          # force the labelled/baseline table to be rebuilt from the new files
    if not a.skip_cv:
        run("test_all.py")
        run("cv_bn.py")
        run("make_results.py")
    run("build_model.py")
    print("\nDone. Start the app with:  python cert_app_server.py")
