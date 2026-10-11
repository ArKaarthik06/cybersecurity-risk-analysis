# CERT_35Node_BN_v3 - standalone project

Insider-threat inference with a 35-node Bayesian network (Noisy-OR parameters, exact variable
elimination, decision network with maximum expected utility), plus the web app with the LLM analysis tab.

**Everything the system needs is inside this folder.** No file outside it is read. Copy the folder anywhere and run.

## Quick start

```
pip install -r requirements.txt
python run_pipeline.py          # test, cross-validate, make plots, fit the final model (a few minutes)
python cert_app_server.py       # web app at http://127.0.0.1:<port printed on start>
```

The shipped `data/` files are enough to run everything. `archive.zip` (6.65 GB) is only needed if you want to
rebuild the table from the raw logs: `python run_pipeline.py --zip path/to/archive.zip`.

For the LLM tab put your key in `results/keys.env` as `OPENROUTER_API_KEY=...` (no key is shipped).
Without a key the tab shows a template text built from the same numbers.

## Pipeline, from archive.zip to the app

| Step | File | Reads | Writes |
|---|---|---|---|
| 1. Daily counts | `cert_build_table.py` | `r4.2/logon.csv, device.csv, file.csv, email.csv, http.csv` inside archive.zip (streamed, 200,000 rows at a time) | `data/user_day_raw.csv.gz` (330,285 rows, 20 counts) |
| 2. Labels | `cert_labels.py` | `answers/insiders.csv`, `answers/r4.2-N/*.csv` | `data/attack_days.csv` (966 attack days), `data/insider_windows.csv` |
| 3. Baselines and flags | `cert_preprocessing.py` (+ `v3_common.py`) | the two data files above | `data/cache/v3_base.pkl` (cache), flag rules |
| 4. Network and learning | `bayes_net.py`, `cert_bn_builder.py` | flags, labels | learned priors, Noisy-OR weights, RiskLevel table |
| 5. Inference | `inference_ve.py` (exact), `inference_mcmc.py` | evidence | posteriors |
| 6. Decision | `cert_decision_network.py` | posteriors | action by MEU |
| 7. Evaluation | `cv_bn.py`, `cert_evaluation.py`, `make_results.py` | base table | `results/` plots, CPT tables, scores |
| 8. App model | `build_model.py` | base table | `data/model_params.json`, `flag_rules.json`, `samples.json` |
| 9. Web app | `cert_app_server.py`, `cert_reporter.py`, `cert_math_tracer.py`, `static/` | the files in `data/` | - |

LDAP and the psychometric files in the archive are not used.

## Files

```
run_pipeline.py        one command for steps 1-8
cert_build_table.py    archive.zip -> 20 daily counts per user-day
cert_labels.py         answers/ -> attack-day labels
cert_preprocessing.py  per-user baselines, z-scores, the 21 evidence flags (FlagRules), grouped folds
cert_bn_builder.py     network structure (47 edges) and parameter learning
bayes_net.py           factors and the network
inference_ve.py        exact variable elimination      inference_mcmc.py  sampling (comparison only)
cert_decision_network.py  actions, costs, losses, MEU
cert_evaluation.py     metrics (PR-AUC, F1, recall at fixed FPR, ECE, user-level)
cv_bn.py  make_results.py  test_all.py  build_model.py
cert_app_server.py  cert_reporter.py  cert_math_tracer.py  static/
data/      user_day_raw.csv.gz, attack_days.csv, insider_windows.csv, model_params.json, flag_rules.json, samples.json
results/   v3_evaluation.png, CPT_Tables.md, flag_rules.csv, evidence_node_analysis.csv
```

## Results this code reproduces (5-fold CV, folds split by user)

ROC-AUC 0.984, PR-AUC 0.849, F1 0.839, recall at 0.5% false-alarm rate 0.93, calibration error 0.002,
69 of 70 insiders found with 1 false alarm in 930 normal users.

## Things to state honestly

- The result leans on one flag (`UnusualWebActivity`). Without it AUC is about 0.88 and F1 about 0.31.
- Two count columns are always 0 (`external_large_email_count`, `unusual_file_activity_count`).
- Labels use two classes in training: attack day and every other day. The Medium risk probabilities in the
  RiskLevel table come from smoothing, not from labelled Medium days (`load_attack_days` + `attach_labels` without
  `windows`; pass `data/insider_windows.csv` as `windows` to try a three-class label).
- Costs and losses in the decision network are assumptions, not measured values.
- Network structure is expert-designed, not learned.
