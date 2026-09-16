# Probabilistic Insider Threat Risk Assessment using Bayesian Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: CERT R4.2](https://img.shields.io/badge/Dataset-CERT--R4.2-red.svg)](https://doi.org/10.1184/R1/12841247.v1)
[![Inference: VE%20%26%20MCMC](https://img.shields.io/badge/Inference-VE%20%26%20MCMC-brightgreen.svg)]()
[![Web Dashboard](https://img.shields.io/badge/Dashboard-Flask%20%2B%20D3/Vis.js-orange.svg)]()

> **Probabilistic Reasoning & Risk Analysis Project**  
> A custom 35-Node Directed Acyclic Graph (DAG) Bayesian Network with Variable Elimination (VE), Gibbs Sampling (MCMC), Expected Utility Decision Networks, and LLM-ready explainability for insider threat detection on the **Carnegie Mellon University (CMU) SEI CERT Dataset R4.2**.

---

## 📌 Executive Summary

Traditional Security Information and Event Management (SIEM) rules generate thousands of alerts per day with high false-positive rates, leading to analyst fatigue. Rule-based alerts fail to capture compound behavioral signals or measure uncertainty.

This project introduces a **35-Node Bayesian Network (BN)** that models complex dependencies between observed audit logs, behavioral anomalies, threat hypotheses, and overall organizational risk:
- **Exact & Approximate Probabilistic Inference**: Built-in Variable Elimination (VE) algorithm (min-degree elimination heuristic) and Gibbs Sampling (MCMC).
- **Decision Network Extension**: Calculates Expected Utility $EU(a)$ over intervention actions (e.g., *Monitor*, *Revoke Access*, *Escalate to HR*, *No Action*).
- **Posterior Sensitivity & Evidence Ablation**: Quantifies the contribution $\Delta P$ of each observed indicator to isolate root causes.
- **Interactive Web Dashboard**: Real-time Flask application with interactive node graph, scenario simulator, and LLM narrative generator.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[CERT R4.2 Log Streams<br/>logon, device, email, http, file] -->|Stream Aggregation| B[User-Day Feature Table]
    B -->|Quantile Binning & Thresholding| C[35-Node Categorical Evidence]
    C -->|User-Level Train Split| D[CPT Parameter Learning]
    D --> E[35-Node Directed Acyclic Graph]
    
    E --> F1[Exact Inference: Variable Elimination]
    E --> F2[Approximate Inference: Gibbs Sampling MCMC]
    
    F1 --> G1[Risk Level Posteriors P(Low, Med, High)]
    F1 --> G2[Threat Posteriors P(IPTheft, Exfil, Sabotage, Unauth)]
    F1 --> G3[Expected Utility Decision Network]
    
    G1 & G2 & G3 --> H[Evidence Contribution Sensitivity ΔP]
    H --> I[LLM Narrative & Web Dashboard]
```

---

## 🧬 35-Node Bayesian Network Topology

The 35 nodes are structured across 4 distinct functional layers:

```
[Level 0: Organizational Risk] (1 Node)
       └── RiskLevel (Low, Medium, High)

[Level 1: Threat Hypotheses] (4 Nodes)
       ├── DataExfiltration
       ├── IPTheft
       ├── ITSabotage
       └── UnauthorizedAccess

[Level 2: Behavioral Anomaly Indicators] (9 Nodes)
       ├── FlightRiskAnomaly         ├── RemovableMediaAnomaly
       ├── CommunicationAnomaly      ├── TimeBasedAnomaly
       ├── DataAccessAnomaly         ├── AuthAnomaly
       ├── DataMovementAnomaly       ├── StagingAnomaly
       └── ShadowITAnomaly

[Level 3: Observational Evidence Nodes] (21 Nodes)
       ├── AfterHoursLogin           ├── RemovableMediaInsert
       ├── FailedLogonCount          ├── USBWriteVolume
       ├── DeviceConnectFreq         ├── FileAccessCount
       ├── USBConnectCount           ├── SensitiveFileRead
       ├── ExternalEmail             ├── HighVolumeFileAccess
       ├── ExternalLargeEmail        ├── ArchiveCreation
       ├── NonWorkHoursEmail         ├── CloudUploadActivity
       ├── BccToPersonalEmail        ├── AdminToolExecution
       ├── WebAccessVolume           ├── SecurityToolDisable
       ├── JobSearchWebCount         └── UnauthorizedPortAccess
       └── SensitiveWebSearch
```

---

## 🧮 Mathematical Formulation

### 1. Joint Probability Distribution & CPT Factorization
$$\mathbb{P}(X_1, X_2, \dots, X_{35}) = \prod_{i=1}^{35} \mathbb{P}\left(X_i \mid \text{Parents}(X_i)\right)$$

### 2. Exact Inference via Variable Elimination (VE)
Given evidence $\mathbf{e} = \{e_1, e_2, \dots, e_k\}$ and target query $Q$:
$$\mathbb{P}(Q \mid \mathbf{e}) = \alpha \sum_{\mathbf{Y}} \prod_{\phi \in \Phi} \phi(\mathbf{X}_\phi)$$
where $\mathbf{Y} = \mathbf{X} \setminus (\{Q\} \cup \mathbf{E})$, $\Phi$ is the set of initial Conditional Probability Tables (CPTs), and elimination ordering is determined via min-degree heuristic.

### 3. Posterior Evidence Sensitivity & Ablation ($\Delta P$)
To determine which observed evidence contributed most to a **HIGH** risk score:
$$\Delta P(e_j) = \mathbb{P}(\text{RiskLevel} = \text{High} \mid \mathbf{e}) - \mathbb{P}(\text{RiskLevel} = \text{High} \mid \mathbf{e} \setminus \{e_j\})$$

### 4. Decision Network Expected Utility $EU(a)$
$$EU(a \mid \mathbf{e}) = \sum_{s \in \text{States}} \mathbb{P}(s \mid \mathbf{e}) \cdot U(a, s)$$

---

## 📊 Benchmark Experiments & Machine Learning Baselines

Evaluating the Bayesian Network against conventional Machine Learning baselines on CERT R4.2 test evidence:

| Model / Algorithm | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **35-Node Bayesian Network (VE)** | **0.1806** | **0.8667** | **0.2989** | **0.8611** | **0.4538** | **0.1500** |
| Random Forest Classifier | 0.4650 | 0.4667 | 0.4650 | 0.8980 | — | — |
| XGBoost Classifier | 0.4000 | 0.4000 | 0.4000 | 0.8850 | — | — |
| Logistic Regression | 0.3660 | 0.3660 | 0.3660 | 0.9100 | — | — |
| Naive Bayes | 0.0930 | 0.6000 | 0.0930 | 0.7200 | — | — |
| Isolation Forest (Unsupervised) | 0.0500 | 0.3333 | 0.0870 | 0.5120 | — | — |

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/ArKaarthik06/cybersecurity-risk-analysis.git
cd cybersecurity-risk-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline CLI
```bash
# Run with synthetic benchmark evidence generation
python CERT_35Node_BN/cert_main.py --synthetic

# Run demo cases only
python CERT_35Node_BN/cert_main.py --demo-only
```

### 4. Launch Interactive Web UI
```bash
python CERT_35Node_BN/cert_app_server.py
```
Open your browser at `http://localhost:5000` to interact with the risk simulator, evidence toggles, NetworkX graph visualization, and math tracer.

---

## 📁 Repository Structure

```
cybersecurity-risk-analysis/
├── CERT_35Node_BN/                 # Main Bayesian Network Module
│   ├── data/                      # Evidence datasets (train_evidence.csv, test_evidence.csv)
│   ├── results/                   # CPT Markdown tables, PNG convergence charts, JSON reports
│   ├── static/                    # Web UI frontend (HTML, CSS, JS dashboard)
│   ├── bayes_net.py               # Custom BN Graph, Factor, and Inference engine
│   ├── cert_app_server.py         # Flask Web Dashboard server
│   ├── cert_baselines.py          # ML Baseline benchmarking (RF, XGBoost, IsoForest)
│   ├── cert_bn_builder.py         # 35-Node DAG topology & CPT parameter learning
│   ├── cert_decision_network.py   # Expected Utility Decision Network
│   ├── cert_experiments.py        # Automated MCMC, VE & Calibration experiments
│   ├── cert_main.py               # Main CLI execution orchestrator
│   ├── cert_math_tracer.py        # Step-by-step mathematical explanation generator
│   ├── cert_preprocessing.py      # CERT R4.2 log aggregator & feature extractor
│   ├── cert_reporter.py           # LLM-style markdown report builder
│   ├── generate_cpt_md.py         # CPT exporter script
│   ├── inference_mcmc.py          # Gibbs Sampling engine
│   ├── inference_ve.py            # Variable Elimination engine
│   ├── requirements.txt           # Dependencies specification
│   └── visualize_bn.py            # Matplotlib / NetworkX Graph visualizer
├── Base Papers/                    # Academic research reference papers
├── docs/                           # Project design specifications & prompts
├── cert_stage1_preprocessing.ipynb # Stage 1 data exploration notebook
├── .gitignore                      # Git exclusion rules
├── requirements.txt                # Global dependencies specification
└── README.md                       # Project documentation
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
