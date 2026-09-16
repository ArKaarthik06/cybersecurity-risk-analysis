I am developing a high-quality **Probabilistic Reasoning course project in cybersecurity**. I want to redesign my original project so that it focuses on **Bayesian Networks only** for probabilistic threat assessment, with an **LLM used only as a final natural-language explanation/report-generation layer**.

Do NOT use HMMs, Hidden Markov Models, Dynamic Bayesian Networks, or BN+HMM hybrid models.

The project should remain fundamentally a **Probabilistic Reasoning project**, not become a generic machine-learning or LLM project.

---

# 1. PROJECT TITLE

Use the following working title:

**Probabilistic Insider Threat Risk Assessment Using Bayesian Networks with LLM-Based Explainability**

Alternative shorter title:

**Bayesian Network-Based Probabilistic Insider Threat Assessment with LLM Explainability**

---

# 2. PROJECT OBJECTIVE

The objective is to develop a probabilistic cybersecurity system that analyzes employee activity and estimates:

1. The probability that the observed behavior represents an insider threat.
2. The probability of different threat categories.
3. The major pieces of evidence contributing to the risk.
4. How the posterior risk changes when important evidence is removed.
5. A human-readable explanation of the probabilistic assessment using an LLM.

The system must NOT claim that an employee is definitively malicious.

The correct interpretation is:

> Given the observed behavioral evidence, the Bayesian Network estimates the posterior probability of different insider-threat hypotheses.

The LLM does NOT perform the actual threat detection. The Bayesian Network performs probabilistic inference, while the LLM converts the structured probabilistic results into an understandable cybersecurity report.

---

# 3. DATASET

Use the:

**CERT Insider Threat Test Dataset R4.2**

Source:

Carnegie Mellon University Software Engineering Institute (CERT Insider Threat Center).

The CERT Insider Threat Test Dataset is a collection of synthetic insider-threat datasets containing both background activity and malicious-actor activity. The official dataset also provides an answer key describing malicious scenarios and identifying the synthetic users involved.

Official source:

https://sei.cmu.edu/library/insider-threat-test-dataset/

Dataset DOI:

10.1184/R1/12841247.v1

Do not replace CERT with UNSW-NB15 for the main project.

UNSW-NB15 can be mentioned as a possible future/secondary dataset, but the main implementation must use CERT.

---

# 4. WHY CERT IS USED

The project should justify CERT over a conventional network-flow dataset.

The reasoning should be:

UNSW-NB15 primarily represents network-flow intrusion data, whereas the CERT Insider Threat Test Dataset provides multi-source employee activity such as authentication/logon behavior, device usage, file activity, email activity, and web activity.

The project is therefore able to construct meaningful behavioral evidence variables for insider-threat assessment.

CERT is also appropriate for probabilistic reasoning because the system does not need to rely on one event to identify a threat. Instead, multiple uncertain behavioral indicators can jointly influence the posterior probability of a threat.

Important:

Do not claim that CERT contains real employee behavior. Explicitly describe it as synthetic data.

---

# 5. DATA SOURCES / LOG TYPES

Use the relevant CERT activity sources, such as:

- Logon activity
- Device / removable-media activity
- File activity
- Email activity
- HTTP / web activity

Do not necessarily feed every raw field into the Bayesian Network.

Instead, convert raw event logs into meaningful behavioral evidence variables.

---

# 6. UNIT OF ANALYSIS

Use a **user-day** as the main unit of analysis.

For each employee and each day, aggregate their activities into a feature vector.

Example:

User U123 on Day D:

- Number of logons
- Number of after-hours logons
- Number of device connections
- Number of file accesses
- Number of unusual/sensitive file accesses
- Number of external emails
- Number of unusual web accesses
- Amount of outbound data/activity
- Number of files copied
- etc.

The final Bayesian Network should operate on these behavioral evidence variables.

Avoid using individual raw events as separate BN nodes because that would make the network unnecessarily large and difficult to interpret.

---

# 7. DATA PREPROCESSING

Build a preprocessing pipeline:

Raw CERT logs
→ timestamp normalization
→ user identification
→ event aggregation
→ user-day feature construction
→ behavioral feature normalization
→ discretization
→ Bayesian Network evidence

For numerical variables, convert them into categorical states.

Example:

File activity:

- Normal
- High

Login activity:

- Normal
- After-hours

Device activity:

- Normal
- Unusual

Data movement:

- Normal
- High

The thresholds should NOT be arbitrarily chosen.

Possible approaches:

1. Percentile-based thresholds calculated from training data.
2. Training-set statistical thresholds.
3. Domain-motivated thresholds where appropriate.

If percentile thresholds are used, calculate them using TRAINING DATA ONLY to avoid data leakage.

---

# 8. IMPORTANT DATA-LEAKAGE RULE

Never use test-set information to construct features, thresholds, CPTs, graph structure, or priors.

The experimental pipeline must be:

Training users/days
→ feature statistics
→ BN structure
→ CPT estimation

Test users/days
→ evidence extraction using training-derived thresholds
→ BN inference
→ evaluation

Prefer a **user-level split** rather than a random row-level split wherever practical.

The goal is to prevent the same employee's behavior patterns from appearing in both training and test sets.

---

# 9. TARGET VARIABLE

Define the main target as:

**Insider Threat Risk**

with states:

- Low Risk
- Medium Risk
- High Risk

Alternatively, use:

- Normal
- Suspicious
- High Risk

Choose one final representation and use it consistently.

The preferred version is:

**Low / Medium / High Risk**

The BN should ultimately estimate:

P(Risk | Evidence)

---

# 10. THREAT CATEGORIES

Where the CERT scenario labels support meaningful threat categories, construct threat-hypothesis nodes such as:

- Credential Misuse / Unauthorized Access
- Data Exfiltration
- Insider Sabotage

Do NOT invent unsupported scenario labels.

The exact threat categories must be mapped to the CERT answer key/scenarios actually present in the selected release/subset.

If the dataset does not support reliable multi-class threat categorization for a particular category, use a binary:

**Insider Threat vs Normal**

target for the main experiment and treat finer threat categories as an optional extension.

This is important because the project should not manufacture labels that the dataset does not actually contain.

---

# 11. BAYESIAN NETWORK — CORE DESIGN

The Bayesian Network should be the central component of the project.

Use a layered structure:

```text
Observed Behavioral Evidence
          ↓
Behavioral Indicators
          ↓
Threat Mechanisms
          ↓
Risk Assessment
```

The graph should NOT simply connect every feature to every other feature.

The graph must have a meaningful probabilistic structure.

---

# 12. PROPOSED BN GRAPH

Construct approximately 15–20 nodes.

Use the following conceptual structure.

## Layer 1 — Observable Evidence

Suggested evidence nodes:

1. After-Hours Login
2. Unusual Login Frequency
3. Unusual Device / USB Activity
4. Unusual File Access
5. Sensitive File Access
6. High File-Copy Activity
7. External Email Activity
8. Unusual Email Volume
9. Unusual Web Activity
10. High Data Movement

These are observed from the CERT logs.

---

# 13. LAYER 2 — BEHAVIORAL INDICATOR NODES

Create intermediate probabilistic nodes:

11. Authentication Anomaly
12. Data Access Anomaly
13. Data Movement Anomaly
14. Communication Anomaly
15. Removable Media Anomaly

These nodes represent higher-level behavioral patterns inferred from the observable evidence.

Example:

```text
After-Hours Login
        \
         → Authentication Anomaly
        /
Unusual Login Frequency
```

Another:

```text
Unusual File Access
        \
         → Data Access Anomaly
        /
Sensitive File Access
```

Another:

```text
High File-Copy Activity
        \
         → Data Movement Anomaly
        /
High Data Movement
```

---

# 14. LAYER 3 — THREAT HYPOTHESES

Create threat mechanism nodes according to the CERT scenarios actually selected.

Possible nodes:

16. Unauthorized Access
17. Data Exfiltration
18. Insider Sabotage

The relationships should be justified by the selected behavioral indicators.

For example:

```text
Authentication Anomaly
        ↓
Unauthorized Access
```

```text
Data Access Anomaly
        ────────────┐
                    ↓
              Data Exfiltration
                    ↑
Data Movement Anomaly ┘
```

```text
Data Access Anomaly
        ↓
Insider Sabotage
```

However, do not blindly use these edges.

The final graph must be justified using:

- Domain reasoning
- CERT scenario definitions
- Statistical relationships in training data
- Structure-learning experiments

---

# 15. FINAL RISK NODE

Create a final:

**Risk Level**

node.

Possible states:

- Low
- Medium
- High

Threat nodes should influence this final risk node.

Example:

```text
Unauthorized Access ──┐
                      │
Data Exfiltration ────┼──→ Risk Level
                      │
Insider Sabotage ─────┘
```

The complete conceptual graph should therefore resemble:

```text
                    OBSERVABLE EVIDENCE
                           │
       ┌───────────────────┼────────────────────┐
       ↓                   ↓                    ↓
Authentication        Data Access         Data Movement
  Evidence              Evidence             Evidence
       ↓                   ↓                    ↓
Authentication        Data Access        Data Movement
  Anomaly                Anomaly              Anomaly
       │                   │                    │
       ↓                   └─────────┬──────────┘
       ↓                             ↓
Unauthorized Access          Data Exfiltration
       │                             │
       └──────────────┬──────────────┘
                      ↓
                 Risk Level
```

Add Communication Anomaly and Removable Media Anomaly where supported by the actual feature relationships.

---

# 16. DO NOT MAKE THE GRAPH ARBITRARY

This is a major requirement.

The graph should be created using a combination of:

### A. Domain-informed structure

Use cybersecurity reasoning to define plausible relationships.

### B. Data-driven validation

Use training data to verify whether the proposed dependencies are statistically meaningful.

### C. Structure learning

Experiment with at least one structure-learning method such as:

- Hill Climbing + BIC
- K2
- PC algorithm

The exact algorithm can depend on the implementation library.

Compare:

1. Expert-designed BN
2. Data-learned BN
3. Hybrid BN

The hybrid BN should use domain-informed constraints/relationships while allowing the data to refine the structure.

The final selected BN should be justified rather than chosen arbitrarily.

---

# 17. CPT LEARNING

After determining the graph, estimate the Conditional Probability Tables from the training data.

Use:

- Maximum Likelihood Estimation
- Bayesian parameter estimation if appropriate

Compare expert-defined probabilities against learned probabilities if time permits.

For the final model, prefer learned CPTs because this makes the system reproducible and data-driven.

---

# 18. CORE PROBABILISTIC REASONING CONCEPTS

The project must explicitly demonstrate the following concepts:

### Probability fundamentals

- Random variables
- Probability distributions
- Joint probability
- Marginal probability
- Conditional probability
- Prior probability
- Posterior probability

### Bayesian reasoning

- Bayes' theorem
- Bayesian updating
- Diagnostic reasoning

### Bayesian Networks

- DAG
- Nodes
- Directed dependencies
- Parents
- Children
- Conditional independence
- Markov blanket
- CPTs
- Joint-distribution factorization

### Inference

- Evidence propagation
- Marginalization
- Posterior inference
- Query variables
- Evidence variables
- Hidden variables

---

# 19. JOINT DISTRIBUTION

Explain that the Bayesian Network compactly represents the joint probability distribution.

For variables X1,...,Xn:

P(X1,...,Xn)
=
Π P(Xi | Parents(Xi))

Use this equation as one of the core mathematical foundations of the project.

---

# 20. DIAGNOSTIC REASONING

The main query should be:

P(Risk | Evidence)

and optionally:

P(ThreatType | Evidence)

For example:

P(High Risk | USB Activity, File Access, After-Hours Login)

The system should output posterior probabilities rather than only a hard classification.

Example:

```text
Low Risk       = 0.08
Medium Risk    = 0.21
High Risk      = 0.71
```

---

# 21. EXACT INFERENCE

Implement:

**Variable Elimination**

Use it as the primary exact inference method.

Demonstrate:

- Factor creation
- Evidence restriction
- Factor multiplication
- Variable elimination
- Normalization
- Posterior computation

Compare elimination-order heuristics:

1. Min-degree
2. Min-fill

Measure:

- Runtime
- Peak/intermediate factor size where possible
- Memory
- Induced width

---

# 22. APPROXIMATE INFERENCE

Implement an approximate inference method, preferably:

**MCMC / Gibbs Sampling**

Compare:

- 1,000 samples
- 5,000 samples
- 10,000 samples
- 50,000 samples

Compare the approximate posterior against the exact Variable Elimination result.

Example:

```text
VE:
P(High Risk) = 0.713

MCMC:
1K  → 0.69
5K  → 0.71
10K → 0.714
50K → 0.712
```

Measure:

- Posterior error
- Runtime
- Convergence behavior

This demonstrates the difference between exact and approximate probabilistic inference.

---

# 23. OPTIONAL NOISY-OR

If appropriate, use a noisy-OR model for behavioral indicators where multiple causes contribute to the same effect.

Example:

```text
After-hours Login ─────┐
USB Activity ──────────┤
Unusual File Access ───┼→ Data Movement Anomaly
External Email ────────┘
```

Do not force noisy-OR into the network if a learned CPT is more appropriate.

Treat it as an advanced modeling technique.

---

# 24. EXPLAINABILITY

The Bayesian Network must provide an explanation before the LLM is used.

For a detected high-risk case, identify:

1. Most influential evidence nodes.
2. Most probable threat hypothesis.
3. Posterior probability of each threat.
4. Risk posterior.
5. Evidence whose removal causes the largest posterior change.

---

# 25. EVIDENCE ABLATION / WHAT-IF ANALYSIS

For evidence e:

Calculate:

P(Risk | E)

and:

P(Risk | E - e)

Then:

ΔP(e) =
P(Risk | E) -
P(Risk | E - e)

Use this to rank evidence.

Example:

```text
Original Risk = 0.91

Remove USB activity:
Risk = 0.74
ΔP = 0.17

Remove after-hours login:
Risk = 0.82
ΔP = 0.09

Remove unusual file access:
Risk = 0.77
ΔP = 0.14
```

This provides a principled basis for the explanation.

Do not call this a formal causal effect unless the model and assumptions justify that interpretation.

Call it:

**posterior sensitivity / evidence contribution / what-if analysis.**

---

# 26. LLM GENERATION LAYER

The LLM is NOT the threat detector.

The LLM receives only the structured output from the Bayesian Network.

Example input:

```json
{
  "risk_probability": 0.91,
  "risk_level": "High",
  "threat_probabilities": {
    "unauthorized_access": 0.21,
    "data_exfiltration": 0.83,
    "insider_sabotage": 0.11
  },
  "key_evidence": [
    {
      "feature": "unusual_file_access",
      "posterior_contribution": 0.14
    },
    {
      "feature": "usb_activity",
      "posterior_contribution": 0.17
    },
    {
      "feature": "after_hours_login",
      "posterior_contribution": 0.09
    }
  ],
  "what_if_analysis": {
    "remove_usb": 0.74,
    "remove_file_access": 0.77,
    "remove_after_hours_login": 0.82
  }
}
```

---

# 27. LLM RESPONSIBILITY

The LLM should generate:

### Threat Summary

A concise explanation of the detected risk.

### Evidence Explanation

Explain which evidence contributed to the assessment.

### Probabilistic Interpretation

Explain the meaning of the probability.

### What-if Interpretation

Explain how the risk changes when important evidence is removed.

### Investigation Suggestions

Suggest reasonable investigation steps.

### Uncertainty

Explicitly state that the result is probabilistic and does not prove malicious intent.

---

# 28. LLM SAFETY / GROUNDING RULES

The LLM must:

1. Never invent events.
2. Never invent evidence.
3. Never modify probabilities.
4. Never claim certainty.
5. Never claim that an employee is definitively malicious.
6. Only use information supplied by the Bayesian Network.
7. Clearly distinguish observed evidence from inferred threat.
8. Clearly state uncertainty.
9. Avoid unsupported causal claims.
10. Produce a structured report.

---

# 29. EXAMPLE LLM OUTPUT

For example:

**Threat Assessment: High Risk**

The Bayesian Network estimates a 91% probability that the observed activity corresponds to high-risk insider behavior.

The most likely threat hypothesis is data exfiltration, with a posterior probability of 83%.

The strongest contributing evidence includes unusual USB activity, unusual file access, and after-hours authentication.

The evidence-ablation analysis shows that removing USB activity reduces the estimated risk from 91% to 74%, making it one of the strongest contributors to the current assessment.

This result represents a probabilistic assessment rather than proof of malicious intent. Further investigation should examine the files accessed, the associated removable-media activity, and any subsequent data-transfer behavior.

---

# 30. LLM EVALUATION

Do not evaluate the LLM as a threat classifier.

Evaluate it as an explanation generator.

Possible evaluation criteria:

- Factual consistency
- Evidence consistency
- Probability consistency
- Completeness
- Clarity
- Human readability
- Hallucination rate

A small human evaluation can be conducted where reviewers compare:

1. Raw probabilistic output.
2. LLM-generated explanation.

Ask reviewers to rate:

- Understandability
- Clarity
- Usefulness
- Faithfulness to model evidence

---

# 31. BASELINE MODELS

Use conventional machine-learning models only as comparison baselines.

Suggested baselines:

1. Naive Bayes
2. Logistic Regression
3. Random Forest
4. XGBoost

The Bayesian Network should remain the central model.

The purpose of the baselines is to answer:

> How does probabilistic graphical reasoning compare with conventional predictive models?

Do not let the baseline models dominate the project.

---

# 32. MAIN EXPERIMENTS

The final experimental plan should contain:

## Experiment 1 — BN Structure

Compare:

- Expert BN
- Data-learned BN
- Hybrid BN

Metrics:

- Predictive performance
- BIC / structure score where appropriate
- Complexity
- Interpretability

---

## Experiment 2 — Exact Inference

Compare:

- Variable Elimination + Min-Degree
- Variable Elimination + Min-Fill

Measure:

- Runtime
- Memory/intermediate factor size
- Induced width
- Posterior probability

---

## Experiment 3 — Approximate Inference

Compare:

- MCMC 1K
- MCMC 5K
- MCMC 10K
- MCMC 50K

Measure:

- Posterior error relative to VE
- Runtime
- Convergence

---

## Experiment 4 — Model Comparison

Compare:

- Naive Bayes
- Logistic Regression
- Random Forest
- XGBoost
- Bayesian Network

Metrics:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

---

## Experiment 5 — Probabilistic Quality

Evaluate:

- Brier Score
- Log Loss
- Calibration curve
- Expected Calibration Error

This experiment is important because the Bayesian Network produces probabilities rather than merely labels.

---

## Experiment 6 — Explainability

For selected high-risk cases:

- Show posterior probability.
- Show top evidence.
- Perform evidence ablation.
- Show posterior changes.
- Generate LLM explanation.

Compare raw BN output against LLM-generated explanation.

---

# 33. IMPORTANT EVALUATION ISSUE: CLASS IMBALANCE

CERT insider-threat data is highly imbalanced.

Therefore:

Do NOT rely only on accuracy.

Prioritize:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- False Positive Rate
- False Negative Rate
- Brier Score
- Calibration

Clearly report class distribution.

Do not use synthetic oversampling blindly for Bayesian Network training without discussing how it affects probability estimates.

---

# 34. FINAL OUTPUT OF THE SYSTEM

For each user/day, produce:

```text
User ID
Date

Risk Probability
Risk Level

Threat Probabilities

Top Evidence

Evidence Contribution

Posterior Sensitivity

BN Explanation

LLM-Generated Threat Report
```

Example:

```text
User: U342
Date: Day 143

Risk:
HIGH

P(High Risk) = 0.91

Threat:
Data Exfiltration = 0.83

Top Evidence:
1. USB Activity
2. Unusual File Access
3. After-Hours Login
4. High Data Movement

LLM Explanation:
[Generated grounded report]
```

---

# 35. SYSTEM ARCHITECTURE

The final architecture should be:

```text
             CERT R4.2 Dataset
                    │
                    ↓
             Raw Activity Logs
                    │
                    ↓
             Data Preprocessing
                    │
                    ↓
          User-Day Feature Extraction
                    │
                    ↓
             Evidence Variables
                    │
                    ↓
          Bayesian Network Construction
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
   Structure Learning      CPT Learning
          │                   │
          └─────────┬─────────┘
                    ↓
             Bayesian Network
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
    Exact Inference       Approx. Inference
    Variable Elimination      MCMC
          │                    │
          └──────────┬─────────┘
                     ↓
              Posterior Risk
                     │
                     ↓
             Evidence Analysis
                     │
              ┌──────┴──────┐
              ↓             ↓
        What-if Analysis   Threat Ranking
              │             │
              └──────┬──────┘
                     ↓
             Structured Result
                     │
                     ↓
              LLM Explanation
                     │
                     ↓
          Human-Readable Report
```

---

# 36. CORE RESEARCH QUESTION

The project should revolve around:

> **Can a Bayesian Network provide interpretable probabilistic assessment of insider-threat risk from heterogeneous employee behavioral evidence, while maintaining reliable probability estimates and enabling human-readable explanations?**

---

# 37. SUB-QUESTIONS

Investigate:

1. Can a Bayesian Network effectively represent dependencies among different insider-threat indicators?
2. Does a hybrid/domain-informed structure perform better or remain more interpretable than purely data-learned structures?
3. How does exact Variable Elimination compare with approximate MCMC inference?
4. How does inference accuracy change with MCMC sample count?
5. Are Bayesian Network probabilities well calibrated?
6. Which behavioral evidence contributes most strongly to the posterior risk?
7. Can posterior sensitivity provide useful explanations?
8. Can an LLM generate faithful natural-language explanations from the probabilistic output without introducing unsupported information?

---

# 38. EXPECTED CONTRIBUTION

The project contribution should NOT be:

"We used an LLM for cybersecurity."

Instead:

> We develop an interpretable Bayesian Network framework for probabilistic insider-threat risk assessment using heterogeneous employee activity. The framework compares learned and domain-informed probabilistic structures, evaluates exact and approximate inference, quantifies posterior sensitivity to evidence, and uses a grounded LLM to translate probabilistic outputs into human-readable threat assessments.

---

# 39. LIMITATIONS

Explicitly acknowledge:

- CERT is synthetic data.
- Behavioral indicators do not prove malicious intent.
- Bayesian Network dependencies depend on modeling assumptions.
- Some edges may represent probabilistic dependency rather than causal relationships.
- LLM explanations are only as reliable as the structured evidence supplied to them.
- Dataset imbalance can affect evaluation.
- Thresholding continuous behavioral features loses some information.

---

# 40. FUTURE WORK

Mention only as future work:

- Dynamic Bayesian Networks
- Temporal models
- Real-world enterprise datasets
- RAG with cybersecurity knowledge bases
- Decision networks
- Real-time streaming inference
- Online Bayesian updating
- Larger-scale deployment

Do not implement these unless the core system is already complete.

---

# 41. FINAL TECHNOLOGY STACK

Use a practical stack such as:

### Python

For the complete probabilistic pipeline.

### pandas / NumPy

Data processing.

### pgmpy or an equivalent Bayesian Network library

For:

- Bayesian Network construction
- Structure learning
- Parameter learning
- Variable Elimination
- MCMC

### scikit-learn

For baseline models and evaluation.

### Matplotlib / Seaborn

For evaluation plots.

### LLM API

For the final explanation-generation layer.

A simple web dashboard can optionally be implemented using Streamlit.

---

# 42. FINAL PROJECT SCOPE

The project must remain achievable for a student team.

MANDATORY:

- CERT R4.2
- Data preprocessing
- User-day feature construction
- Bayesian Network
- Graph design
- CPT learning
- Variable Elimination
- MCMC
- Evidence-based explanation
- Model evaluation
- LLM explanation layer

OPTIONAL:

- Structure-learning comparison
- Noisy-OR
- Streamlit dashboard
- Human evaluation of explanations

FUTURE WORK:

- DBN
- HMM
- RAG
- Real-time streaming
- Decision networks

Do NOT add HMM to the implementation.

---

# 43. FINAL PPT CONCEPTUAL FLOW

The project presentation should be organized approximately as:

1. Problem Statement
2. Motivation
3. Insider Threat Concept
4. Dataset — CERT R4.2
5. Dataset Characteristics
6. Feature Extraction
7. Bayesian Network Concept
8. Proposed BN Graph
9. Graph Construction Methodology
10. CPT Learning
11. Bayesian Inference
12. Exact Inference — Variable Elimination
13. Approximate Inference — MCMC
14. Evidence Sensitivity / Explainability
15. LLM Generation Layer
16. Experimental Design
17. Baselines
18. Evaluation Metrics
19. Expected Results
20. Limitations
21. Future Work

---

# 44. MOST IMPORTANT MODELING PRINCIPLE

The Bayesian Network must remain the **decision-making/probabilistic reasoning engine**.

The LLM must remain the **communication/explanation engine**.

Therefore:

```text
CERT
 ↓
Evidence
 ↓
Bayesian Network
 ↓
Probability
 ↓
Evidence Contribution
 ↓
LLM
 ↓
Explanation
```

NOT:

```text
CERT
 ↓
LLM
 ↓
Threat Detection
```

The latter would undermine the Probabilistic Reasoning objective.

---

# 45. FINAL ONE-SENTENCE DESCRIPTION

Use this as the project's concise description:

> **A Bayesian Network-based probabilistic framework for assessing insider-threat risk from heterogeneous employee behavior, using exact and approximate inference for uncertainty-aware risk estimation and a grounded LLM layer for human-readable explanation of the probabilistic assessment.**

When developing the project, prioritize mathematical correctness, reproducibility, leakage-free evaluation, probabilistic calibration, and interpretability over adding unnecessary AI components.