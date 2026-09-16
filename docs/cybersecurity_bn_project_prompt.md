# Project Build Prompt: Probabilistic Cyberattack Risk Assessment System (Bayesian Networks)

## Objective

Build a cybersecurity investigation system that takes system/network/authentication logs as input, converts them into Boolean evidence variables, and runs probabilistic inference over a 25-30 node Bayesian Network to output:

1. Overall probability that an attack is occurring
2. Probability distribution over attack types (Brute Force, Account Takeover, Malware, Insider Threat, Data Exfiltration)
3. Risk level classification (Low/Medium/High/Critical)
4. Most likely underlying cause(s) via diagnostic reasoning
5. Key evidence that drove the conclusion (explainability)
6. Recommended security action (optional decision network extension)

This is NOT a machine-learning classifier. The core of the system is a Bayesian Network with exact inference (Variable Elimination) and approximate inference (MCMC), compared against each other. ML may be used only to estimate CPT parameters from historical data, never to replace the inference engine.

## Datasets (three sources, each backing a different branch of the network)

1. **RBA Dataset (Wiefling et al., "Login Data Set for Risk-Based Authentication")** — ~33M labeled login events with country, device, IP/ASN, browser, timestamp, attack label. Use this to estimate CPTs for the authentication / account-takeover branch: MultipleFailedLogins, UnusualLoginLocation, NewDevice, UnusualLoginTime, SuspiciousIP, CredentialCompromise, AccountTakeover.

2. **CERT Insider Threat Dataset (CMU SEI r4.2)** — synthetic-but-realistic per-user logon/device/HTTP/email/file logs with labeled insider-threat scenarios. Use this to estimate CPTs for the insider/exfiltration branch: InsiderThreat, PrivilegeEscalation, LargeFileTransfer, AbnormalSessionDuration, DataExfiltration.

3. **CICIDS2017** — labeled network flow data (DoS, DDoS, brute force, infiltration, web attack, botnet, port scan). Use this to estimate CPTs for the malware/network branch: PortScanning, SuspiciousProcess, UnusualDNSActivity, HighOutboundTraffic, MalwareInfection, LateralMovement.

Each dataset estimates CPTs only for the branch it naturally covers (compare feature distributions between labeled-attack rows and benign rows to derive conditional probabilities). Do not attempt to merge the three datasets into one unified table — they stay separate as CPD-estimation sources, then all learned CPTs are assembled into one combined network. For the live demo, hand-construct 10-15 synthetic full-session log examples (JSON/CSV) that populate all evidence nodes simultaneously, since no single real dataset naturally produces a row touching every node at once.

## Network Structure (25-30 nodes, DAG)

**A. Threat/Cause variables (root nodes, priors from data or literature):**
External Attacker, Phishing Exposure, Stolen Credentials, Malware Presence, Insider Threat, Vulnerable Service

**B. Observable log/evidence variables (Boolean, derived from raw logs via thresholding):**
Multiple Failed Logins, Unusual Login Location, New Device, Unusual Login Time, Suspicious IP, Port Scanning, Suspicious Process, Unusual DNS Activity, Privilege Escalation, High Outbound Traffic, Large File Transfer, Abnormal Session Duration

**C. Intermediate security states (hidden, inferred):**
Credential Compromise, Initial Access, Account Takeover, Malware Infection, Lateral Movement, Data Exfiltration

**D. Final attack hypotheses / risk variables (query targets):**
Brute Force Attack, Malware Attack, Insider Threat Attack, Account Takeover Attack, Data Exfiltration Attack, Overall Cybersecurity Risk

Structure edges by genuine causal/diagnostic relationships (see reference structure below), not full connectivity. Keep it a valid DAG.

Reference structure:
```
External Attacker → {Port Scanning, Phishing, Brute Force}
Port Scanning → Initial Access
Phishing → Stolen Credentials
Brute Force → Multiple Failed Logins
Stolen Credentials + Multiple Failed Logins → Credential Compromise
Credential Compromise → {New Device, Unusual Location, Account Takeover}
Account Takeover → Privilege Escalation → Lateral Movement → Data Exfiltration
Data Exfiltration ← {Large File Transfer, High Outbound Traffic}

Malware Presence → {Suspicious Process, Unusual DNS Activity} → Malware Infection → Lateral Movement → Data Exfiltration

Insider Threat → {Privilege Escalation, Abnormal Session Duration, Large File Transfer} → Data Exfiltration
```

## CPD / CPT Construction Method

- Root node priors: frequency of the labeled event in its source dataset (e.g., P(StolenCredentials=True) = fraction of RBA logins labeled as credential-based attacks).
- Non-root nodes with few parents: estimate directly as conditional frequency from the relevant dataset (e.g., P(CredentialCompromise=True | StolenCredentials=True) from RBA).
- Nodes with many parents (e.g., Account Takeover with 3 Boolean parents = 8 rows): use noisy-OR parameterization instead of a full CPT — estimate one leak probability (P(effect=True | all parents False), from benign/normal rows) plus one weight per parent (P(effect=True | only this parent True)), then combine via the noisy-OR formula: P(effect=True | active parents) = 1 - product(1 - weight_i) over active parents, adjusted by the leak term. This avoids hand-filling 2^k table rows and mirrors the QMR-DT approach.
- Explicitly document, per node, which dataset (or literature/expert justification) backs its CPT.

## Inference Requirements

1. **Exact inference — Sum-Product Variable Elimination**: implement factor construction from CPTs, factor multiplication, summing out hidden variables, normalization. Use ancestor-graph pruning first to drop variables irrelevant to the current query/evidence (Irrelevance Theorem). Implement and compare at least two elimination orderings (e.g., min-degree vs min-fill) and report resulting factor sizes / induced width.

2. **Approximate inference — MCMC (Gibbs sampling or Metropolis-Hastings)**: run at least 10,000 samples per query, report the estimated posterior, and compare directly against the exact VE result for the same query (probability value, runtime, memory, accuracy, scalability as sample count grows).

3. **Query/Evidence/Hidden variable bookkeeping**: every inference call must explicitly log which variables are Query, which are Evidence (from logs), and which are Hidden (marginalized out).

## Explainability & Output Format

For a given set of input logs, output:
```
Overall Attack Probability: <value>
Risk Level: <Low|Medium|High|Critical>   (thresholds: 0-0.25 Low, 0.25-0.50 Medium, 0.50-0.75 High, 0.75-1.00 Critical)

Attack Type Probabilities (ranked):
  Account Takeover     <value>
  Brute Force          <value>
  Malware Attack       <value>
  Insider Threat       <value>
  Data Exfiltration    <value>

Key Evidence:
  <list of the Boolean evidence variables that were True>

Most Likely Underlying Cause:
  <highest-posterior hidden/intermediate variable>

Recommended Action (if decision network included):
  <Monitor | Request MFA | Block IP | Lock Account | Isolate Machine | Start Incident Response>
```

## Experiments to Include

- Evidence sensitivity: show how P(AccountTakeover) changes as evidence is added incrementally (none → failed logins → + new device → + unusual location).
- Diagnostic ranking: given a fixed evidence set, rank all attack-type hypotheses by posterior probability.
- Counterfactual/what-if: show how the posterior shifts when one evidence variable is toggled or removed.
- Exact vs approximate comparison table: posterior value, runtime, memory, accuracy, sample count (MCMC only).
- Network complexity report: node count, edge count, max parents per node, induced width under chosen elimination ordering.

## Optional Decision Network Extension

Add a decision node ("recommended action") and utility node combining inferred risk level against fixed, justified costs (false alarm cost vs missed-attack cost), mapping: Low→Monitor, Medium→Request MFA, High→Block source, Critical→Isolate + Incident Response.

## Deliverables

1. 25-30 node Bayesian Network (DAG) with visualization
2. Documented CPTs/noisy-OR parameters with source dataset per node
3. Log preprocessing + evidence extraction module (raw log → Boolean evidence)
4. Variable Elimination inference engine (with ancestor pruning + elimination ordering comparison)
5. MCMC approximate inference engine
6. Exact vs approximate comparison report
7. Risk scoring + attack-type ranking + explainability report generator
8. (Optional) Decision network for recommended actions
9. Simple interface (CLI or basic web form) to input/upload logs and view the security report
10. 10-15 synthetic demo log sessions covering the full evidence set

## Concepts to explicitly demonstrate in code/comments/report

Uncertain knowledge representation, Bayesian Network syntax (DAG/CPT), prior vs posterior probability, joint distribution via chain rule, conditional independence, Markov blanket, compactness (O(n·2^k) vs O(2^n)), Bayes' rule, diagnostic vs causal reasoning, query/evidence/hidden variables, marginalization, normalization, exact inference (Variable Elimination), elimination ordering, irrelevance/ancestor reduction, approximate inference (MCMC), and optionally decision networks.
