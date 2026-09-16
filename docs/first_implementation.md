# Project Build Prompt: Probabilistic Cyberattack Risk Assessment System (Bayesian Networks, UNSW-NB15)

## Objective

Build a cybersecurity investigation system that takes network traffic flow records as input, converts them into Boolean evidence variables, and runs probabilistic inference over a 29-node Bayesian Network to output:

1. Overall probability that an attack is occurring
2. Probability distribution over attack types (Reconnaissance, DoS, Exploits, Generic, Fuzzers)
3. Risk level classification (Low/Medium/High/Critical)
4. Most likely underlying cause(s) via diagnostic reasoning
5. Key evidence that drove the conclusion (explainability)
6. Recommended security action (decision network mapping)

This is NOT a machine-learning classifier. The core of the system is a Bayesian Network with exact inference (Variable Elimination) and approximate inference (MCMC), compared against each other. ML/statistics is used only to estimate CPT parameters and empirical priors from the dataset, never to replace the inference engine.

## Rationale for Refactoring (Removal of Redundant Root Threat-Cause Layer)

The original model contained separate root-cause variables such as DoS_Activity and final hypothesis variables such as DoS_Attack, which represented essentially the same attack concept twice. This created redundant nodes and an ambiguous causal interpretation. The revised model uses a single attack-hypothesis node as the latent/query variable, analogous to the Burglary node in the classic Burglary–Alarm Bayesian Network. Observable flow features provide evidence for the attack hypothesis, while intermediate behavioral variables provide hierarchical explanation.

## Dataset

**UNSW-NB15** (single dataset, no other sources needed). Publicly available (UNSW Canberra Cyber / Kaggle mirrors). Contains labeled network flow records with:

- Flow/connection features: `dur`, `proto`, `service`, `state`, `sbytes`, `dbytes`, `sttl`, `dttl`, `sloss`, `dloss`, `sload`, `dload`, `spkts`, `dpkts`, `smeansz`, `dmeansz`, `sjit`, `djit`, `sintpkt`, `dintpkt`, `tcprtt`, `synack`, `ackdat`, `swin`, `dwin`
- Connection-pattern features: `is_sm_ips_ports`, `ct_state_ttl`, `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd`, `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, `ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`
- Labels: `attack_cat` (Normal, Reconnaissance, DoS, Exploits, Generic, Fuzzers, Backdoor, Analysis, Shellcode, Worms) and `label` (0/1 attack flag)

The five representative attack categories for the hypothesis layer are: **Reconnaissance, DoS, Exploits, Generic, Fuzzers**.

All CPTs in this project are estimated from this dataset using empirical frequencies with Laplace smoothing (+1 / +2).

## Node List (Exactly 29 Nodes, 4 Functional Layers)

### Layer 1: Observable Flow Evidence Variables (19 nodes)
Thresholded from raw flow features (e.g. 90th percentile of the value among Normal-labeled rows):
1. High Connection Count to Service (`ct_srv_dst_high`)
2. High Connection Count from Source (`ct_srv_src_high`)
3. Repeated Connections Same Dest Port (`ct_dst_sport_high`)
4. Repeated Connections Same Src Port (`ct_src_dport_high`)
5. Unusual Service/Port (`unusual_srv`)
6. Abnormal Connection State (`abnormal_state`)
7. Unusual Protocol (`unusual_proto`)
8. HTTP Method Anomaly (`http_mthd_anom`)
9. FTP Login/Command Activity (`ftp_cmd_high`)
10. High Source Packet Rate (`spkts_high`)
11. High Destination Packet Rate (`dpkts_high`)
12. Abnormal Flow Duration (`dur_anom`)
13. Abnormal TCP Handshake Timing (`tcp_rtt_anom`)
14. High Source Byte Volume (`sbytes_high`)
15. High Destination Byte Volume (`dbytes_high`)
16. High Jitter (`jit_high`)
17. Same Source IP-Port Reuse (`sm_ips_ports`)
18. Small Mean Packet Size (`small_pkt_sz`)
19. Unusual TTL Pattern (`ttl_anom`)

### Layer 2: Intermediate Security States (4 nodes)
Latent behavioral states inferred from observable evidence via Noisy-OR parameterization:
20. Network Scanning Behavior (`Net_Scanning_Beh`)
21. Service Exploitation (`Srv_Exploitation`)
22. Traffic Flooding Behavior (`Traffic_Flooding`)
23. Malicious Payload Delivery (`Malicious_Payload_Deliv`)

### Layer 3: Attack Hypotheses (5 nodes)
Root query hypotheses whose priors are learned directly from UNSW-NB15 attack frequencies:
24. Reconnaissance Attack (`Recon_Attack`)
25. Exploitation Attack (`Srv_Exploit_Attack`)
26. DoS Attack (`DoS_Attack`)
27. Generic Attack (`Generic_Attack`)
28. Fuzzing Attack (`Fuzzing_Attack`)

### Layer 4: Overall Risk Assessment (1 node)
29. Overall Cybersecurity Risk (`Overall_Risk`)

**Total = 19 + 4 + 5 + 1 = 29 Nodes.**

## Network Architecture & Directed Acyclic Graph (DAG)

```
        UNSW-NB15 raw flow
                ↓
          preprocessing
                ↓
    19 observable evidence nodes
                ↓
    4 intermediate behavioral nodes
                ↓
    5 attack hypothesis nodes
                ↓
          Overall_Risk
```

```
Layer 1 (Observable Evidence) → Layer 2 (Intermediate Security States)
  - {ct_srv_dst_high, ct_srv_src_high, ct_dst_sport_high, ct_src_dport_high, unusual_srv} → Net_Scanning_Beh
  - {abnormal_state, unusual_proto, http_mthd_anom, ftp_cmd_high} → Srv_Exploitation
  - {spkts_high, dpkts_high, dur_anom, tcp_rtt_anom} → Traffic_Flooding
  - {sbytes_high, dbytes_high, jit_high, sm_ips_ports, small_pkt_sz, ttl_anom} → Malicious_Payload_Deliv

Layer 2 (Intermediate Security States) → Layer 3 (Attack Hypotheses)
  - Net_Scanning_Beh → Recon_Attack
  - Srv_Exploitation → Srv_Exploit_Attack
  - Traffic_Flooding → DoS_Attack
  - Malicious_Payload_Deliv → Generic_Attack
  - Malicious_Payload_Deliv → Fuzzing_Attack

Layer 3 (Attack Hypotheses) → Layer 4 (Overall Risk)
  - {Recon_Attack, Srv_Exploit_Attack, DoS_Attack, Generic_Attack, Fuzzing_Attack} → Overall_Risk
```

## CPD / Parameter Learning

- **Attack Hypothesis Priors**: $P(\text{Attack} = \text{True}) = \frac{\text{Count}(\text{attack\_cat} == C)}{\text{Total rows}}$
- **Evidence Priors**: $P(E_i = \text{True}) = \frac{\text{Count}(E_i = \text{True}) + 1}{N + 2}$
- **Intermediate Noisy-OR Parameters**: $\theta_i = P(E_i = \text{True} \mid \text{attack\_cat} == C) = \frac{\text{Count}(E_i = \text{True}, C) + 1}{\text{Count}(C) + 2}$, with leak $\lambda$ estimated from normal traffic false-activation rates.
- **Hypothesis CPTs**: Conditional distributions over the intermediate parent states.
- **Overall Risk Noisy-OR**: Aggregates the 5 attack hypotheses into a final posterior risk score.

## Inference Engines

1. **Exact Variable Elimination**: Prunes irrelevant non-ancestors using the Irrelevance Theorem, conditions on observed evidence, sums out hidden variables using min-degree or min-fill elimination ordering heuristics, and normalizes factor tables.
2. **Approximate MCMC (Gibbs Sampling)**: Explores joint assignments by sampling each unobserved variable from its Markov blanket conditional distribution over 10,000+ iterations.
