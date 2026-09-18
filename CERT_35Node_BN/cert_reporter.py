"""
cert_reporter.py
================
CERT Insider Threat Assessment Reporter.

Given a user-day evidence dict, this module:
  1. Runs VE inference to compute posterior probabilities for all BN nodes.
  2. Performs evidence ablation (what-if analysis) to rank evidence contribution.
  3. Assembles a structured JSON-serialisable report dict.
  4. Formats a human-readable console report.
  5. Builds a grounded LLM prompt and (optionally) calls an LLM API.

LLM Integration
---------------
Set OPENAI_API_KEY or GEMINI_API_KEY environment variables to enable.
If neither key is present, a grounded template-based explanation is used.
"""

import os
import json
from typing import Optional
from cert_bn_builder import RISK_LABELS, RISK_STATES, EVIDENCE_NODES, BEHAVIORAL_NODES, THREAT_NODES, RISK_NODE


# ---------------------------------------------------------------------------
# Evidence node descriptions (for human-readable output)
# ---------------------------------------------------------------------------
EVIDENCE_DESCRIPTIONS = {
    "AfterHoursLogin":       "After-hours login activity (before 09:00 or after 17:00)",
    "UnusualLoginFreq":      "Unusually high number of logon events",
    "DeviceConnectActivity": "Unusual removable device (USB) connections",
    "FileAccessCount":       "High number of file access events",
    "FileCopyToRemovable":   "Files copied to removable media",
    "ExternalEmail":         "Emails sent to external (non-organisation) addresses",
    "UnusualEmailVolume":    "Unusually high total email volume",
    "UnusualWebActivity":    "High web/HTTP access activity",
    "HighDataMovement":      "Large total outbound email data volume",
    "HighFileCopyActivity":  "Very high rate of file copying to removable media",
}

THREAT_DESCRIPTIONS = {
    "DataExfiltration":  "Data Exfiltration (data theft via USB/email/cloud)",
    "UnauthorizedAccess":"Unauthorized Access (credential misuse / masquerade)",
}


class CERTReporter:
    """
    Generates insider-threat risk reports from Bayesian Network posteriors.

    Parameters
    ----------
    ve_engine  : VariableEliminationEngine instance (fitted and built).
    mcmc_engine: MCMCEngine instance (optional, for cross-check).
    """

    def __init__(self, ve_engine, mcmc_engine=None):
        self.ve   = ve_engine
        self.mcmc = mcmc_engine

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------
    def _query_all(self, evidence: dict) -> dict:
        """Run VE for all query nodes given the evidence dict."""
        results = {}

        # Risk Level (ternary)
        risk_post, risk_stats = self.ve.query(RISK_NODE, evidence=evidence)
        results[RISK_NODE] = {"posterior": risk_post, "stats": risk_stats}

        # Threat hypotheses (binary)
        for t in THREAT_NODES:
            post, stats = self.ve.query(t, evidence=evidence)
            results[t] = {"posterior": post, "stats": stats}

        # Behavioral indicators (binary)
        for b in BEHAVIORAL_NODES:
            post, stats = self.ve.query(b, evidence=evidence)
            results[b] = {"posterior": post, "stats": stats}

        return results

    # ------------------------------------------------------------------
    # Evidence ablation / what-if analysis
    # ------------------------------------------------------------------
    def _evidence_ablation(
        self, evidence: dict, baseline_high_prob: float
    ) -> list:
        """
        For each active evidence node, compute:
          Delta_P(High Risk) = P(High | E) - P(High | E minus {e})

        Returns a list of (node, delta) sorted by |delta| descending.
        """
        active = [k for k, v in evidence.items() if v == 1]
        ablation = []
        for node in active:
            reduced = {k: v for k, v in evidence.items() if k != node}
            post_reduced, _ = self.ve.query(RISK_NODE, evidence=reduced)
            p_high_reduced  = post_reduced.get(2, 0.0)
            delta = baseline_high_prob - p_high_reduced
            ablation.append((node, delta, p_high_reduced))

        ablation.sort(key=lambda x: abs(x[1]), reverse=True)
        return ablation

    # ------------------------------------------------------------------
    # Main report generation
    # ------------------------------------------------------------------
    def generate_report(
        self,
        evidence: dict,
        user:     str = "Unknown",
        day:      str = "Unknown",
    ) -> dict:
        """
        Run complete BN inference and produce a structured report.

        Parameters
        ----------
        evidence : Dict {evidence_node: 0 or 1}  (from CERTPreprocessor.transform_row)
        user     : User identifier (for display purposes).
        day      : Date string (for display purposes).

        Returns
        -------
        Structured dict suitable for JSON serialization and LLM prompting.
        """
        # 1. Inference
        inference_results = self._query_all(evidence)
        risk_post  = inference_results[RISK_NODE]["posterior"]
        risk_stats = inference_results[RISK_NODE]["stats"]

        p_low    = risk_post.get(0, 0.0)
        p_med    = risk_post.get(1, 0.0)
        p_high   = risk_post.get(2, 0.0)
        risk_level = RISK_LABELS[max(risk_post, key=risk_post.get)]

        # 2. Threat posteriors
        threat_probs = {}
        for t in THREAT_NODES:
            tp = inference_results[t]["posterior"]
            threat_probs[t] = tp.get(1, 0.0)

        # 3. Behavioral indicator posteriors
        behavioral_probs = {}
        for b in BEHAVIORAL_NODES:
            bp = inference_results[b]["posterior"]
            behavioral_probs[b] = bp.get(1, 0.0)

        # 4. Active evidence (triggered nodes)
        active_evidence = [k for k, v in evidence.items() if v == 1]

        # 5. Evidence ablation (only meaningful if high risk)
        ablation = []
        if p_high > 0.3 and active_evidence:
            ablation = self._evidence_ablation(evidence, p_high)

        # 6. Top contributing threat
        top_threat = max(threat_probs, key=threat_probs.get) if threat_probs else "Unknown"

        report = {
            "user": user,
            "day":  day,
            "risk": {
                "posterior": {"Low": p_low, "Medium": p_med, "High": p_high},
                "level":     risk_level,
                "p_high":    p_high,
            },
            "threats": {
                t: {
                    "probability": threat_probs[t],
                    "description": THREAT_DESCRIPTIONS.get(t, t),
                }
                for t in THREAT_NODES
            },
            "top_threat":  top_threat,
            "behavioral_indicators": behavioral_probs,
            "active_evidence": [
                {
                    "node":        node,
                    "description": EVIDENCE_DESCRIPTIONS.get(node, node),
                }
                for node in active_evidence
            ],
            "evidence_ablation": [
                {
                    "node":           node,
                    "delta_p_high":   round(delta, 4),
                    "p_high_without": round(p_without, 4),
                    "description":    EVIDENCE_DESCRIPTIONS.get(node, node),
                }
                for node, delta, p_without in ablation[:5]   # top-5
            ],
            "inference_stats": {
                "strategy":          risk_stats.get("strategy", "min-degree"),
                "induced_width":     risk_stats.get("induced_width", -1),
                "max_factor_size":   risk_stats.get("max_factor_size", -1),
                "runtime_ms":        round(risk_stats.get("runtime_sec", 0) * 1000, 2),
            },
        }

        return report

    # ------------------------------------------------------------------
    # Console text report
    # ------------------------------------------------------------------
    def format_text_report(self, report: dict) -> str:
        """Format a structured report dict as a human-readable text report."""
        lines = [
            "=" * 70,
            "   INSIDER THREAT PROBABILISTIC RISK ASSESSMENT",
            "=" * 70,
            f"  User : {report['user']}",
            f"  Day  : {report['day']}",
            "",
            "── RISK ASSESSMENT ──────────────────────────────────────────────",
            f"  Overall Risk Level : {report['risk']['level'].upper()}",
            f"  P(Low Risk)        = {report['risk']['posterior']['Low']:.4f}",
            f"  P(Medium Risk)     = {report['risk']['posterior']['Medium']:.4f}",
            f"  P(High Risk)       = {report['risk']['posterior']['High']:.4f}",
            "",
            "── THREAT HYPOTHESIS POSTERIORS ─────────────────────────────────",
        ]
        for t, info in sorted(report["threats"].items(),
                               key=lambda x: x[1]["probability"], reverse=True):
            bar = "█" * int(info["probability"] * 30)
            lines.append(f"  {info['description']}")
            lines.append(f"    P = {info['probability']:.4f}  {bar}")
        lines.append("")

        lines.append("── BEHAVIORAL INDICATORS ────────────────────────────────────────")
        for beh, prob in sorted(
            report["behavioral_indicators"].items(), key=lambda x: x[1], reverse=True
        ):
            flag = " ⚠" if prob > 0.50 else ""
            lines.append(f"  {beh:<28}  P = {prob:.4f}{flag}")
        lines.append("")

        lines.append("── ACTIVE EVIDENCE ──────────────────────────────────────────────")
        if report["active_evidence"]:
            for ev in report["active_evidence"]:
                lines.append(f"  [!] {ev['description']}")
        else:
            lines.append("  [i] No anomalous evidence detected (Normal behaviour).")
        lines.append("")

        if report["evidence_ablation"]:
            lines.append("── EVIDENCE CONTRIBUTION (Posterior Sensitivity) ────────────────")
            lines.append("  Removing each evidence item causes the following risk drop:")
            for abl in report["evidence_ablation"]:
                direction = "↓" if abl["delta_p_high"] >= 0 else "↑"
                lines.append(
                    f"  {abl['description'][:45]:<45}  ΔP = {direction}{abs(abl['delta_p_high']):.4f}"
                    f"  (new P(High) = {abl['p_high_without']:.4f})"
                )
            lines.append("")

        lines.append("── INFERENCE STATS ──────────────────────────────────────────────")
        st = report["inference_stats"]
        lines.append(f"  Strategy     : {st['strategy']}")
        lines.append(f"  Induced Width: {st['induced_width']}")
        lines.append(f"  Runtime      : {st['runtime_ms']} ms")
        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM prompt builder
    # ------------------------------------------------------------------
    def build_llm_prompt(self, report: dict) -> str:
        """
        Construct a grounded, structured prompt to send to the LLM.

        The prompt provides ONLY the structured BN output — the LLM
        must not invent additional evidence or change probabilities.
        """
        abl_text = ""
        if report["evidence_ablation"]:
            abl_text = "\n".join(
                f"  - Removing '{abl['description']}' → P(High Risk) changes by {abl['delta_p_high']:+.3f}"
                for abl in report["evidence_ablation"]
            )

        ev_text = "\n".join(
            f"  - {ev['description']}"
            for ev in report["active_evidence"]
        ) or "  (none)"

        threat_text = "\n".join(
            f"  - {info['description']}: {info['probability']:.3f}"
            for info in sorted(report["threats"].values(),
                               key=lambda x: x["probability"], reverse=True)
        )

        prompt = f"""You are a cybersecurity analyst assistant. You must write a structured threat assessment report using ONLY the probabilistic information provided below. Do NOT invent any additional events or evidence. Do NOT claim certainty. All statements must be grounded in the probabilities listed.

== BAYESIAN NETWORK OUTPUT ==

User: {report['user']}
Date: {report['day']}

Risk Level: {report['risk']['level']}
  P(Low Risk)    = {report['risk']['posterior']['Low']:.3f}
  P(Medium Risk) = {report['risk']['posterior']['Medium']:.3f}
  P(High Risk)   = {report['risk']['posterior']['High']:.3f}

Threat Hypothesis Posteriors:
{threat_text}

Most Likely Threat Type: {THREAT_DESCRIPTIONS.get(report['top_threat'], report['top_threat'])}

Active Observed Evidence (behaviours detected above normal threshold):
{ev_text}

Evidence Contribution Analysis (posterior sensitivity):
{abl_text if abl_text else '  (not computed — risk level below threshold)'}

== YOUR REPORT MUST CONTAIN THE FOLLOWING SECTIONS ==

1. **Threat Summary** — Concise 2–3 sentence summary of the assessed risk.
2. **Evidence Explanation** — Which observed evidence contributed to the assessment and how.
3. **Probabilistic Interpretation** — Explain what the probability values mean (uncertainty, not certainty).
4. **What-If Analysis** — Explain how the risk changes when the strongest evidence is removed (if available).
5. **Investigation Suggestions** — 2–3 actionable recommendations for a security team.
6. **Uncertainty Statement** — Clearly state this is a probabilistic assessment, not proof of malicious intent.

Remember: You are ONLY allowed to reference the information above. Do not add external knowledge about specific attack tools, real vulnerabilities, or specific employee details not mentioned.
"""
        return prompt

    # ------------------------------------------------------------------
    # LLM call (optional)
    # ------------------------------------------------------------------
    def call_llm(self, prompt: str, max_tokens: int = 600) -> str:
        """
        Optionally call an LLM API to generate a natural-language explanation.

        Tries in order:
          1. OpenAI (OPENAI_API_KEY env var)
          2. Google Gemini (GEMINI_API_KEY env var)
          3. Returns a template-based fallback explanation.
        """
        # ---- OpenAI ----
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import openai
                client   = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"  [LLM] OpenAI call failed: {e}")

        # ---- Gemini ----
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if not gemini_key:
            for env_path in [".env", os.path.join("..", ".env"), os.path.join(os.path.dirname(__file__), ".env")]:
                if os.path.exists(env_path):
                    try:
                        with open(env_path) as f:
                            for line in f:
                                if line.strip().startswith("GEMINI_API_KEY="):
                                    gemini_key = line.strip().split("=", 1)[1].strip().strip("\"'")
                                    os.environ["GEMINI_API_KEY"] = gemini_key
                                    break
                    except Exception:
                        pass
                if gemini_key:
                    break
        if gemini_key:
            import urllib.request
            import json
            for model_name in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                    payload = json.dumps({
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "maxOutputTokens": max_tokens,
                            "temperature": 0.2
                        }
                    }).encode("utf-8")
                    req = urllib.request.Request(
                        url,
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": gemini_key
                        }
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        res = json.loads(resp.read().decode("utf-8"))
                        text = res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text:
                            return text.strip()
                except Exception as e:
                    continue

        # ---- Fallback: structured template ----
        return self._template_explanation_from_prompt(prompt)

    # ------------------------------------------------------------------
    def _template_explanation_from_prompt(self, prompt: str) -> str:
        """
        Generate a grounded template-based explanation when no LLM key is set.
        Extracts key values directly from the prompt text.
        """
        # We pass in the report directly via generate_and_explain()
        return (
            "[LLM Explanation — Template Mode (no API key set)]\n"
            "The Bayesian Network has estimated a probabilistic insider threat risk score "
            "based on the observed behavioural evidence listed in the report above. "
            "The risk level and threat hypotheses are derived from the posterior distributions "
            "computed via exact Variable Elimination. This is a probabilistic assessment; "
            "it does not constitute proof of malicious intent. "
            "A security team should investigate the flagged evidence items to determine "
            "whether the observed behaviour has a legitimate business explanation."
        )

    # ------------------------------------------------------------------
    def generate_and_explain(
        self,
        evidence:    dict,
        user:        str = "Unknown",
        day:         str = "Unknown",
        use_llm:     bool = True,
        max_tokens:  int  = 600,
    ) -> tuple[dict, str]:
        """
        Full pipeline: inference → report → LLM explanation.

        Returns (report_dict, llm_text).
        """
        report  = self.generate_report(evidence, user=user, day=day)
        prompt  = self.build_llm_prompt(report)
        llm_out = self.call_llm(prompt, max_tokens=max_tokens) if use_llm else \
                  self._template_explanation_from_prompt(prompt)
        report["llm_explanation"] = llm_out
        return report, llm_out
