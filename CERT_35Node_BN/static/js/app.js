/* ==========================================================================
   Antigravity Security - Client-Side App Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  let networkGraph = null;
  let nodesDataSet = null;
  let edgesDataSet = null;
  let currentSchema = null;
  let currentEvidence = {};

  // ------------------------------------------------------------------------
  // Initialize App
  // ------------------------------------------------------------------------
  initTabSwitching();
  fetchNetworkSchema();
  fetchSamples();
  setupAccordion();
  setupEvents();

  // ------------------------------------------------------------------------
  // Tab Switching Logic
  // ------------------------------------------------------------------------
  function initTabSwitching() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        const targetTab = tab.getAttribute("data-tab");
        document.querySelectorAll(".canvas-tab").forEach(c => c.classList.remove("active"));
        document.getElementById(targetTab).classList.add("active");

        if (targetTab === "tab-network" && networkGraph) {
          networkGraph.fit();
        }
      });
    });
  }

  // ------------------------------------------------------------------------
  // Accordion Setup
  // ------------------------------------------------------------------------
  function setupAccordion() {
    const btn = document.getElementById("evidence-accordion-btn");
    const grid = document.getElementById("evidence-checkbox-grid");
    btn.addEventListener("click", () => {
      if (grid.style.display === "none") {
        grid.style.display = "grid";
      } else {
        grid.style.display = grid.style.display === "grid" ? "none" : "grid";
      }
    });
  }

  // ------------------------------------------------------------------------
  // Fetch Network Schema & Build Vis.js Graph
  // ------------------------------------------------------------------------
  async function fetchNetworkSchema() {
    try {
      const res = await fetch("/api/schema");
      currentSchema = await res.json();
      renderEvidenceCheckboxes(currentSchema.nodes.filter(n => n.layer === 1));
      buildVisGraph(currentSchema);
      renderCPTTable(currentSchema);
    } catch (err) {
      console.error("Failed to load network schema:", err);
    }
  }

  // Render Checkboxes for 21 Evidence Nodes
  function renderEvidenceCheckboxes(evidenceNodes) {
    const container = document.getElementById("evidence-checkbox-grid");
    container.innerHTML = "";

    evidenceNodes.forEach(node => {
      const label = document.createElement("label");
      label.className = "ev-checkbox-label";
      label.innerHTML = `
        <input type="checkbox" value="${node.id}" class="ev-checkbox">
        <span>${node.id}</span>
      `;
      container.appendChild(label);
    });

    // Add change listeners
    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      cb.addEventListener("change", updateActiveEvidenceFromUI);
    });
  }

  function updateActiveEvidenceFromUI() {
    currentEvidence = {};
    let count = 0;
    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      if (cb.checked) {
        currentEvidence[cb.value] = 1;
        count++;
      }
    });
    document.getElementById("active-ev-count").innerText = count;
  }

  // ------------------------------------------------------------------------
  // Build Vis.js Hierarchical Graph
  // ------------------------------------------------------------------------
  function buildVisGraph(schema) {
    const container = document.getElementById("network-container");

    const layerColors = {
      1: { background: "#1e293b", border: "#38bdf8" },
      2: { background: "#1e293b", border: "#fbbf24" },
      3: { background: "#1e293b", border: "#a855f7" },
      4: { background: "#1e293b", border: "#f43f5e" }
    };

    const visNodes = schema.nodes.map(n => ({
      id: n.id,
      label: n.id,
      level: n.layer,
      shape: n.layer === 4 ? "diamond" : "box",
      margin: 10,
      font: { color: "#f8fafc", face: "Inter", size: 12 },
      color: layerColors[n.layer],
      borderWidth: 2,
      shadow: true
    }));

    const visEdges = schema.edges.map(e => ({
      from: e.from,
      to: e.to,
      arrows: "to",
      color: { color: "rgba(255, 255, 255, 0.15)", highlight: "#06b6d4" },
      width: 1.5
    }));

    nodesDataSet = new vis.DataSet(visNodes);
    edgesDataSet = new vis.DataSet(visEdges);

    const data = { nodes: nodesDataSet, edges: edgesDataSet };
    const options = {
      layout: {
        hierarchical: {
          direction: "LR", // Left-to-Right layout
          sortMethod: "directed",
          levelSeparation: 220,
          nodeSpacing: 60
        }
      },
      physics: {
        hierarchicalRepulsion: {
          centralGravity: 0.0,
          springLength: 100,
          springConstant: 0.01,
          nodeDistance: 80
        }
      },
      interaction: { hover: true, tooltipDelay: 100 }
    };

    networkGraph = new vis.Network(container, data, options);
  }

  // ------------------------------------------------------------------------
  // Fetch Sample User-Days
  // ------------------------------------------------------------------------
  async function fetchSamples() {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      const select = document.getElementById("sample-select");
      
      data.samples.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = s.label;
        opt.dataset.evidence = JSON.stringify(s.evidence);
        select.appendChild(opt);
      });

      select.addEventListener("change", (e) => {
        const selectedOpt = select.options[select.selectedIndex];
        if (selectedOpt && selectedOpt.dataset.evidence) {
          const ev = JSON.parse(selectedOpt.dataset.evidence);
          setEvidenceCheckboxes(ev);
          runAnalysis();
        }
      });
    } catch (err) {
      console.log("Samples load error:", err);
    }
  }

  function setEvidenceCheckboxes(evidenceObj) {
    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      cb.checked = (evidenceObj[cb.value] === 1);
    });
    updateActiveEvidenceFromUI();
  }

  // ------------------------------------------------------------------------
  // Setup Event Handlers
  // ------------------------------------------------------------------------
  function setupEvents() {
    document.getElementById("btn-run-analysis").addEventListener("click", runAnalysis);

    document.getElementById("btn-fit-graph").addEventListener("click", () => {
      if (networkGraph) networkGraph.fit();
    });

    document.getElementById("btn-physics-toggle").addEventListener("click", () => {
      if (networkGraph) {
        const options = { physics: { enabled: !networkGraph.physics.options.enabled } };
        networkGraph.setOptions(options);
      }
    });
  }

  // ------------------------------------------------------------------------
  // Main Analysis Runner
  // ------------------------------------------------------------------------
  async function runAnalysis() {
    updateActiveEvidenceFromUI();

    const sampleSelect = document.getElementById("sample-select");
    const selectedText = sampleSelect.options[sampleSelect.selectedIndex]?.textContent || "Interactive Query";

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user: selectedText.split(" ")[0] || "USER_QUERY",
          day: "2026-02-11",
          evidence: currentEvidence
        })
      });

      const res = await response.json();
      if (res.status === "success") {
        updateRiskBadge(res.risk_assessment);
        highlightNetworkPaths(res.active_threat_paths, res.active_evidence);
        renderLLMReport(res);
        renderMathTrace(res.math_trace);
      }
    } catch (err) {
      console.error("Analysis execution error:", err);
    }
  }

  // ------------------------------------------------------------------------
  // Update Top Risk Badge
  // ------------------------------------------------------------------------
  function updateRiskBadge(riskAss) {
    const badge = document.getElementById("risk-badge");
    const text = document.getElementById("risk-level-text");
    const level = riskAss.level;
    const pHigh = (riskAss.p_high * 100).toFixed(1);

    badge.className = `risk-badge risk-${level.toLowerCase()}`;
    text.innerText = `${level.toUpperCase()} RISK (${pHigh}%)`;
  }

  // ------------------------------------------------------------------------
  // Highlight Active DAG Threat Trace Paths
  // ------------------------------------------------------------------------
  function highlightNetworkPaths(pathData, activeEvidence) {
    if (!nodesDataSet || !pathData) return;

    const activations = pathData.node_activations || {};

    nodesDataSet.forEach(node => {
      const act = activations[node.id] || 0.05;
      const isEvActive = activeEvidence.includes(node.id);

      let borderColor = "#38bdf8";
      if (node.level === 2) borderColor = "#fbbf24";
      if (node.level === 3) borderColor = "#a855f7";
      if (node.level === 4) borderColor = "#f43f5e";

      let bg = "#1e293b";
      if (act > 0.4) bg = "#334155";
      if (isEvActive) {
        borderColor = "#06b6d4";
        bg = "#0891b2";
      }

      nodesDataSet.update({
        id: node.id,
        borderWidth: act > 0.3 ? 3 : 1,
        color: { background: bg, border: borderColor },
        shadow: act > 0.4
      });
    });
  }

  // ------------------------------------------------------------------------
  // Render LLM Security Report in IDE Chat Window
  // ------------------------------------------------------------------------
  function renderLLMReport(data) {
    const container = document.getElementById("report-content-body");
    const risk = data.risk_assessment;
    const decision = data.decision_network;
    const threats = data.threat_posteriors;
    const ablation = data.evidence_ablation;

    let html = `
      <h4><i class="fa-solid fa-shield-virus"></i> Threat Assessment: ${risk.level.toUpperCase()} RISK</h4>
      <p><strong>P(High Risk)</strong> = ${(risk.p_high * 100).toFixed(1)}% | <strong>P(Low Risk)</strong> = ${(risk.posterior.Low * 100).toFixed(1)}%</p>

      <h4><i class="fa-solid fa-gavel"></i> Decision Network Action: ${decision.recommended_action}</h4>
      <p><em>${decision.action_description}</em></p>

      <h4><i class="fa-solid fa-skull-crossbones"></i> Threat Hypotheses Posteriors</h4>
      <ul>
    `;

    for (let [threat, prob] of Object.entries(threats)) {
      html += `<li><strong>${threat}</strong>: ${(prob * 100).toFixed(1)}%</li>`;
    }

    html += `</ul>`;

    if (ablation.length > 0) {
      html += `<h4><i class="fa-solid fa-filter"></i> Top Evidence Contributors (Risk Drop ΔP)</h4><ul>`;
      ablation.forEach(item => {
        html += `<li><strong>${item.node}</strong>: ΔP = ↓${(item.delta_p_high * 100).toFixed(1)}% (new P(High) = ${(item.p_high_without * 100).toFixed(1)}%)</li>`;
      });
      html += `</ul>`;
    }

    html += `<hr style="border-color: rgba(255,255,255,0.08); margin: 0.75rem 0;"><p style="font-size: 0.75rem; color: #94a3b8;">${data.llm_report}</p>`;

    container.innerHTML = html;
  }

  // ------------------------------------------------------------------------
  // Render Step-by-Step Mathematical Calculation Page (KaTeX)
  // ------------------------------------------------------------------------
  function renderMathTrace(mathTrace) {
    const container = document.getElementById("math-trace-container");
    if (!mathTrace) return;

    let html = ``;

    // Step 1: Evidence Vector
    html += `
      <div class="math-step-card">
        <h3><span class="step-num-badge">STEP 1</span> Evidence Discretization & Prior Distributions</h3>
        <p class="text-sm text-muted">Raw logs are mapped to binary evidence variables E_i. Prior probabilities P(E_i = 1) are learned from training baseline.</p>
        <div class="katex-render">
          $$\\mathbf{E} = \\left[ ${mathTrace.step1_evidence.map(e => e.value).join(", ")} \\right]$$
        </div>
      </div>
    `;

    // Step 2: Noisy-OR Calculations
    html += `
      <div class="math-step-card">
        <h3><span class="step-num-badge">STEP 2</span> Noisy-OR Intermediate Behavioral Evaluations</h3>
        <p class="text-sm text-muted">Intermediate nodes combine active evidence using the closed-form Noisy-OR parameterization: P(Y=1 | E) = 1 - (1 - λ) ∏ (1 - w_i).</p>
    `;

    mathTrace.step2_noisy_or.forEach(item => {
      html += `
        <div style="margin: 0.75rem 0;">
          <strong>${item.node}</strong> (Leak λ = ${item.leak}):
          <div class="katex-render">$$${item.latex}$$</div>
        </div>
      `;
    });
    html += `</div>`;

    // Step 3: Exact Variable Elimination Sequence
    const step3 = mathTrace.step3_variable_elimination;
    html += `
      <div class="math-step-card">
        <h3><span class="step-num-badge">STEP 3</span> Variable Elimination Exact Inference</h3>
        <p class="text-sm text-muted">Irrelevance ancestor pruning drops non-ancestor nodes. Elimination ordering strategy: <strong>${step3.strategy}</strong> (Induced Width = ${step3.induced_width}, Max Factor Size = ${step3.max_factor_size}, Runtime = ${step3.runtime_ms} ms).</p>
        <div class="katex-render">$$${step3.latex_normalization}$$</div>
      </div>
    `;

    // Step 4: Decision Network Expected Utility Algebra
    const step4 = mathTrace.step4_decision_utility;
    html += `
      <div class="math-step-card">
        <h3><span class="step-num-badge">STEP 4</span> Decision Network Expected Utility Maximization</h3>
        <p class="text-sm text-muted">Optimal action a* is selected by maximizing expected utility across Risk and Threat posteriors.</p>
        <div class="katex-render">$$${step4.latex_optimal}$$</div>
    `;

    step4.items.forEach(item => {
      html += `
        <div style="margin: 0.5rem 0;">
          <span style="color: ${item.is_selected ? '#06b6d4' : '#94a3b8'}; font-weight: ${item.is_selected ? '700' : '400'}">
            ${item.action} ${item.is_selected ? '★ (Selected)' : ''}
          </span>
          <div class="katex-render">$$${item.latex}$$</div>
        </div>
      `;
    });
    html += `</div>`;

    container.innerHTML = html;

    // Render KaTeX expressions
    if (window.renderMathInElement) {
      renderMathInElement(container, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false }
        ]
      });
    }
  }

  // ------------------------------------------------------------------------
  // Render CPT Specifications Table
  // ------------------------------------------------------------------------
  function renderCPTTable(schema) {
    const container = document.getElementById("cpt-container");
    let html = `
      <div class="math-step-card">
        <h3>35-Node Network Structure Summary</h3>
        <p>Total Nodes: ${schema.total_nodes} | Total Edges: ${schema.total_edges}</p>
        <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.825rem;">
          <thead>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; color: var(--accent-cyan);">
              <th style="padding: 0.5rem;">Node Name</th>
              <th style="padding: 0.5rem;">Layer</th>
              <th style="padding: 0.5rem;">Parent Nodes</th>
              <th style="padding: 0.5rem;">Parameter Type</th>
            </tr>
          </thead>
          <tbody>
    `;

    schema.nodes.forEach(n => {
      html += `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
          <td style="padding: 0.4rem; font-weight: 600;">${n.id}</td>
          <td style="padding: 0.4rem;">Layer ${n.layer} (${n.layer_name})</td>
          <td style="padding: 0.4rem; color: #94a3b8;">${n.parents.length > 0 ? n.parents.join(", ") : "—"}</td>
          <td style="padding: 0.4rem; color: #38bdf8;">${n.layer === 1 ? 'Prior P(E)' : (n.layer === 4 ? 'Categorical CPT' : 'Noisy-OR')}</td>
        </tr>
      `;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;
  }
});
