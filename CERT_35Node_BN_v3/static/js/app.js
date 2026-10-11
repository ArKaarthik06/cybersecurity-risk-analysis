/* ==========================================================================
   CERT Threat Analyzer — Client-Side App Logic (v2)
   Fixes: LaTeX rendering, node highlighting, file upload parsing,
          LLM streaming via OpenRouter, UI toggle, new result panels
   ========================================================================== */

"use strict";

document.addEventListener("DOMContentLoaded", () => {

  /* ── State ─────────────────────────────────────────────────────────────── */
  let networkGraph   = null;
  let nodesDataSet   = null;
  let edgesDataSet   = null;
  let currentSchema  = null;
  let currentEvidence = {};
  let lastAnalysisResult = null;
  let currentUI      = "classic"; // "classic" | "neo"
  let llmAbortCtrl   = null;
  let posteriorMap   = {};   // node → posterior (for tooltip)

  /* ── Init ──────────────────────────────────────────────────────────────── */
  initTabSwitching();
  initUIToggle();
  initFileUpload();
  fetchNetworkSchema();
  fetchSamples();
  setupAccordion();
  setupEvents();

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     TAB SWITCHING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function initTabSwitching() {
    document.querySelectorAll(".nav-tab").forEach(tab => {
      tab.addEventListener("click", () => {
        const targetId = tab.getAttribute("data-tab");
        document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");
        document.querySelectorAll(".canvas-tab").forEach(c => c.classList.remove("active"));
        document.getElementById(targetId)?.classList.add("active");

        if (targetId === "tab-network" && networkGraph) {
          setTimeout(() => { networkGraph.redraw(); networkGraph.fit(); }, 80);
        }

        if (targetId === "tab-llm") {
          const badge = document.getElementById("llm-tab-badge");
          if (badge && badge.classList.contains("ready")) {
            badge.style.display = "none";
            badge.className = "nav-tab-badge";
          }
        }
      });
    });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     UI TOGGLE (Classic ↔ Neo)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function initUIToggle() {
    const btn   = document.getElementById("ui-toggle-btn");
    const icon  = document.getElementById("ui-toggle-icon");
    const label = document.getElementById("ui-toggle-label");
    if (!btn) return;

    btn.addEventListener("click", () => {
      if (currentUI === "classic") {
        currentUI = "neo";
        document.documentElement.setAttribute("data-ui", "neo");
        if (icon) icon.className  = "fa-solid fa-sun";
        if (label) label.textContent = "Classic UI";
        applyNeoGraphStyle();
      } else {
        currentUI = "classic";
        document.documentElement.setAttribute("data-ui", "classic");
        if (icon) icon.className  = "fa-solid fa-terminal";
        if (label) label.textContent = "Neo UI";
        applyClassicGraphStyle();
      }
    });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     ACCORDION
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function setupAccordion() {
    const btn  = document.getElementById("evidence-accordion-btn");
    const body = document.getElementById("evidence-checkbox-grid");
    const icon = document.getElementById("acc-icon");

    // Wrap contents in inner div for proper animation
    if (!body.querySelector(".accordion-body-inner")) {
      const inner = document.createElement("div");
      inner.className = "accordion-body-inner";
      while (body.firstChild) inner.appendChild(body.firstChild);
      body.appendChild(inner);
    }

    btn.addEventListener("click", () => {
      body.classList.toggle("open");
      icon.classList.toggle("open");
    });

    // Open by default
    body.classList.add("open");
    icon.classList.add("open");
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     NETWORK SCHEMA & GRAPH
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  async function fetchNetworkSchema() {
    try {
      const res = await fetch("/api/schema");
      currentSchema = await res.json();
      const evidenceNodes = currentSchema.nodes.filter(n => n.layer === 1);
      buildEvidenceCheckboxes(evidenceNodes);
      buildVisGraph(currentSchema);
      renderCPTTable(currentSchema);
    } catch (err) {
      console.error("Schema fetch failed:", err);
    }
  }

  /* Evidence Checkboxes ─────────────────────────────────────────────────── */
  function buildEvidenceCheckboxes(nodes) {
    const container = document.getElementById("evidence-checkbox-grid");
    container.innerHTML = '<div class="accordion-body-inner"></div>';
    const inner = container.querySelector(".accordion-body-inner");

    nodes.forEach(node => {
      const lbl = document.createElement("label");
      lbl.className = "ev-checkbox-label";
      lbl.innerHTML = `
        <input type="checkbox" value="${node.id}" class="ev-checkbox">
        <span>${node.id}</span>
      `;
      inner.appendChild(lbl);
    });

    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      cb.addEventListener("change", updateActiveEvidenceCount);
    });
  }

  function updateActiveEvidenceCount() {
    currentEvidence = {};
    let count = 0;
    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      if (cb.checked) {
        currentEvidence[cb.value] = 1;
        count++;
        cb.closest(".ev-checkbox-label").classList.add("active");
      } else {
        cb.closest(".ev-checkbox-label").classList.remove("active");
      }
    });
    document.getElementById("active-ev-count").textContent = count;
  }

  function setEvidenceCheckboxes(evidenceObj) {
    document.querySelectorAll(".ev-checkbox").forEach(cb => {
      cb.checked = (evidenceObj[cb.value] === 1);
    });
    updateActiveEvidenceCount();
  }

  /* Vis.js Graph ────────────────────────────────────────────────────────── */
  function getLayerColors(layer) {
    const colors = {
      1: { bg: "#e0f2fe", border: "#0284c7" },
      2: { bg: "#fef3c7", border: "#d97706" },
      3: { bg: "#f3e8ff", border: "#7e22ce" },
      4: { bg: "#fee2e2", border: "#dc2626" }
    };
    return colors[layer] || { bg: "#f1f5f9", border: "#64748b" };
  }

  function getNeoLayerColors(layer) {
    const colors = {
      1: { bg: "#e0f2fe", border: "#0284c7" },
      2: { bg: "#fef3c7", border: "#d97706" },
      3: { bg: "#f3e8ff", border: "#7e22ce" },
      4: { bg: "#fee2e2", border: "#dc2626" }
    };
    return colors[layer] || { bg: "#f1f5f9", border: "#64748b" };
  }

  function buildVisGraph(schema) {
    const container = document.getElementById("network-container");

    const visNodes = schema.nodes.map(n => {
      const c = getLayerColors(n.layer);
      return {
        id: n.id,
        label: n.id,
        level: n.layer,
        shape: n.layer === 4 ? "diamond" : n.layer === 3 ? "ellipse" : "box",
        margin: { top: 8, bottom: 8, left: 10, right: 10 },
        font: { color: "#0f172a", face: "Inter", size: 11, bold: n.layer >= 3 },
        color: { background: c.bg, border: c.border, highlight: { background: "#bae6fd", border: "#0284c7" } },
        borderWidth: 1.5,
        borderWidthSelected: 3,
        shadow: { enabled: n.layer >= 3, size: 6, color: "rgba(0,0,0,0.06)" }
      };
    });

    const visEdges = schema.edges.map(e => ({
      from: e.from,
      to: e.to,
      arrows: { to: { enabled: true, scaleFactor: 0.7 } },
      color: { color: "#cbd5e1", highlight: "#2563eb", hover: "#2563eb" },
      width: 1.2,
      smooth: { type: "cubicBezier", forceDirection: "horizontal", roundness: 0.5 }
    }));

    nodesDataSet = new vis.DataSet(visNodes);
    edgesDataSet = new vis.DataSet(visEdges);

    const options = {
      layout: {
        hierarchical: {
          direction: "LR",
          sortMethod: "directed",
          levelSeparation: 210,
          nodeSpacing: 52,
          treeSpacing: 80
        }
      },
      physics: {
        enabled: false,
        hierarchicalRepulsion: { centralGravity: 0.0, springLength: 80, nodeDistance: 70 }
      },
      interaction: { hover: true, tooltipDelay: 150, multiselect: false, navigationButtons: false }
    };

    networkGraph = new vis.Network(container, { nodes: nodesDataSet, edges: edgesDataSet }, options);

    networkGraph.once("afterDrawing", () => { networkGraph.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" } }); });

    // Node click → tooltip
    networkGraph.on("click", params => {
      if (params.nodes.length > 0) showNodeTooltip(params.nodes[0]);
      else hideNodeTooltip();
    });

    networkGraph.on("hoverNode", params => { showNodeTooltip(params.node); });
    networkGraph.on("blurNode",  () => { /* keep tooltip until click elsewhere */ });
  }

  function applyNeoGraphStyle() {
    if (!nodesDataSet || !currentSchema) return;
    currentSchema.nodes.forEach(n => {
      const c = getNeoLayerColors(n.layer);
      nodesDataSet.update({
        id: n.id,
        color: { background: c.bg, border: c.border, highlight: { background: "#001a00", border: "#00ff41" } },
        font: { color: "#00ff41", size: 11 },
        shadow: { enabled: n.layer >= 3, color: c.border + "88" }
      });
    });
    edgesDataSet.forEach(e => {
      edgesDataSet.update({ id: e.id, color: { color: "rgba(0,255,65,0.08)", highlight: "#00ff41" } });
    });
  }

  function applyClassicGraphStyle() {
    if (!nodesDataSet || !currentSchema) return;
    currentSchema.nodes.forEach(n => {
      const c = getLayerColors(n.layer);
      nodesDataSet.update({
        id: n.id,
        color: { background: c.bg, border: c.border, highlight: { background: "#bae6fd", border: "#0284c7" } },
        font: { color: "#0f172a", size: 11 }
      });
    });
    edgesDataSet.forEach(e => {
      edgesDataSet.update({ id: e.id, color: { color: "#cbd5e1", highlight: "#2563eb" } });
    });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     NODE TOOLTIP
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function showNodeTooltip(nodeId) {
    const tooltip = document.getElementById("node-tooltip");
    if (!currentSchema) return;

    const node = currentSchema.nodes.find(n => n.id === nodeId);
    if (!node) return;

    const layerNames = { 1: "L1: Evidence", 2: "L2: Indicator", 3: "L3: Threat", 4: "L4: Risk" };
    const layerCols  = { 1: "#38bdf8", 2: "#fbbf24", 3: "#a855f7", 4: "#f43f5e" };

    document.getElementById("tooltip-name").textContent = nodeId;
    const layerSpan = document.getElementById("tooltip-layer");
    layerSpan.textContent = layerNames[node.layer] || "";
    layerSpan.style.background = (layerCols[node.layer] || "#666") + "22";
    layerSpan.style.color = layerCols[node.layer] || "#aaa";

    let html = "";
    const post = posteriorMap[nodeId];
    if (post !== undefined) {
      html += `<div class="stat-row"><span class="stat-label">Posterior P=1</span><span class="stat-val" style="color:${getValueColor(post)}">${(post * 100).toFixed(2)}%</span></div>`;
    }
    if (node.prior !== undefined) {
      html += `<div class="stat-row"><span class="stat-label">Prior P=1</span><span class="stat-val">${(node.prior * 100).toFixed(2)}%</span></div>`;
    }
    if (node.leak !== undefined) {
      html += `<div class="stat-row"><span class="stat-label">Noisy-OR Leak λ</span><span class="stat-val">${node.leak.toFixed(3)}</span></div>`;
    }
    if (node.parents && node.parents.length) {
      html += `<div class="stat-row"><span class="stat-label">Parents (${node.parents.length})</span><span class="stat-val" style="font-size:0.68rem">${node.parents.slice(0,3).join(", ")}${node.parents.length > 3 ? " …" : ""}</span></div>`;
    }
    html += `<div class="stat-row"><span class="stat-label">Layer</span><span class="stat-val">${node.layer_name || layerNames[node.layer]}</span></div>`;

    document.getElementById("tooltip-body").innerHTML = html;
    tooltip.style.display = "block";
  }

  function hideNodeTooltip() {
    document.getElementById("node-tooltip").style.display = "none";
  }

  function getValueColor(v) {
    if (v >= 0.6) return "#f87171";
    if (v >= 0.35) return "#fbbf24";
    return "#34d399";
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     FILE UPLOAD — CSV / JSON / TXT / LOG PARSING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function initFileUpload() {
    const dropZone  = document.getElementById("drop-zone");
    const uploadBtn = document.getElementById("upload-btn");
    const fileInput = document.getElementById("file-input");

    uploadBtn.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", () => {
      if (fileInput.files[0]) processUploadedFile(fileInput.files[0]);
    });

    // Drag and drop
    dropZone.addEventListener("dragover", e => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files[0]) processUploadedFile(e.dataTransfer.files[0]);
    });
  }

  function processUploadedFile(file) {
    const reader = new FileReader();
    const ext    = file.name.split(".").pop().toLowerCase();

    reader.onload = e => {
      const content = e.target.result;
      let evidence  = {};
      let parseErr  = null;

      try {
        if (ext === "json") {
          evidence = parseJSONEvidence(content);
        } else if (ext === "csv") {
          evidence = parseCSVEvidence(content);
        } else {
          // .log / .txt — keyword scan
          evidence = parseLogTextEvidence(content);
        }
      } catch (err) {
        parseErr = err.message;
      }

      const statusDiv = document.getElementById("upload-status");
      const statusMsg = document.getElementById("upload-status-msg");

      if (parseErr) {
        statusDiv.style.display = "flex";
        statusMsg.textContent   = "Parse error: " + parseErr;
        statusDiv.querySelector("i").style.color = "#ef4444";
      } else {
        const count = Object.keys(evidence).filter(k => evidence[k] === 1).length;
        statusDiv.style.display = "flex";
        statusMsg.textContent = `Parsed "${file.name}" — ${count} active evidence signal(s) found`;
        statusDiv.querySelector("i").style.color = "#10b981";
        setEvidenceCheckboxes(evidence);
      }
    };

    reader.readAsText(file);
  }

  /* Parse JSON: handles various schema formats */
  function parseJSONEvidence(content) {
    const parsed = JSON.parse(content);
    let ev = {};

    // Schema 1: flat {NodeName: 0/1} dict
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      // Check if it's our own case JSON format
      if (parsed.active_evidence && Array.isArray(parsed.active_evidence)) {
        parsed.active_evidence.forEach(n => { ev[n] = 1; });
        return ev;
      }
      // Check for evidence sub-key
      if (parsed.evidence && typeof parsed.evidence === "object") {
        ev = parseJSONEvidence(JSON.stringify(parsed.evidence));
        return ev;
      }
      // Flat dict — scan for known evidence nodes
      const knownPrefixes = ["AfterHours", "Unusual", "Device", "File", "External", "High", "USB", "RemovableMedia", "Cloud", "Shadow", "Printer", "Resignation", "Job", "Performance"];
      Object.entries(parsed).forEach(([k, v]) => {
        if (typeof v === "number" || typeof v === "boolean") {
          ev[k] = Number(v) >= 1 ? 1 : 0;
        }
      });
      return ev;
    }

    // Schema 2: array of log events — count anomalies
    if (Array.isArray(parsed)) {
      parsed.forEach(row => {
        Object.entries(row).forEach(([k, v]) => {
          if ((Number(v) >= 1) && !ev[k]) ev[k] = 1;
        });
      });
      return ev;
    }

    return ev;
  }

  /* Parse CSV: first data row of a evidence-column CSV */
  function parseCSVEvidence(content) {
    const lines = content.trim().split("\n").filter(l => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV must have a header row and at least one data row.");

    const sep     = lines[0].includes("\t") ? "\t" : ",";
    const headers = lines[0].split(sep).map(h => h.trim().replace(/^"|"$/g, ""));
    const ev      = {};

    // Find the first non-header, non-metadata row that has evidence columns
    // We look for rows where at least one column matches a known evidence-node pattern.
    // Strategy: try each data row and pick the one with most active evidence signals.
    let bestRow = null, bestCount = -1;

    for (let i = 1; i < Math.min(lines.length, 10); i++) {
      const vals = lines[i].split(sep).map(v => v.trim().replace(/^"|"$/g, ""));
      const row  = {};
      headers.forEach((h, idx) => { row[h] = vals[idx]; });

      const activeCount = Object.values(row).filter(v => Number(v) === 1).length;
      if (activeCount > bestCount) { bestCount = activeCount; bestRow = row; }
    }

    if (bestRow) {
      Object.entries(bestRow).forEach(([k, v]) => {
        const num = Number(v);
        if (!isNaN(num)) ev[k] = num >= 1 ? 1 : 0;
      });
    }
    return ev;
  }

  /* Parse log/txt: keyword matching against known evidence node names */
  function parseLogTextEvidence(content) {
    const evidenceKeywords = {
      "AfterHoursLogin":       ["after.?hours", "outside.?hours", "off.?hours.?login", "3am", "2am", "midnight"],
      "UnusualLoginFreq":      ["multiple.?login", "repeated.?auth", "login.?spike", "high.?logon"],
      "DeviceConnectActivity": ["usb", "removable.?media", "external.?drive", "thumb.?drive", "mass.?storage"],
      "FileAccessCount":       ["file.?access", "mass.?file", "file.?enumerat", "dir.?scan"],
      "FileCopyToRemovable":   ["copy.?usb", "file.?copy.*removable", "xcopy.*d:\\\\", "exfil.*usb"],
      "ExternalEmail":         ["external.?email", "personal.?email", "gmail", "hotmail", "yahoo"],
      "UnusualEmailVolume":    ["email.?volume", "mass.?email", "bulk.?email"],
      "UnusualWebActivity":    ["tor", "vpn", "proxy", "unusual.?web", "high.?http"],
      "HighDataMovement":      ["large.?attachment", "data.?movement", "high.*bytes", "exfil.*email"],
      "HighFileCopyActivity":  ["high.?copy", "rapid.?copy", "multiple.?file.*copied"],
      "CloudUploadAnomaly":    ["dropbox", "googledrive", "s3.?upload", "cloud.?upload", "sharepoint.*anomal"],
      "PrinterActivity":       ["print", "printer", "pdf.*creat"],
      "ResignationSent":       ["resignation", "quit", "leaving", "two.?weeks", "notice"],
      "JobSearchActivity":     ["linkedin", "indeed", "job.?board", "headhunter"],
      "PerformanceReview":     ["performance", "pip", "review.*negative", "terminated"],
      "ShadowITUsage":         ["unauthorized.?app", "shadow.?it", "steganograph", "encrypted.*personal"],
    };

    const lower = content.toLowerCase();
    const ev = {};

    Object.entries(evidenceKeywords).forEach(([nodeName, patterns]) => {
      const found = patterns.some(p => new RegExp(p, "i").test(lower));
      ev[nodeName] = found ? 1 : 0;
    });

    return ev;
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     SAMPLES
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  async function fetchSamples() {
    try {
      const res  = await fetch("/api/samples");
      const data = await res.json();
      const sel  = document.getElementById("sample-select");

      data.samples.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = s.label;
        opt.dataset.evidence = JSON.stringify(s.evidence);
        sel.appendChild(opt);
      });

      sel.addEventListener("change", () => {
        const opt = sel.options[sel.selectedIndex];
        if (opt && opt.dataset.evidence) {
          setEvidenceCheckboxes(JSON.parse(opt.dataset.evidence));
          runAnalysis();
        }
      });
    } catch (err) {
      console.warn("Samples load:", err);
    }
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     EVENTS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function setupEvents() {
    document.getElementById("btn-run-analysis").addEventListener("click", runAnalysis);

    document.getElementById("btn-fit-graph").addEventListener("click", () => {
      if (networkGraph) networkGraph.fit({ animation: { duration: 400 } });
    });

    document.getElementById("btn-physics-toggle").addEventListener("click", () => {
      if (networkGraph) {
        const cur = networkGraph.physics.options.enabled;
        networkGraph.setOptions({ physics: { enabled: !cur } });
      }
    });

    document.getElementById("btn-reset-highlight").addEventListener("click", () => {
      if (!currentSchema) return;
      currentSchema.nodes.forEach(n => {
        const c = currentUI === "neo" ? getNeoLayerColors(n.layer) : getLayerColors(n.layer);
        nodesDataSet.update({ id: n.id, borderWidth: 1.5, color: { background: c.bg, border: c.border }, shadow: { enabled: n.layer >= 3 } });
      });
      edgesDataSet.forEach(e => {
        edgesDataSet.update({ id: e.id, color: { color: currentUI === "neo" ? "rgba(0,255,65,0.08)" : "rgba(255,255,255,0.1)" }, width: 1 });
      });
    });

    document.getElementById("btn-export-math").addEventListener("click", exportMathTrace);
    
    // Setup PDF Export Button
    const pdfBtn = document.getElementById("btn-export-pdf");
    if (pdfBtn) {
      pdfBtn.addEventListener("click", exportPDFReport);
    }

    document.getElementById("network-container").addEventListener("click", e => {
      if (e.target === e.currentTarget) hideNodeTooltip();
    });

    window.addEventListener("resize", () => { if (networkGraph) networkGraph.fit(); });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     MAIN ANALYSIS (Delayed 3-4s Evaluation)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  async function runAnalysis() {
    updateActiveEvidenceCount();

    const runBtn = document.getElementById("btn-run-analysis");
    runBtn.disabled = true;
    runBtn.classList.add("loading");
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Evaluating…';

    const panel = document.getElementById("results-panel");
    const body  = document.getElementById("results-body");
    panel.style.display = "block";
    body.innerHTML = `
      <div class="eval-loading-box">
        <div class="eval-loading-header">
          <i class="fa-solid fa-gear fa-spin eval-gear-icon"></i>
          <span>Evaluating Bayesian Network</span>
        </div>
        <div class="eval-progress-bar-wrap">
          <div class="eval-progress-bar-fill" id="eval-progress-fill" style="width: 10%;"></div>
        </div>
        <div class="eval-loading-status" id="eval-loading-status">
          Initializing Variable Elimination Engine...
        </div>
      </div>
    `;

    const sel = document.getElementById("sample-select");
    const userLabel = sel.options[sel.selectedIndex]?.textContent || "Interactive Query";

    const statusEl = () => document.getElementById("eval-loading-status");
    const fillEl   = () => document.getElementById("eval-progress-fill");

    const t1 = setTimeout(() => {
      if (fillEl()) fillEl().style.width = "45%";
      if (statusEl()) statusEl().textContent = "Extracting log evidence & factors...";
    }, 900);

    const t2 = setTimeout(() => {
      if (fillEl()) fillEl().style.width = "75%";
      if (statusEl()) statusEl().textContent = "Eliminating variables & computing marginals...";
    }, 1900);

    const t3 = setTimeout(() => {
      if (fillEl()) fillEl().style.width = "95%";
      if (statusEl()) statusEl().textContent = "Computing decision network utility & threat vectors...";
    }, 2800);

    const minTimePromise = new Promise(resolve => setTimeout(resolve, 3500));

    const fetchPromise = fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user: userLabel.split(" ")[0] || "ANALYST",
        day:  new Date().toISOString().slice(0, 10),
        evidence: currentEvidence
      })
    }).then(res => res.json());

    try {
      const [_, data] = await Promise.all([minTimePromise, fetchPromise]);
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);

      if (data.status === "success") {
        lastAnalysisResult = data;

        // Build posterior lookup map for tooltips
        posteriorMap = {};
        if (data.behavioral_indicators) Object.assign(posteriorMap, data.behavioral_indicators);
        if (data.threat_posteriors)      Object.assign(posteriorMap, data.threat_posteriors);
        const riskPost = data.risk_assessment.posterior;
        posteriorMap["RiskLevel"] = riskPost.High;

        updateRiskBadge(data.risk_assessment);
        renderSidebarResults(data);
        highlightNetworkPaths(data.active_threat_paths, data.active_evidence);
        renderMathTrace(data.math_trace);

        // Trigger LLM analysis in background without auto-switching tabs
        triggerLLMAnalysis(data);
      } else {
        showError(data.message || "Analysis failed.");
      }
    } catch (err) {
      clearTimeout(t1); clearTimeout(t2); clearTimeout(t3);
      showError("Network error: " + err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.classList.remove("loading");
      runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Threat Inference';
    }
  }

  function activateTab(tabId) {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelector(`[data-tab="${tabId}"]`)?.classList.add("active");
    document.querySelectorAll(".canvas-tab").forEach(c => c.classList.remove("active"));
    document.getElementById(tabId)?.classList.add("active");
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     RISK BADGE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function updateRiskBadge(riskAss) {
    const badge = document.getElementById("risk-badge");
    const text  = document.getElementById("risk-level-text");
    const level = riskAss.level.toLowerCase();
    badge.className = `risk-badge risk-${level}`;
    text.textContent = `${riskAss.level.toUpperCase()} RISK (${(riskAss.p_high * 100).toFixed(1)}%)`;
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     SIDEBAR RESULTS PANEL
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function renderSidebarResults(data) {
    const panel = document.getElementById("results-panel");
    const body  = document.getElementById("results-body");
    panel.style.display = "block";

    const risk = data.risk_assessment;
    const threats = data.threat_posteriors;

    let html = "";

    // Risk posteriors
    ["High", "Medium", "Low"].forEach(r => {
      const p = risk.posterior[r] || 0;
      const col = r === "High" ? "#dc2626" : r === "Medium" ? "#d97706" : "#16a34a";
      const bgCol = r === "High" ? "#fee2e2" : r === "Medium" ? "#fef3c7" : "#dcfce7";
      html += `
        <div class="result-row">
          <span class="result-label">P(Risk = ${r})</span>
          <span class="result-value" style="color:${col}">${(p*100).toFixed(1)}%</span>
        </div>
        <div class="result-bar" style="background:${bgCol}">
          <div class="result-bar-fill" style="width:${(p*100).toFixed(1)}%;background:${col}"></div>
        </div>
      `;
    });

    // Separator
    html += `<div style="height:1px;background:#e2e8f0;margin:0.5rem 0;"></div>`;

    // Top threat
    Object.entries(threats).sort((a,b) => b[1]-a[1]).forEach(([name, prob]) => {
      const col = prob > 0.5 ? "#dc2626" : prob > 0.25 ? "#d97706" : "#64748b";
      const bgCol = prob > 0.5 ? "#fee2e2" : prob > 0.25 ? "#fef3c7" : "#f1f5f9";
      html += `
        <div class="result-row">
          <span class="result-label">${name}</span>
          <span class="result-value" style="color:${col}">${(prob*100).toFixed(1)}%</span>
        </div>
        <div class="result-bar" style="background:${bgCol}">
          <div class="result-bar-fill" style="width:${Math.min(prob*100,100).toFixed(1)}%;background:${col}"></div>
        </div>
      `;
    });

    // Decision
    html += `<div style="height:1px;background:#e2e8f0;margin:0.5rem 0;"></div>`;
    html += `
      <div class="result-row">
        <span class="result-label" style="font-weight:700;color:#0f172a;">Recommended Action</span>
      </div>
      <div style="padding: 0.35rem 0 0.5rem;">
        <span style="font-size:0.78rem;font-weight:700;color:#2563eb;background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:0.25rem 0.65rem;display:inline-block;">
          ${data.decision_network.recommended_action}
        </span>
      </div>
    `;

    body.innerHTML = html;
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     NETWORK PATH HIGHLIGHTING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function highlightNetworkPaths(pathData, activeEvidence) {
    if (!nodesDataSet || !pathData) return;

    const activations = pathData.node_activations || {};

    nodesDataSet.forEach(node => {
      const nodeInfo = currentSchema?.nodes.find(n => n.id === node.id);
      const layer = nodeInfo?.layer || 1;
      const act = activations[node.id] ?? 0.05;
      const isEv = Array.isArray(activeEvidence) && activeEvidence.includes(node.id);

      const baseColors = getLayerColors(layer);
      let borderColor = baseColors.border;
      let bgColor     = baseColors.bg;
      let borderWidth = 1.5;
      let shadow      = { enabled: false };

      if (isEv) {
        // Active evidence nodes: bright blue highlight
        borderColor = "#2563eb";
        bgColor     = "#dbeafe";
        borderWidth = 3;
        shadow      = { enabled: true, size: 8, color: "rgba(37,99,235,0.3)" };
      } else if (act > 0.6) {
        // High activation (>60%)
        borderColor = "#dc2626";
        bgColor     = "#fee2e2";
        borderWidth = 2.5;
        shadow      = { enabled: true, size: 6, color: "rgba(220,38,38,0.25)" };
      } else if (act > 0.35) {
        // Moderate activation
        borderColor = "#d97706";
        bgColor     = "#fef3c7";
        borderWidth = 2;
        shadow      = { enabled: true, size: 4, color: "rgba(217,119,6,0.2)" };
      } else {
        // Low activation — dim
        borderColor = "#cbd5e1";
        bgColor     = "#f8fafc";
        borderWidth = 1;
      }

      nodesDataSet.update({
        id: node.id,
        borderWidth,
        color: { background: bgColor, border: borderColor },
        shadow
      });
    });

    // Highlight active edges
    const activePaths = pathData.active_threat_paths || pathData.paths || [];
    const edgeSet = new Set();
    if (Array.isArray(activePaths)) {
      activePaths.forEach(p => {
        if (Array.isArray(p)) {
          for (let i = 0; i < p.length - 1; i++) {
            edgeSet.add(`${p[i]}->${p[i+1]}`);
          }
        }
      });
    }

    edgesDataSet.forEach(e => {
      const k = `${e.from}->${e.to}`;
      if (edgeSet.has(k)) {
        edgesDataSet.update({ id: e.id, color: { color: "#dc2626" }, width: 2.5 });
      } else {
        edgesDataSet.update({ id: e.id, color: { color: "#cbd5e1" }, width: 1.2 });
      }
    });
  }

  function adjustColorOpacity(hexOrRgb, opacity) {
    return hexOrRgb + Math.round(opacity * 255).toString(16).padStart(2, "0");
  }

  function adjustBg(baseHex, mix) {
    return baseHex; // simplified
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     MATH TRACE — Fixed LaTeX Rendering
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function renderMathTrace(mathTrace) {
    const container = document.getElementById("math-trace-container");
    const placeholder = document.getElementById("math-placeholder");
    if (!mathTrace) return;

    if (placeholder) placeholder.style.display = "none";
    container.innerHTML = "";

    // ── STEP 1: Evidence Vector ─────────────────────────────────────────────
    let html = `
      <div class="math-step-card">
        <div class="step-header">
          <span class="step-badge">STEP 1</span>
          <span class="step-title">Evidence Discretization &amp; Prior Distributions</span>
        </div>
        <p class="step-desc">
          Raw system logs are mapped to binary evidence variables
          <em>E<sub>i</sub> ∈ {0, 1}</em>. Prior probabilities P(E<sub>i</sub> = 1) reflect
          flag firing rates learned from the CERT dataset (per-user baselines).
        </p>
    `;

    // Evidence as a vector with active nodes highlighted
    const allEv   = mathTrace.step1_evidence || [];
    const activeEv = allEv.filter(e => e.value === 1);

    // Full evidence vector display
    const vecStr = allEv.map(e => e.value).join(",\\; ");
    html += renderKatexBlock(
      `\\mathbf{E} = \\begin{bmatrix} ${vecStr} \\end{bmatrix}^\\top \\quad |\\mathbf{E}_{\\text{active}}| = ${activeEv.length}`
    );

    // Table of active nodes
    if (activeEv.length > 0) {
      html += `
        <table style="font-size:0.78rem;margin-top:0.75rem;width:100%;border-collapse:collapse;">
          <thead>
            <tr style="border-bottom:1px solid rgba(255,255,255,0.07);">
              <th style="padding:0.4rem 0.75rem;color:#06d6f0;text-align:left;font-size:0.7rem;text-transform:uppercase;">Node</th>
              <th style="padding:0.4rem 0.75rem;color:#06d6f0;text-align:center;font-size:0.7rem;text-transform:uppercase;">State</th>
              <th style="padding:0.4rem 0.75rem;color:#06d6f0;text-align:center;font-size:0.7rem;text-transform:uppercase;">Prior P(E=1)</th>
            </tr>
          </thead>
          <tbody>
      `;
      allEv.filter(e => e.value === 1).forEach(e => {
        html += `
          <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
            <td style="padding:0.35rem 0.75rem;color:#f0f4ff;font-weight:600;">${e.node}</td>
            <td style="padding:0.35rem 0.75rem;text-align:center;color:#06d6f0;font-weight:700;">1 (Active)</td>
            <td style="padding:0.35rem 0.75rem;text-align:center;color:#a0b3cc;">${e.prior.toFixed(3)}</td>
          </tr>
        `;
      });
      html += `</tbody></table>`;
    } else {
      html += `<p class="step-desc" style="margin-top:0.5rem;color:#f59e0b;">No active evidence signals — baseline prior distribution used.</p>`;
    }
    html += `</div>`;

    // ── STEP 2: Noisy-OR ────────────────────────────────────────────────────
    const step2 = mathTrace.step2_noisy_or || [];
    html += `
      <div class="math-step-card">
        <div class="step-header">
          <span class="step-badge">STEP 2</span>
          <span class="step-title">Noisy-OR Intermediate Node Calculations</span>
        </div>
        <p class="step-desc">
          Intermediate behavioral and threat hypothesis nodes aggregate their parent evidence
          using the <strong>Noisy-OR parameterization</strong>:
        </p>
    `;

    // Noisy-OR formula header
    html += renderKatexBlock(
      `P(Y_j = 1 \\mid \\mathbf{E}) = 1 - \\underbrace{(1 - \\lambda_j)}_{\\text{leak}} \\prod_{i \\in \\mathcal{P}_j^{\\text{active}}} (1 - w_{ij})`
    );

    // Only show nodes that have active parents (or are interesting)
    const interestingNodes = step2.filter(n => n.active_parents.length > 0 || n.calculated_probability > 0.1);
    if (interestingNodes.length === 0 && step2.length > 0) {
      // Show first 4 even if all base rates
      interestingNodes.push(...step2.slice(0, 4));
    }

    interestingNodes.forEach(item => {
      const prob = item.calculated_probability;
      const col  = prob > 0.5 ? "#f87171" : prob > 0.25 ? "#fbbf24" : "#34d399";
      html += `
        <div class="math-subitem">
          <div class="math-subitem-label">
            ${item.node}
            <span style="margin-left:0.5rem;font-weight:400;color:${col}">→ P = ${(prob*100).toFixed(2)}%</span>
            ${item.active_parents.length > 0 ? `<span style="margin-left:0.5rem;font-size:0.68rem;color:#64748b;">(${item.active_parents.length} active parent${item.active_parents.length>1?"s":""})</span>` : ""}
          </div>
      `;
      html += renderKatexBlock(item.latex);
      html += `</div>`;
    });

    if (step2.length > interestingNodes.length) {
      html += `<p class="step-desc" style="text-align:center;margin-top:0.5rem;">${step2.length - interestingNodes.length} additional nodes at baseline prior (no active parents).</p>`;
    }
    html += `</div>`;

    // ── STEP 3: Variable Elimination ────────────────────────────────────────
    const step3 = mathTrace.step3_variable_elimination;
    if (step3) {
      html += `
        <div class="math-step-card">
          <div class="step-header">
            <span class="step-badge">STEP 3</span>
            <span class="step-title">Exact Variable Elimination — Posterior Inference</span>
          </div>
          <p class="step-desc">
            Ancestor-pruning reduces the 35-node DAG to the active causal subgraph.
            The Variable Elimination algorithm marginalizes hidden variables using
            <strong>${step3.strategy}</strong> elimination ordering
            (Induced Width = ${step3.induced_width}, Max Factor Size = ${step3.max_factor_size},
            Runtime = ${step3.runtime_ms} ms).
          </p>
          <p class="step-desc" style="margin-bottom:0.75rem;">
            The normalization step converts unnormalized potentials to a valid probability distribution:
          </p>
      `;
      html += renderKatexBlock(step3.latex_normalization);
      html += `</div>`;
    }

    // ── STEP 4: Decision Network ─────────────────────────────────────────────
    const step4 = mathTrace.step4_decision_utility;
    if (step4) {
      html += `
        <div class="math-step-card">
          <div class="step-header">
            <span class="step-badge">STEP 4</span>
            <span class="step-title">Decision Network — Maximum Expected Utility</span>
          </div>
          <p class="step-desc">
            The security response decision is chosen by maximizing expected utility over the posterior
            risk distribution and threat hypothesis probabilities. The utility function balances
            <em>false positive costs</em> (over-reacting) vs <em>missed-attack costs</em> (under-reacting):
          </p>
      `;

      // Optimal action formula
      html += renderKatexBlock(
        `a^{*} = \\arg\\max_{a \\in \\mathcal{A}} \\; E[U(a)] = \\arg\\max_{a} \\left[ -C(a) - \\sum_{t} P(T_t \\mid \\mathbf{E}) \\cdot L_t \\cdot r_t(a) \\right]`
      );

      html += `<p class="step-desc" style="margin:0.6rem 0 0.75rem;">Expected utility per action:</p>`;

      const items = step4.items || [];
      items.forEach(item => {
        const isSelected = item.is_selected;
        html += `
          <div class="math-subitem">
            <div class="math-subitem-label ${isSelected ? 'selected-action' : ''}">
              ${item.action}
              ${isSelected ? '<i class="fa-solid fa-star" style="font-size:0.7rem;"></i> <span style="font-size:0.7rem;font-weight:400;">Selected Optimal Action</span>' : `<span style="font-size:0.7rem;font-weight:400;color:#64748b;">E[U] = ${item.utility.toFixed(2)}</span>`}
            </div>
        `;
        html += renderKatexBlock(item.latex, isSelected);
        html += `</div>`;
      });

      html += `</div>`;
    }

    // ── STEP 5: Final Risk Assessment ────────────────────────────────────────
    const step5 = mathTrace.step5_final_risk;
    if (step5) {
      const isHigh = step5.classification === "HIGH RISK";
      const isMed  = step5.classification === "MEDIUM RISK";

      html += `
        <div class="math-step-card">
          <div class="step-header">
            <span class="step-badge highlight">STEP 5</span>
            <span class="step-title">Final Risk Probability &amp; Assessment Score</span>
          </div>
          <p class="step-desc">
            Composite risk metrics derived from the posterior distribution.
            The expected risk score uses a linear utility scale:
            L(Low)=0.0, L(Medium)=0.5, L(High)=1.0.
          </p>
      `;
      html += renderKatexBlock(step5.latex, true);

      const riskClass = isHigh ? "high" : isMed ? "med" : "low";
      html += `
        <div class="risk-summary-grid">
          <div class="risk-summary-card">
            <div class="rs-label">P(High Risk)</div>
            <div class="rs-value ${isHigh ? 'high' : ''}">${(step5.p_high * 100).toFixed(1)}%</div>
          </div>
          <div class="risk-summary-card">
            <div class="rs-label">P(Elevated)</div>
            <div class="rs-value ${isMed || isHigh ? 'med' : ''}">${(step5.p_elevated * 100).toFixed(1)}%</div>
          </div>
          <div class="risk-summary-card">
            <div class="rs-label">Risk Score</div>
            <div class="rs-value ${riskClass}">${(step5.risk_score * 100).toFixed(1)}%</div>
          </div>
          <div class="risk-summary-card">
            <div class="rs-label">Classification</div>
            <div class="rs-value ${riskClass}">${step5.classification}</div>
          </div>
        </div>
      `;
      html += `</div>`;
    }

    container.innerHTML = html;

    // ── Render all KaTeX ─────────────────────────────────────────────────────
    renderAllKatex(container);
  }

  /* Render a KaTeX display block safely */
  function renderKatexBlock(latexStr, isHighlight = false) {
    const highlightClass = isHighlight ? " highlight-block" : "";
    // Use a data attribute so we can render later in batch
    return `<div class="math-block${highlightClass}" data-latex="${encodeURIComponent(latexStr)}"></div>`;
  }

  /* Batch KaTeX render — safer than innerHTML $$…$$ delimiters */
  function renderAllKatex(container) {
    const blocks = container.querySelectorAll("[data-latex]");
    blocks.forEach(block => {
      const latex = decodeURIComponent(block.getAttribute("data-latex"));
      try {
        katex.render(latex, block, {
          displayMode: true,
          throwOnError: false,
          errorColor: "#f87171",
          output: "htmlAndMathml"
        });
        block.removeAttribute("data-latex");
      } catch (e) {
        block.textContent = "⚠ LaTeX error: " + e.message;
        block.style.color = "#f87171";
        console.warn("KaTeX render error:", e.message, "\nLatex:", latex.slice(0, 200));
      }
    });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     CPT TABLE RENDERING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function renderCPTTable(schema) {
    const container = document.getElementById("cpt-container");
    if (!schema?.nodes) return;

    const byLayer = [1, 2, 3, 4].map(l => schema.nodes.filter(n => n.layer === l));
    const [l1, l2, l3, l4] = byLayer;

    let html = "";

    // Layer 1
    html += `
      <div class="cpt-layer-card">
        <h3>
          <span><i class="fa-solid fa-eye" style="color:#38bdf8;margin-right:0.5rem;"></i>Layer 1: Observable Evidence Nodes (${l1.length})</span>
          <span class="layer-badge layer-badge-1">L1 Evidence</span>
        </h3>
        <p class="cpt-desc">21 binary evidence variables E<sub>i</sub> ∈ {0,1} extracted from user-day system logs.
          Prior probabilities estimated from the CERT insider threat dataset baseline.</p>
        <table class="cpt-table">
          <thead><tr>
            <th>Node Name</th>
            <th>Prior P(E=1)</th>
            <th>Prior P(E=0)</th>
            <th>Type</th>
          </tr></thead>
          <tbody>
    `;
    l1.forEach(n => {
      const p1 = (n.prior ?? 0.05).toFixed(4);
      const p0 = (1 - (n.prior ?? 0.05)).toFixed(4);
      html += `
        <tr>
          <td style="font-weight:600;color:#f0f4ff;">${n.id}</td>
          <td style="color:#38bdf8;font-weight:600;">${p1}</td>
          <td style="color:#5a7090;">${p0}</td>
          <td style="color:#5a7090;font-size:0.7rem;">Bernoulli Root Node</td>
        </tr>
      `;
    });
    html += `</tbody></table></div>`;

    // Layer 2
    html += `
      <div class="cpt-layer-card">
        <h3>
          <span><i class="fa-solid fa-user-gear" style="color:#fbbf24;margin-right:0.5rem;"></i>Layer 2: Behavioral Indicator Nodes (${l2.length})</span>
          <span class="layer-badge layer-badge-2">L2 Indicators</span>
        </h3>
        <p class="cpt-desc">9 intermediate anomaly indicators parameterized using Noisy-OR combinations of active evidence parents.
          P(Y=1|E) = 1 − (1−λ) ∏ (1−w<sub>i</sub>).</p>
        <table class="cpt-table">
          <thead><tr>
            <th>Indicator</th>
            <th>Parent Nodes</th>
            <th>Leak λ</th>
            <th>Weights w<sub>i</sub></th>
          </tr></thead>
          <tbody>
    `;
    l2.forEach(n => {
      const pStr = (n.parents?.length ? n.parents.join(", ") : "—");
      const wStr = (n.weights?.length ? n.weights.map(w => w.toFixed(3)).join(", ") : "—");
      const lStr = n.leak !== undefined ? n.leak.toFixed(4) : "0.0300";
      html += `
        <tr>
          <td style="font-weight:600;color:#f0f4ff;">${n.id}</td>
          <td style="color:#fbbf24;font-size:0.72rem;">${pStr}</td>
          <td style="color:#f87171;font-weight:600;">${lStr}</td>
          <td style="color:#38bdf8;font-size:0.72rem;">${wStr}</td>
        </tr>
      `;
    });
    html += `</tbody></table></div>`;

    // Layer 3
    html += `
      <div class="cpt-layer-card">
        <h3>
          <span><i class="fa-solid fa-skull-crossbones" style="color:#a855f7;margin-right:0.5rem;"></i>Layer 3: Threat Hypothesis Nodes (${l3.length})</span>
          <span class="layer-badge layer-badge-3">L3 Threats</span>
        </h3>
        <p class="cpt-desc">4 core insider threat vectors evaluated from behavioral indicator activations using Noisy-OR.</p>
        <table class="cpt-table">
          <thead><tr>
            <th>Threat Hypothesis</th>
            <th>Parent Indicators</th>
            <th>Leak λ</th>
            <th>Causal Weights w<sub>i</sub></th>
          </tr></thead>
          <tbody>
    `;
    l3.forEach(n => {
      const pStr = (n.parents?.length ? n.parents.join(", ") : "—");
      const wStr = (n.weights?.length ? n.weights.map(w => w.toFixed(3)).join(", ") : "—");
      const lStr = n.leak !== undefined ? n.leak.toFixed(4) : "0.0200";
      html += `
        <tr>
          <td style="font-weight:600;color:#f0f4ff;">${n.id}</td>
          <td style="color:#a855f7;font-size:0.72rem;">${pStr}</td>
          <td style="color:#f87171;font-weight:600;">${lStr}</td>
          <td style="color:#38bdf8;font-size:0.72rem;">${wStr}</td>
        </tr>
      `;
    });
    html += `</tbody></table></div>`;

    // Layer 4
    const riskNode = l4[0];
    html += `
      <div class="cpt-layer-card">
        <h3>
          <span><i class="fa-solid fa-triangle-exclamation" style="color:#f43f5e;margin-right:0.5rem;"></i>Layer 4: Risk Level Node (${l4.length})</span>
          <span class="layer-badge layer-badge-4">L4 Risk</span>
        </h3>
        <p class="cpt-desc">Ternary target variable (Low=0, Medium=1, High=2) defined by a full CPT over 2<sup>4</sup>=16 parent threat configurations.</p>
        <table class="cpt-table">
          <thead><tr>
            <th>Risk Node</th>
            <th>Parent Threats</th>
            <th>States</th>
            <th>CPT Type</th>
          </tr></thead>
          <tbody>
    `;
    l4.forEach(n => {
      const pStr = n.parents?.length ? n.parents.join(", ") : "DataExfiltration, UnauthorizedAccess, ITSabotage, IPTheft";
      html += `
        <tr>
          <td style="font-weight:600;color:#f0f4ff;">${n.id}</td>
          <td style="color:#f43f5e;">${pStr}</td>
          <td style="color:#38bdf8;font-weight:600;">3 (Low, Medium, High)</td>
          <td style="color:#5a7090;font-size:0.72rem;">Categorical CPT (16 configurations)</td>
        </tr>
      `;
    });
    html += `</tbody></table>`;

    // Full CPT Matrix if available
    if (riskNode?.cpt && Object.keys(riskNode.cpt).length > 0) {
      html += `
        <h4 style="margin:1.25rem 0 0.6rem;font-size:0.82rem;color:#38bdf8;display:flex;align-items:center;gap:0.5rem;">
          <i class="fa-solid fa-table-list"></i>
          P(RiskLevel | Threat Combination) — Full Categorical CPT Matrix
        </h4>
        <table class="cpt-table">
          <thead><tr>
            <th>Active Threats</th>
            <th style="font-family:var(--font-mono);">(Exfil, Unauth, Sabotage, IPTheft)</th>
            <th style="color:#34d399;">P(Low)</th>
            <th style="color:#fbbf24;">P(Medium)</th>
            <th style="color:#f87171;">P(High)</th>
          </tr></thead>
          <tbody>
      `;
      Object.entries(riskNode.cpt).forEach(([comboStr, probs]) => {
        const comboArr  = comboStr.replace(/[()\\s]/g, "").split(",").map(Number);
        const activeNum = comboArr.reduce((a, b) => a + b, 0);
        const pL = (probs[0] ?? probs["0"] ?? 0).toFixed(3);
        const pM = (probs[1] ?? probs["1"] ?? 0).toFixed(3);
        const pH = (probs[2] ?? probs["2"] ?? 0).toFixed(3);
        html += `
          <tr>
            <td style="font-weight:600;color:${activeNum > 0 ? '#fbbf24' : '#5a7090'}">${activeNum} active</td>
            <td style="font-family:var(--font-mono);font-size:0.72rem;color:#a0b3cc;">${comboStr}</td>
            <td style="color:#34d399;">${pL}</td>
            <td style="color:#fbbf24;">${pM}</td>
            <td style="color:#f87171;font-weight:700;">${pH}</td>
          </tr>
        `;
      });
      html += `</tbody></table>`;
    }

    html += `</div>`;
    container.innerHTML = html;
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     LLM AI ANALYSIS — OpenRouter Streaming
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     LLM AI ANALYSIS — OpenRouter Streaming
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  async function triggerLLMAnalysis(analysisData) {
    const placeholder = document.getElementById("llm-placeholder");
    const output      = document.getElementById("llm-output");
    const content     = document.getElementById("llm-content");
    const cursor      = document.getElementById("llm-cursor");
    const timestamp   = document.getElementById("llm-timestamp");
    const pdfBtn      = document.getElementById("btn-export-pdf");
    const tabBadge    = document.getElementById("llm-tab-badge");

    if (pdfBtn) pdfBtn.style.display = "none";
    placeholder.style.display = "none";
    output.style.display = "block";
    content.innerHTML = "";
    cursor.style.display = "block";
    timestamp.textContent = new Date().toLocaleTimeString();

    if (tabBadge) {
      tabBadge.style.display = "inline-flex";
      tabBadge.className = "nav-tab-badge streaming";
    }

    setLLMStatus("thinking");

    // Build the rich prompt
    const prompt = buildLLMPrompt(analysisData);

    // Abort previous stream if active
    if (llmAbortCtrl) { try { llmAbortCtrl.abort(); } catch(e) {} }
    llmAbortCtrl = new AbortController();

    let fullText = "";

    try {
      const resp = await fetch("/api/llm-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
        signal: llmAbortCtrl.signal
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}: ${resp.statusText}`);
      }

      if (!resp.body) {
        // Fallback: read full response at once
        const text = await resp.text();
        content.innerHTML = (typeof marked !== "undefined") ? marked.parse(text) : text;
        cursor.style.display = "none";
        setLLMStatus("idle");
        if (tabBadge) tabBadge.className = "nav-tab-badge ready";
        return;
      }

      setLLMStatus("streaming");

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        let done, value;
        try {
          ({ done, value } = await reader.read());
        } catch (readErr) {
          // Stream ended / aborted
          break;
        }
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE lines
        const lines = buffer.split("\n");
        // Keep the last (potentially incomplete) line in buffer
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === "data: [DONE]") continue;
          if (trimmed.startsWith("data: ")) {
            const jsonStr = trimmed.slice(6);
            try {
              const parsed = JSON.parse(jsonStr);
              const delta  = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                fullText += delta;
                content.innerHTML = (typeof marked !== "undefined")
                  ? marked.parse(fullText)
                  : fullText.replace(/\n/g, "<br>");
              }
            } catch { /* non-JSON SSE line, skip */ }
          }
        }
      }

      cursor.style.display = "none";
      setLLMStatus("idle");
      if (tabBadge) tabBadge.className = "nav-tab-badge ready";

      // Final render pass
      if (fullText) {
        content.innerHTML = (typeof marked !== "undefined")
          ? marked.parse(fullText)
          : fullText;
        if (pdfBtn) pdfBtn.style.display = "inline-flex"; // Show PDF button when done
      }

    } catch (err) {
      if (err.name === "AbortError") {
        cursor.style.display = "none";
        setLLMStatus("idle");
        if (tabBadge) tabBadge.style.display = "none";
        return;
      }
      console.error("LLM stream error:", err);
      cursor.style.display = "none";
      setLLMStatus("idle");
      if (tabBadge) tabBadge.style.display = "none";
      if (!fullText) {
        content.innerHTML = `<div style="color:#f87171;padding:0.5rem 0;">
          <i class="fa-solid fa-triangle-exclamation"></i>
          <strong>LLM analysis error:</strong> ${err.message}
          <p style="margin-top:0.5rem;font-size:0.75rem;color:#64748b;">
            Check that the Flask server is running and /api/llm-analyze is accessible.
          </p>
        </div>`;
      }
    }
  }


  function setLLMStatus(state) {
    const dot = document.getElementById("llm-status-dot");
    if (state === "thinking") {
      dot.className = "llm-status-thinking";
      dot.innerHTML = '<i class="fa-solid fa-spinner"></i> Analyzing…';
    } else if (state === "streaming") {
      dot.className = "llm-status-streaming";
      dot.innerHTML = '<i class="fa-solid fa-circle-notch"></i> Streaming…';
    } else {
      dot.className = "llm-status-idle";
      dot.innerHTML = '<i class="fa-solid fa-circle"></i> Ready';
    }
  }

  function buildLLMPrompt(data) {
    const risk = data.risk_assessment;
    const threats = data.threat_posteriors;
    const behavioral = data.behavioral_indicators;
    const decision = data.decision_network;
    const ablation = data.evidence_ablation || [];
    const activeEv = data.active_evidence || [];
    const mathTrace = data.math_trace;
    const stats = data.inference_stats;

    const threatText = Object.entries(threats)
      .sort((a,b) => b[1]-a[1])
      .map(([t, p]) => `  - ${t}: P = ${(p*100).toFixed(2)}%`)
      .join("\n");

    const behavText = Object.entries(behavioral)
      .sort((a,b) => b[1]-a[1])
      .map(([b, p]) => `  - ${b}: P = ${(p*100).toFixed(2)}%`)
      .join("\n");

    const ablText = ablation.slice(0,5).map(a =>
      `  - Removing ${a.node}: ΔP(High) = ${a.delta_p_high > 0 ? "↓" : "↑"}${Math.abs(a.delta_p_high*100).toFixed(2)}% → new P(High) = ${(a.p_high_without*100).toFixed(2)}%`
    ).join("\n");

    const utilText = Object.entries(decision.expected_utilities || {})
      .map(([a, u]) => `  - ${a}: E[U] = ${u.toFixed(2)}${a === decision.recommended_action ? " ← SELECTED" : ""}`)
      .join("\n");

    let noisyOrSummary = "";
    if (mathTrace?.step2_noisy_or) {
      const active = mathTrace.step2_noisy_or.filter(n => n.active_parents.length > 0);
      noisyOrSummary = active.map(n =>
        `  - ${n.node}: λ=${n.leak.toFixed(3)}, active_parents=[${n.active_parents.join(", ")}], P=${(n.calculated_probability*100).toFixed(2)}%`
      ).join("\n");
    }

    return `You are an expert probabilistic cybersecurity analyst specializing in insider threat detection using Bayesian Networks. Your task is to produce a rigorous, technically accurate, and actionable threat analysis report based on the output of a 35-node Bayesian Network inference run.

## PROBABILISTIC INFERENCE OUTPUT

**User/Day:** ${data.user} / ${data.day}

**Risk Posterior Distribution:**
- P(Risk = Low)    = ${(risk.posterior.Low*100).toFixed(3)}%
- P(Risk = Medium) = ${(risk.posterior.Medium*100).toFixed(3)}%
- P(Risk = High)   = ${(risk.posterior.High*100).toFixed(3)}%
- **Overall Assessment: ${risk.level.toUpperCase()} RISK**

**Threat Hypothesis Posteriors (Layer 3):**
${threatText}

**Behavioral Indicator Activations (Layer 2):**
${behavText}

**Active Evidence Signals (Layer 1):** ${activeEv.length > 0 ? activeEv.join(", ") : "(none — baseline prior)"}

**Decision Network — Expected Utilities:**
${utilText}
Recommended Action: **${decision.recommended_action}**
Action Description: ${decision.action_description}

---

## YOUR TASK

Write a structured technical threat analysis report with these exact sections. Be precise, technical, and grounded ONLY in the numbers above. Do NOT invent evidence or change probabilities. This report will be exported as a professional PDF for management.

## 1. Executive Risk Summary
2-3 sentences. State the overall risk level and what it means probabilistically. Reference P(High) explicitly.

## 2. Active Evidence & Indicators
Briefly list the active evidence signals and how they triggered the behavioral indicators.

## 3. Threat Assessment
Explain the highest probability threat vectors (Layer 3) and their correlation with the observed behavior.

## 4. Proposed Solution & Actions
Explain WHY the utility maximization chose ${decision.recommended_action}. Provide 3-5 concrete, prioritized, actionable recommendations for the security operations team.

Keep each section technically rigorous but clear. Use exact probability values from the data above.`;
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     EXPORT PDF REPORT (html2pdf)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     EXPORT PDF REPORT (html2pdf)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function exportPDFReport() {
    const riskBadgeText = document.getElementById("risk-level-text")?.textContent || "ASSESSMENT UNKNOWN";
    const rawContentHtml = document.getElementById("llm-content")?.innerHTML || "";
    
    // Extract metadata from last analysis result if available
    const userLabel     = lastAnalysisResult?.user || "ANALYST";
    const dateStr       = lastAnalysisResult?.day || new Date().toISOString().slice(0, 10);
    const pHigh         = lastAnalysisResult?.risk_assessment?.p_high !== undefined 
                          ? (lastAnalysisResult.risk_assessment.p_high * 100).toFixed(1) + "%" 
                          : "N/A";
    const recAction     = lastAnalysisResult?.decision_network?.recommended_action || "N/A";
    const activeEvCount = lastAnalysisResult?.active_evidence?.length || 0;

    const riskColor = riskBadgeText.includes("HIGH") 
                      ? "#ef4444" 
                      : riskBadgeText.includes("MEDIUM") 
                      ? "#f59e0b" 
                      : "#10b981";

    const printWrapper = document.createElement("div");
    printWrapper.style.padding = "0";
    printWrapper.style.margin = "0";
    printWrapper.style.background = "#ffffff";

    printWrapper.innerHTML = `
      <div style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: #1e293b; padding: 24px 30px; max-width: 800px; margin: 0 auto; background: #ffffff;">
        <!-- Header Banner -->
        <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #0f172a; padding-bottom: 12px; margin-bottom: 20px;">
          <div>
            <h1 style="font-size: 20px; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -0.02em;">
              CERT Insider Threat Analysis Report
            </h1>
            <p style="color: #64748b; font-size: 11px; margin: 4px 0 0; font-weight: 500;">
              Probabilistic Risk Evaluation · 35-Node Bayesian Network · CMU CERT R4.2
            </p>
          </div>
          <div style="text-align: right;">
            <div style="font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Generated</div>
            <div style="font-size: 11px; font-weight: 600; color: #334155;">${dateStr}</div>
          </div>
        </div>

        <!-- Executive Summary Cards Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 22px;">
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid ${riskColor}; padding: 10px; border-radius: 6px;">
            <div style="font-size: 9px; font-weight: 700; color: #64748b; text-transform: uppercase;">Risk Classification</div>
            <div style="font-size: 12px; font-weight: 800; color: ${riskColor}; margin-top: 2px;">${riskBadgeText}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid #3b82f6; padding: 10px; border-radius: 6px;">
            <div style="font-size: 9px; font-weight: 700; color: #64748b; text-transform: uppercase;">P(High Risk)</div>
            <div style="font-size: 13px; font-weight: 800; color: #1e293b; margin-top: 2px;">${pHigh}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid #06b6d4; padding: 10px; border-radius: 6px;">
            <div style="font-size: 9px; font-weight: 700; color: #64748b; text-transform: uppercase;">Optimal Action</div>
            <div style="font-size: 11px; font-weight: 800; color: #0891b2; margin-top: 2px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${recAction}</div>
          </div>
          <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-top: 3px solid #8b5cf6; padding: 10px; border-radius: 6px;">
            <div style="font-size: 9px; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Signals</div>
            <div style="font-size: 13px; font-weight: 800; color: #1e293b; margin-top: 2px;">${activeEvCount} node(s)</div>
          </div>
        </div>

        <!-- Markdown Body Content -->
        <div class="pdf-content-body" style="font-size: 12.5px; line-height: 1.65; color: #334155;">
          ${rawContentHtml}
        </div>

        <!-- Footer -->
        <div style="margin-top: 35px; padding-top: 12px; border-top: 1px solid #e2e8f0; font-size: 10px; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
          <span>CONFIDENTIAL — SECURITY OPERATIONS EYES ONLY</span>
          <span>CERT 35-Node Bayesian Network &amp; Decision System</span>
        </div>
      </div>
    `;

    // Apply PDF print styles to headings, tables, code blocks, lists, etc.
    const body = printWrapper.querySelector('.pdf-content-body');
    if (body) {
      body.querySelectorAll('h1').forEach(h => {
        h.style.fontSize = '16px';
        h.style.color = '#0f172a';
        h.style.borderBottom = '1px solid #cbd5e1';
        h.style.paddingBottom = '4px';
        h.style.marginTop = '18px';
        h.style.marginBottom = '8px';
      });
      body.querySelectorAll('h2').forEach(h => {
        h.style.fontSize = '14px';
        h.style.color = '#0f172a';
        h.style.marginTop = '16px';
        h.style.marginBottom = '6px';
        h.style.fontWeight = '700';
      });
      body.querySelectorAll('h3').forEach(h => {
        h.style.fontSize = '13px';
        h.style.color = '#1e293b';
        h.style.marginTop = '12px';
        h.style.marginBottom = '4px';
      });
      body.querySelectorAll('p').forEach(p => {
        p.style.marginBottom = '8px';
      });
      body.querySelectorAll('ul, ol').forEach(list => {
        list.style.paddingLeft = '20px';
        list.style.marginBottom = '10px';
      });
      body.querySelectorAll('li').forEach(li => {
        li.style.marginBottom = '4px';
      });
      body.querySelectorAll('code').forEach(c => {
        c.style.background = '#f1f5f9';
        c.style.color = '#0f172a';
        c.style.padding = '1px 5px';
        c.style.borderRadius = '3px';
        c.style.fontFamily = 'monospace';
        c.style.fontSize = '11.5px';
      });
      body.querySelectorAll('pre').forEach(p => {
        p.style.background = '#f8fafc';
        p.style.border = '1px solid #e2e8f0';
        p.style.padding = '10px';
        p.style.borderRadius = '6px';
        p.style.overflowX = 'auto';
        p.style.marginBottom = '10px';
      });
      body.querySelectorAll('blockquote').forEach(bq => {
        bq.style.borderLeft = '3px solid #3b82f6';
        bq.style.background = '#f0f9ff';
        bq.style.padding = '8px 12px';
        bq.style.margin = '10px 0';
        bq.style.borderRadius = '0 4px 4px 0';
        bq.style.color = '#1e3a8a';
      });
      body.querySelectorAll('table').forEach(t => {
        t.style.width = '100%';
        t.style.borderCollapse = 'collapse';
        t.style.marginBottom = '12px';
        t.style.fontSize = '11.5px';
      });
      body.querySelectorAll('th').forEach(th => {
        th.style.background = '#f1f5f9';
        th.style.color = '#0f172a';
        th.style.border = '1px solid #cbd5e1';
        th.style.padding = '6px 8px';
        th.style.textAlign = 'left';
      });
      body.querySelectorAll('td').forEach(td => {
        td.style.border = '1px solid #e2e8f0';
        td.style.padding = '5px 8px';
        td.style.color = '#334155';
      });
    }

    const opt = {
      margin:       [10, 10, 10, 10],
      filename:     `CERT_Threat_Report_${userLabel}_${Date.now()}.pdf`,
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2, useCORS: true, logging: false },
      jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' },
      pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
    };

    const pdfBtn = document.getElementById("btn-export-pdf");
    const originalHtml = pdfBtn.innerHTML;
    pdfBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating PDF...';
    pdfBtn.disabled = true;

    html2pdf().set(opt).from(printWrapper).save().then(() => {
      pdfBtn.innerHTML = originalHtml;
      pdfBtn.disabled = false;
    }).catch(err => {
      console.error("PDF generation failed:", err);
      pdfBtn.innerHTML = originalHtml;
      pdfBtn.disabled = false;
      alert("Failed to generate PDF report. Check console for details.");
    });
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     EXPORT MATH TRACE
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function exportMathTrace() {
    if (!lastAnalysisResult?.math_trace) {
      alert("Run an analysis first to export the math trace.");
      return;
    }
    const blob = new Blob(
      [JSON.stringify(lastAnalysisResult.math_trace, null, 2)],
      { type: "application/json" }
    );
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `cert_math_trace_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     UTILITY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
  function showError(msg) {
    const llmContent = document.getElementById("llm-content");
    if (llmContent) {
      document.getElementById("llm-placeholder").style.display = "none";
      document.getElementById("llm-output").style.display = "block";
      llmContent.innerHTML = `<p style="color:#f87171;"><i class="fa-solid fa-triangle-exclamation"></i> ${msg}</p>`;
    }
  }

}); // End DOMContentLoaded
