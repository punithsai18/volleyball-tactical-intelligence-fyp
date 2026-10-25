/**
 * VolleyTactix AI - Front-End Controller & Interactive Formation Canvas
 */

// Render Backend Integration
const RENDER_BACKEND_KEY = "VOLLEYTACTIX_RENDER_BACKEND";
const DEFAULT_RENDER_BACKEND = "https://volleyball-tactical-intelligence-backend.onrender.com";

function getApiBaseUrl() {
  const customUrl = localStorage.getItem(RENDER_BACKEND_KEY);
  if (customUrl) return customUrl.replace(/\/$/, "");
  if (window.RENDER_BACKEND_URL) return window.RENDER_BACKEND_URL.replace(/\/$/, "");
  // When running live on Vercel or public domain, auto-connect to Render backend
  if (typeof window !== "undefined" && window.location && window.location.hostname) {
    if (!["localhost", "127.0.0.1"].includes(window.location.hostname)) {
      return DEFAULT_RENDER_BACKEND;
    }
  }
  return "";
}

function apiUrl(endpoint) {
  if (!endpoint) return "";
  if (endpoint.startsWith("http://") || endpoint.startsWith("https://")) return endpoint;
  const base = getApiBaseUrl();
  return base ? `${base}${endpoint}` : endpoint;
}

// Application State
const state = {
  currentModel: "hist_gradient_boosting",
  currentSampleId: null,
  currentPrediction: null,
  currentFrameIdx: 0,
  isPlaying: false,
  playSpeed: 1.0,
  playTimer: null,
  samples: [],
  models: [],
  matchTimeline: null,
  activeMatchWindow: null,
  consensusChart: null,

  // Sandbox State
  sandbox: {
    players: [
      [850, 680],  // P1 (setter / backright)
      [680, 620],  // P2
      [540, 610],  // P3 (left)
      [950, 750],  // P4
      [1100, 660], // P5
      [1250, 560]  // P6 (front attacker)
    ],
    ball: [1260, 200],
    draggingIdx: -1, // 0..5 for players, 6 for ball, -1 for none
    isDragging: false,
    ripples: []
  },

  // Audio FX Suite
  audioEnabled: localStorage.getItem("volleytactix_audio") !== "false",
  audioCtx: null,

  // Spatial Visualizer Layers
  layers: {
    hull: true,
    mst: true,
    heatmap: true,
    vectors: true,
    zones: true,
    trails: true,
    centroid: true,
    roles: true
  },

  // Match Stream Autoplay
  matchReplay: {
    isPlaying: false,
    timer: null,
    speed: 1.0,
    currentIndex: 0
  }
};

// Web Audio API Sound Suite
function playTone(freq, type = "sine", duration = 0.08, gainVal = 0.04) {
  if (!state.audioEnabled) return;
  try {
    if (!state.audioCtx) {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioCtx.state === "suspended") {
      state.audioCtx.resume();
    }
    const osc = state.audioCtx.createOscillator();
    const gain = state.audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, state.audioCtx.currentTime);
    gain.gain.setValueAtTime(gainVal, state.audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(state.audioCtx.destination);
    osc.start();
    osc.stop(state.audioCtx.currentTime + duration);
  } catch (e) {}
}

function playClickSound() {
  playTone(850, "sine", 0.04, 0.02);
}

function playTacticChime() {
  playTone(523.25, "sine", 0.12, 0.03); // C5
  setTimeout(() => playTone(659.25, "sine", 0.16, 0.03), 80); // E5
}

function playWhistle() {
  playTone(1800, "square", 0.05, 0.015);
  setTimeout(() => playTone(2100, "square", 0.07, 0.015), 60);
}

// Tactical Colors Mapping (High Contrast for Light & Dark)
const TACTIC_COLORS = {
  "Coordinated Attack": "#d97706",
  "Coordinated Defense": "#059669",
  "Delayed Support": "#db2777",
  "Spacing Breakdown": "#7c3aed",
  "Unclassified": "#64748b"
};

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", () => {
  initLenis();
  initVanta();
  initHeroConstellation();
  initGSAPAnimations();
  setupLandingHeroInteractions();

  setupNavigation();
  setupGlobalControls();
  setupSoundToggle();
  setupCourtCanvas();
  setupSandboxCanvas();
  setupMatchReplayControls();
  setupModelDuel();
  setupReportExporter();
  setupUploadDropzone();
  setupVideoUpload();
  setup3DCardTilt();
  setupProductShowcaseModal();
  startAmbientCanvasLoop();
  
  // Initial API Fetch
  loadSamplesAndModels();
  loadMatchTimeline();
});

// Setup Tab Navigation with GSAP Smooth Page Transitions
function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
        
        // GSAP smooth page transition for every tab/page
        if (typeof gsap !== "undefined") {
          gsap.fromTo(targetPane,
            { opacity: 0, y: 18, filter: "blur(2px)" },
            { opacity: 1, y: 0, filter: "blur(0px)", duration: 0.42, ease: "power2.out" }
          );

          // Stagger direct cards inside the incoming tab
          const innerCards = targetPane.querySelectorAll(".glass-card, .model-card, .timeline-control-panel, .feature-card");
          if (innerCards.length > 0) {
            gsap.from(innerCards, {
              opacity: 0,
              y: 16,
              duration: 0.38,
              stagger: 0.05,
              ease: "power2.out"
            });
          }
        }
      }

      // Handle specific tab focus events
      if (tab.dataset.tab === "tab-consensus") {
        runConsensusInference();
      } else if (tab.dataset.tab === "tab-sandbox") {
        renderSandboxCourt();
        triggerSandboxPrediction();
      }
    });
  });
}

// Global Controls
function setupGlobalControls() {
  const modelSelect = document.getElementById("globalModelSelect");
  modelSelect.addEventListener("change", (e) => {
    state.currentModel = e.target.value;
    if (state.currentSampleId) {
      loadSample(state.currentSampleId);
    }
  });

  document.getElementById("reRunInferenceBtn").addEventListener("click", () => {
    if (state.currentSampleId) loadSample(state.currentSampleId);
  });

  document.getElementById("refreshConsensusBtn").addEventListener("click", () => {
    runConsensusInference();
  });

  // Setup Theme Toggle (Light as Default)
  setupThemeToggle();
}

function setupThemeToggle() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  
  // Default to light theme as requested!
  const savedTheme = localStorage.getItem("volleytactix_theme") || "light";
  applyTheme(savedTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "light" ? "dark" : "light";
      applyTheme(next);
      localStorage.setItem("volleytactix_theme", next);
      renderCourt();
      renderSandboxCourt();
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const themeIcon = document.getElementById("themeIcon");
  const themeText = document.getElementById("themeText");
  if (themeIcon && themeText) {
    if (theme === "dark") {
      themeIcon.innerText = "🌙";
      themeText.innerText = "Dark";
    } else {
      themeIcon.innerText = "☀️";
      themeText.innerText = "Light";
    }
  }
  updateVantaTheme(theme);
}

// ---------------------------------------------------------------------------
// 1. DATA LOADING & API REQUESTS
// ---------------------------------------------------------------------------

async function loadSamplesAndModels() {
  try {
    // 1. Fetch Models & Leaderboard
    const modelsRes = await fetch(apiUrl("/api/models"));
    const modelsData = await modelsRes.json();
    state.models = modelsData.models;
    populateLeaderboard(modelsData.leaderboard);

    // 2. Fetch Samples
    const samplesRes = await fetch(apiUrl("/api/samples"));
    const samplesData = await samplesRes.json();
    state.samples = samplesData.samples;
    renderSampleChips(state.samples);

    // Load first sample by default
    if (state.samples.length > 0) {
      loadSample(state.samples[0].id);
    }
  } catch (err) {
    console.error("Error loading initial data:", err);
  }
}

function renderSampleChips(samples) {
  const container = document.getElementById("sampleChipsList");
  container.innerHTML = "";

  samples.forEach((sample, idx) => {
    const chip = document.createElement("button");
    chip.className = `sample-chip ${idx === 0 ? "active" : ""}`;
    chip.dataset.id = sample.id;
    chip.innerHTML = `
      <span class="chip-dot" style="background: ${sample.color};"></span>
      <span>${sample.name}</span>
    `;

    chip.addEventListener("click", () => {
      document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      loadSample(sample.id);
    });

    container.appendChild(chip);
  });
}

async function loadSample(sampleId) {
  try {
    state.currentSampleId = sampleId;
    const res = await fetch(apiUrl(`/api/sample/${sampleId}?model_key=${state.currentModel}`));
    const data = await res.json();
    state.currentPrediction = data;
    state.currentFrameIdx = 0;
    
    // Update UI elements
    updateVerdictUI(data);
    updatePlaybackControls(data);
    renderCourt();
  } catch (err) {
    console.error("Failed to load sample:", err);
  }
}

function updateVerdictUI(data) {
  const pred = data.prediction;
  const label = pred.label;
  const color = pred.color || TACTIC_COLORS[label] || "#64748b";

  // Subtitle
  document.getElementById("currentClipSubtitle").innerText = `${data.sample_id || 'Rally'} • 29 Frames • Model: ${data.model_name}`;
  document.getElementById("verdictSubtitle").innerText = `Active Model: ${data.model_name} (Rank #${data.model_rank} ${data.model_type})`;

  // Hero Box
  const hero = document.getElementById("verdictHero");
  hero.style.background = `${color}18`;
  hero.style.borderColor = `${color}66`;

  const labelEl = document.getElementById("verdictLabel");
  labelEl.innerText = label;
  labelEl.style.color = color;

  const catEl = document.getElementById("verdictCategory");
  catEl.innerText = pred.tactic_type;
  catEl.style.color = color;

  const confEl = document.getElementById("confidenceValue");
  const confPct = (pred.confidence * 100).toFixed(1);
  confEl.innerText = `${confPct}%`;
  confEl.style.color = color;

  const barFill = document.getElementById("confidenceBarFill");
  barFill.style.width = `${confPct}%`;
  barFill.style.background = color;

  document.getElementById("verdictDescription").innerText = pred.description;

  // Entropy & Anomaly
  document.getElementById("entropyVal").innerText = `${pred.entropy} bits (norm: ${pred.entropy_norm})`;
  const anomalyTile = document.getElementById("anomalyTile");
  const anomalyVal = document.getElementById("anomalyVal");
  if (pred.is_anomaly) {
    anomalyVal.innerText = "⚠️ Low Confidence Flag";
    anomalyVal.style.color = "#ef4444";
    anomalyTile.style.borderColor = "rgba(239, 68, 68, 0.4)";
  } else {
    anomalyVal.innerText = "Nominal (Valid)";
    anomalyVal.style.color = "var(--defense-color)";
    anomalyTile.style.borderColor = "var(--border-glass)";
  }

  // Probability Breakdown Bars
  const probs = pred.probabilities;
  const attackPct = ((probs["Coordinated Attack"] || 0) * 100).toFixed(1);
  const defensePct = ((probs["Coordinated Defense"] || 0) * 100).toFixed(1);
  const delayedPct = ((probs["Delayed Support"] || 0) * 100).toFixed(1);
  const spacingPct = ((probs["Spacing Breakdown"] || 0) * 100).toFixed(1);

  document.getElementById("probValAttack").innerText = `${attackPct}%`;
  document.getElementById("probValDefense").innerText = `${defensePct}%`;
  document.getElementById("probValDelayed").innerText = `${delayedPct}%`;
  document.getElementById("probValSpacing").innerText = `${spacingPct}%`;

  // GSAP animations for verdict hero, probability meters, and telemetry
  if (typeof gsap !== "undefined") {
    gsap.fromTo(hero, { scale: 0.94, opacity: 0.7 }, { scale: 1, opacity: 1, duration: 0.45, ease: "back.out(1.5)" });
    gsap.to(barFill, { width: `${confPct}%`, duration: 0.75, ease: "power2.out" });
    gsap.to("#probFillAttack", { width: `${attackPct}%`, duration: 0.75, ease: "power2.out" });
    gsap.to("#probFillDefense", { width: `${defensePct}%`, duration: 0.75, ease: "power2.out" });
    gsap.to("#probFillDelayed", { width: `${delayedPct}%`, duration: 0.75, ease: "power2.out" });
    gsap.to("#probFillSpacing", { width: `${spacingPct}%`, duration: 0.75, ease: "power2.out" });
    gsap.from(".telemetry-tile", { opacity: 0, scale: 0.92, duration: 0.35, stagger: 0.04, ease: "power1.out" });
  } else {
    barFill.style.width = `${confPct}%`;
    document.getElementById("probFillAttack").style.width = `${attackPct}%`;
    document.getElementById("probFillDefense").style.width = `${defensePct}%`;
    document.getElementById("probFillDelayed").style.width = `${delayedPct}%`;
    document.getElementById("probFillSpacing").style.width = `${spacingPct}%`;
  }

  // AI Tactical Spatial Reasoning & Explainability
  updateTacticalExplainability(data);
}

function updateTacticalExplainability(data) {
  const pred = data.prediction;
  const label = pred.label;
  const explainText = document.getElementById("explainText");
  const explainBadge = document.getElementById("explainSpatialBadge");
  const explainTags = document.getElementById("explainTags");
  if (!explainText || !explainBadge || !explainTags) return;

  let explanation = "";
  let badge = "";
  let tags = [];

  if (label === "Coordinated Attack") {
    badge = "Offensive In-System";
    explanation = "Front-court players are synchronized within the 3m attack zone with the team centroid advancing aggressively toward the net. Minimal angular dispersion across spikers indicates planned set execution.";
    tags = ["🎯 Net Cluster: < 2.8m", "⚡ Acceleration: High", "📐 Hull Area: Compact Attack", "🏐 Setter Corridor: Open"];
  } else if (label === "Coordinated Defense") {
    badge = "Perimeter Envelope";
    explanation = "Backcourt defense forms a balanced arc across the 6-meter dig zone. Spacing between adjacent defenders is uniform (> 2.5m), optimizing lateral reaction against hard cross-court spikes.";
    tags = ["🛡️ Perimeter Depth: Balanced", "📐 Pair Dist: Uniform", "⭐ Centroid: Mid-Court Base", "🏐 Dig Coverage: 85%"];
  } else if (label === "Delayed Support") {
    badge = "Recovery Scramble";
    explanation = "Spatial lag detected between the ball trajectory and secondary support players. Centroid is misaligned from the defensive apex, indicating an out-of-system block touch or scrambled coverage.";
    tags = ["⚠️ Centroid Lag: High", "⏱️ Support Latency: > 0.4s", "📐 Asymmetric Spread", "🏐 Off-System Recovery"];
  } else if (label === "Spacing Breakdown") {
    badge = "Critical Miscommunication";
    explanation = "Severe spatial anomaly detected. Multiple players occupy overlapping coordinate clusters (< 1.4m separation) or an extreme rotational void (> 9m) leaves the court center unguarded.";
    tags = ["🚨 Cluster Conflict: Overlap", "📐 Seam Void: High", "⭐ Entropy Flagged", "🛑 Uncovered Zone: Center"];
  } else {
    badge = "Dynamic Transition";
    explanation = "Formation undergoing neutral positional transition between offensive approach and defensive posture.";
    tags = ["🔄 Neutral Setup", "📐 Spread: Standard", "⭐ Centroid: Center"];
  }

  explainBadge.innerText = badge;
  explainText.innerText = explanation;
  explainTags.innerHTML = tags.map(t => `<span class="tag-pill">${t}</span>`).join("");
}

// ---------------------------------------------------------------------------
// 2. 2D VOLLEYBALL COURT VISUALIZER
// ---------------------------------------------------------------------------

let courtCtx = null;

function setupCourtCanvas() {
  const canvas = document.getElementById("courtCanvas");
  courtCtx = canvas.getContext("2d");

  // Playback Buttons
  const playBtn = document.getElementById("playBtn");
  playBtn.addEventListener("click", togglePlayback);

  document.getElementById("prevBtn").addEventListener("click", () => {
    pausePlayback();
    if (state.currentFrameIdx > 0) {
      state.currentFrameIdx--;
      updateFrameUI();
    }
  });

  document.getElementById("nextBtn").addEventListener("click", () => {
    pausePlayback();
    if (state.currentPrediction && state.currentFrameIdx < state.currentPrediction.n_frames - 1) {
      state.currentFrameIdx++;
      updateFrameUI();
    }
  });

  const slider = document.getElementById("frameSlider");
  slider.addEventListener("input", (e) => {
    pausePlayback();
    state.currentFrameIdx = parseInt(e.target.value);
    updateFrameUI();
  });

  const speedBtn = document.getElementById("speedBtn");
  speedBtn.addEventListener("click", () => {
    if (state.playSpeed === 1.0) state.playSpeed = 0.5;
    else if (state.playSpeed === 0.5) state.playSpeed = 2.0;
    else state.playSpeed = 1.0;
    speedBtn.innerText = `${state.playSpeed}x`;
    if (state.isPlaying) {
      pausePlayback();
      startPlayback();
    }
  });

  // Layer Toggles
  ["hull", "mst", "heatmap", "vectors", "zones", "trails", "centroid", "roles"].forEach(layer => {
    const btnId = "toggleLayer" + layer.charAt(0).toUpperCase() + layer.slice(1);
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener("click", () => {
        state.layers[layer] = !state.layers[layer];
        btn.classList.toggle("active", state.layers[layer]);
        playClickSound();
        renderCourt();
      });
    }
  });
}

function togglePlayback() {
  if (state.isPlaying) {
    pausePlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  state.isPlaying = true;
  document.getElementById("playBtn").innerText = "⏸";
  document.getElementById("playBtn").classList.add("active-play");

  const intervalMs = (1000 / 30) / state.playSpeed;
  state.playTimer = setInterval(() => {
    if (!state.currentPrediction) return;
    if (state.currentFrameIdx >= state.currentPrediction.n_frames - 1) {
      state.currentFrameIdx = 0;
    } else {
      state.currentFrameIdx++;
    }
    updateFrameUI();
  }, intervalMs);
}

function pausePlayback() {
  state.isPlaying = false;
  if (state.playTimer) clearInterval(state.playTimer);
  document.getElementById("playBtn").innerText = "▶";
  document.getElementById("playBtn").classList.remove("active-play");
}

function updatePlaybackControls(data) {
  const slider = document.getElementById("frameSlider");
  slider.max = data.n_frames - 1;
  slider.value = state.currentFrameIdx;
  updateFrameUI();
}

function updateFrameUI() {
  document.getElementById("frameSlider").value = state.currentFrameIdx;
  const timeSec = (state.currentFrameIdx / 30).toFixed(2);
  document.getElementById("frameCounter").innerText = `Frame ${state.currentFrameIdx + 1} / 29 (${timeSec}s)`;

  // Update telemetry for this frame
  if (state.currentPrediction && state.currentPrediction.frames && state.currentPrediction.frames[state.currentFrameIdx]) {
    const frameData = state.currentPrediction.frames[state.currentFrameIdx];
    const geom = frameData.geometry;
    document.getElementById("telemetryCompactness").innerText = `${geom.compactness} m²`;
    document.getElementById("telemetrySpread").innerText = `${geom.spread} m`;
    document.getElementById("telemetryMinDist").innerText = `${geom.min_dist} m`;
    document.getElementById("telemetryActivePlayers").innerText = `${geom.active_count} / 6 Players`;
  }

  renderCourt();
}

// Convert broadcast coordinates (typically ~300..1600 X, ~100..900 Y) to canvas coordinates
function mapCourtCoord(x, y, canvasWidth, canvasHeight) {
  // Court bounding box in source broadcast resolution:
  // X: 450 to 1650 (span ~1200)
  // Y: 100 to 850 (span ~750)
  const pad = 40;
  const cw = canvasWidth - pad * 2;
  const ch = canvasHeight - pad * 2;

  const minX = 400, maxX = 1700;
  const minY = 100, maxY = 860;

  const nx = (x - minX) / (maxX - minX);
  const ny = (y - minY) / (maxY - minY);

  const cx = pad + Math.max(0, Math.min(1, nx)) * cw;
  const cy = pad + Math.max(0, Math.min(1, ny)) * ch;
  return [cx, cy];
}

function renderCourt() {
  if (!courtCtx) return;
  const canvas = courtCtx.canvas;
  const W = canvas.width;
  const H = canvas.height;

  courtCtx.clearRect(0, 0, W, H);

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";

  // 1. Draw Court Background & Flooring (Theme-Aware)
  courtCtx.fillStyle = isDark ? "#0c1626" : "#f1f5f9";
  courtCtx.fillRect(0, 0, W, H);

  // Volleyball Court Area
  const courtX = 40, courtY = 40, courtW = W - 80, courtH = H - 80;
  
  // Floor gradient: Electric Olympic Blue
  const floorGrad = courtCtx.createLinearGradient(0, courtY, 0, courtY + courtH);
  if (isDark) {
    floorGrad.addColorStop(0, "#13213a");
    floorGrad.addColorStop(1, "#1a2e4e");
  } else {
    floorGrad.addColorStop(0, "#1d4ed8");
    floorGrad.addColorStop(1, "#1e3a8a");
  }
  courtCtx.fillStyle = floorGrad;
  courtCtx.fillRect(courtX, courtY, courtW, courtH);

  // 1B. Animated Tactical Radar Sweep Beam
  const sweepProgress = (Date.now() / 2400) % 1;
  const sweepX = courtX + sweepProgress * courtW;
  const sweepGrad = courtCtx.createLinearGradient(sweepX - 50, 0, sweepX + 50, 0);
  sweepGrad.addColorStop(0, "rgba(56, 189, 248, 0)");
  sweepGrad.addColorStop(0.5, isDark ? "rgba(56, 189, 248, 0.18)" : "rgba(255, 255, 255, 0.28)");
  sweepGrad.addColorStop(1, "rgba(56, 189, 248, 0)");
  courtCtx.fillStyle = sweepGrad;
  courtCtx.fillRect(sweepX - 50, courtY, 100, courtH);

  // 2. Draw Court Boundary & Markings
  courtCtx.strokeStyle = "rgba(255, 255, 255, 0.65)";
  courtCtx.lineWidth = 2.5;
  courtCtx.strokeRect(courtX, courtY, courtW, courtH);

  // Net Line (Center)
  const netY = courtY + courtH * 0.35;
  courtCtx.strokeStyle = "#f43f5e";
  courtCtx.lineWidth = 3.5;
  courtCtx.beginPath();
  courtCtx.moveTo(courtX - 10, netY);
  courtCtx.lineTo(courtX + courtW + 10, netY);
  courtCtx.stroke();

  // Net 3D Mesh hatching
  courtCtx.strokeStyle = "rgba(244, 63, 94, 0.25)";
  courtCtx.lineWidth = 1;
  for (let x = courtX; x <= courtX + courtW; x += 14) {
    courtCtx.beginPath();
    courtCtx.moveTo(x, netY - 5);
    courtCtx.lineTo(x, netY + 5);
    courtCtx.stroke();
  }

  // 3-Meter Attack Line
  const attackLineY = courtY + courtH * 0.60;
  courtCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  courtCtx.lineWidth = 2;
  courtCtx.setLineDash([8, 8]);
  courtCtx.beginPath();
  courtCtx.moveTo(courtX, attackLineY);
  courtCtx.lineTo(courtX + courtW, attackLineY);
  courtCtx.stroke();
  courtCtx.setLineDash([]);

  // Court Labels
  courtCtx.fillStyle = "rgba(255, 255, 255, 0.35)";
  courtCtx.font = "600 11px Outfit, sans-serif";
  courtCtx.fillText("NET (FAR TEAM / OPPONENT)", courtX + 10, netY - 10);
  courtCtx.fillText("ATTACK LINE (3M)", courtX + 10, attackLineY - 8);
  courtCtx.fillText("DEFENSE BACKCOURT", courtX + 10, courtY + courtH - 12);

  // 2A. Draw FIVB Rotation Zones (Zones 1-6) if layer enabled
  if (state.layers.zones) {
    courtCtx.save();
    courtCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
    courtCtx.lineWidth = 1;
    courtCtx.setLineDash([4, 4]);

    const colW = courtW / 3;
    const col1X = courtX + colW;
    const col2X = courtX + colW * 2;
    courtCtx.beginPath();
    courtCtx.moveTo(col1X, courtY); courtCtx.lineTo(col1X, courtY + courtH);
    courtCtx.moveTo(col2X, courtY); courtCtx.lineTo(col2X, courtY + courtH);
    courtCtx.stroke();

    courtCtx.fillStyle = "rgba(255, 255, 255, 0.22)";
    courtCtx.font = "bold 12px Outfit, sans-serif";
    courtCtx.textAlign = "center";
    courtCtx.textBaseline = "middle";

    const frontY = (netY + attackLineY) / 2;
    courtCtx.fillText("ZONE 4", courtX + colW * 0.5, frontY);
    courtCtx.fillText("ZONE 3", courtX + colW * 1.5, frontY);
    courtCtx.fillText("ZONE 2", courtX + colW * 2.5, frontY);

    const backY = (attackLineY + courtY + courtH) / 2;
    courtCtx.fillText("ZONE 5", courtX + colW * 0.5, backY);
    courtCtx.fillText("ZONE 6", courtX + colW * 1.5, backY);
    courtCtx.fillText("ZONE 1", courtX + colW * 2.5, backY);
    courtCtx.restore();
  }

  if (!state.currentPrediction || !state.currentPrediction.frames) return;
  const frame = state.currentPrediction.frames[state.currentFrameIdx];
  if (!frame) return;

  const players = frame.players; // 6 players [ [x, y] or null ]
  const ball = frame.ball;       // [x, y] or null
  const geom = frame.geometry;

  // 2B. Draw Territorial Heatmap Density if layer enabled
  if (state.layers.heatmap) {
    players.forEach(p => {
      if (!p) return;
      const [px, py] = mapCourtCoord(p[0], p[1], W, H);
      const radGrad = courtCtx.createRadialGradient(px, py, 4, px, py, 60);
      radGrad.addColorStop(0, "rgba(217, 119, 6, 0.32)");
      radGrad.addColorStop(0.5, "rgba(2, 132, 199, 0.16)");
      radGrad.addColorStop(1, "rgba(2, 132, 199, 0)");
      courtCtx.fillStyle = radGrad;
      courtCtx.beginPath();
      courtCtx.arc(px, py, 60, 0, Math.PI * 2);
      courtCtx.fill();
    });
  }

  // 2B. Draw Motion Trails (Previous 2 frames) if layer enabled
  if (state.layers.trails && state.currentFrameIdx > 0) {
    const prevFrames = [state.currentPrediction.frames[state.currentFrameIdx - 1]];
    if (state.currentFrameIdx > 1) prevFrames.unshift(state.currentPrediction.frames[state.currentFrameIdx - 2]);

    prevFrames.forEach((pf, pfi) => {
      const alpha = (pfi + 1) * 0.18;
      // Faint ghost players
      pf.players.forEach(p => {
        if (!p) return;
        const [gpx, gpy] = mapCourtCoord(p[0], p[1], W, H);
        courtCtx.beginPath();
        courtCtx.arc(gpx, gpy, 8, 0, Math.PI * 2);
        courtCtx.fillStyle = `rgba(2, 132, 199, ${alpha})`;
        courtCtx.fill();
      });
      // Faint ghost ball
      if (pf.ball) {
        const [gbx, gby] = mapCourtCoord(pf.ball[0], pf.ball[1], W, H);
        courtCtx.beginPath();
        courtCtx.arc(gbx, gby, 5, 0, Math.PI * 2);
        courtCtx.fillStyle = `rgba(234, 179, 8, ${alpha})`;
        courtCtx.fill();
      }
    });
  }

  // 3. Draw Convex Hull Formation Polygon
  if (state.layers.hull && geom.hull && geom.hull.length >= 3) {
    courtCtx.beginPath();
    geom.hull.forEach((pt, i) => {
      const [hx, hy] = mapCourtCoord(pt[0], pt[1], W, H);
      if (i === 0) courtCtx.moveTo(hx, hy);
      else courtCtx.lineTo(hx, hy);
    });
    courtCtx.closePath();
    courtCtx.fillStyle = "rgba(6, 182, 212, 0.09)";
    courtCtx.fill();
    courtCtx.strokeStyle = "rgba(6, 182, 212, 0.45)";
    courtCtx.lineWidth = 1.75;
    courtCtx.stroke();
  }

  // 4. Draw MST Formation Connectivity Graph
  if (state.layers.mst && geom.mst_edges && geom.mst_edges.length > 0) {
    courtCtx.strokeStyle = "rgba(56, 189, 248, 0.65)";
    courtCtx.lineWidth = 2;
    geom.mst_edges.forEach(edge => {
      const [x1, y1] = mapCourtCoord(edge[0][0], edge[0][1], W, H);
      const [x2, y2] = mapCourtCoord(edge[1][0], edge[1][1], W, H);
      courtCtx.beginPath();
      courtCtx.moveTo(x1, y1);
      courtCtx.lineTo(x2, y2);
      courtCtx.stroke();
    });
  }

  // 5. Draw Center of Mass (Centroid)
  if (state.layers.centroid && geom.centroid && geom.centroid[0] > 0) {
    const [cx, cy] = mapCourtCoord(geom.centroid[0], geom.centroid[1], W, H);
    courtCtx.fillStyle = "rgba(245, 158, 11, 0.9)";
    courtCtx.beginPath();
    courtCtx.arc(cx, cy, 5.5, 0, Math.PI * 2);
    courtCtx.fill();
    courtCtx.strokeStyle = "#ffffff";
    courtCtx.lineWidth = 1.75;
    courtCtx.stroke();

    courtCtx.fillStyle = "rgba(245, 158, 11, 0.95)";
    courtCtx.font = "700 10px Inter, sans-serif";
    courtCtx.fillText("★ Centroid", cx + 8, cy + 4);
  }

  // 5B. Draw Velocity Vectors if layer enabled
  if (state.layers.vectors) {
    courtCtx.save();
    players.forEach((p, idx) => {
      if (!p) return;
      const [px, py] = mapCourtCoord(p[0], p[1], W, H);
      let vx = 0, vy = -10;
      if (state.currentFrameIdx > 0 && state.currentPrediction.frames[state.currentFrameIdx - 1]) {
        const prevP = state.currentPrediction.frames[state.currentFrameIdx - 1].players[idx];
        if (prevP) {
          const [prevX, prevY] = mapCourtCoord(prevP[0], prevP[1], W, H);
          vx = (px - prevX) * 2.5;
          vy = (py - prevY) * 2.5;
        }
      }
      const len = Math.min(30, Math.max(10, Math.hypot(vx, vy)));
      const angle = Math.atan2(vy, vx);

      courtCtx.strokeStyle = "rgba(56, 189, 248, 0.85)";
      courtCtx.lineWidth = 1.75;
      courtCtx.beginPath();
      courtCtx.moveTo(px, py);
      courtCtx.lineTo(px + Math.cos(angle) * len, py + Math.sin(angle) * len);
      courtCtx.stroke();

      const headLen = 5;
      courtCtx.beginPath();
      courtCtx.moveTo(px + Math.cos(angle) * len, py + Math.sin(angle) * len);
      courtCtx.lineTo(px + Math.cos(angle) * len - headLen * Math.cos(angle - Math.PI / 6),
                      py + Math.sin(angle) * len - headLen * Math.sin(angle - Math.PI / 6));
      courtCtx.moveTo(px + Math.cos(angle) * len, py + Math.sin(angle) * len);
      courtCtx.lineTo(px + Math.cos(angle) * len - headLen * Math.cos(angle + Math.PI / 6),
                      py + Math.sin(angle) * len - headLen * Math.sin(angle + Math.PI / 6));
      courtCtx.stroke();
    });
    courtCtx.restore();
  }

  // 6. Draw 6 Players with Role Badges
  players.forEach((p, idx) => {
    if (!p) return;
    const [px, py] = mapCourtCoord(p[0], p[1], W, H);

    // Player node dynamic pulsing outer glow
    const pulse = Math.sin(Date.now() / 220 + idx) * 3;
    courtCtx.beginPath();
    courtCtx.arc(px, py, 14 + pulse, 0, Math.PI * 2);
    courtCtx.fillStyle = "rgba(2, 132, 199, 0.25)";
    courtCtx.fill();

    // Player circle badge
    courtCtx.beginPath();
    courtCtx.arc(px, py, 11, 0, Math.PI * 2);
    courtCtx.fillStyle = "#0284c7";
    courtCtx.fill();
    courtCtx.strokeStyle = "#ffffff";
    courtCtx.lineWidth = 2;
    courtCtx.stroke();

    // Player Label (P1 - P6)
    courtCtx.fillStyle = "#ffffff";
    courtCtx.font = "bold 10px Outfit, sans-serif";
    courtCtx.textAlign = "center";
    courtCtx.textBaseline = "middle";
    courtCtx.fillText(`P${idx + 1}`, px, py);

    // Role Tag if layer enabled
    if (state.layers.roles) {
      let roleName = "Player";
      if (idx === 0) roleName = "Setter";
      else if (idx === 5) roleName = "Spiker";
      else if (p[1] > 700) roleName = "Defender";
      else roleName = "Middle";

      courtCtx.font = "600 9px Inter, sans-serif";
      courtCtx.fillStyle = "rgba(255, 255, 255, 0.85)";
      courtCtx.fillText(roleName, px, py + 18);
    }
  });

  // 7. Draw Ball with speed trail & animated shockwave ripple
  if (ball) {
    const [bx, by] = mapCourtCoord(ball[0], ball[1], W, H);

    // Ball glowing aura with harmonic breathing
    const ballPulse = Math.sin(Date.now() / 150) * 3.5;
    courtCtx.beginPath();
    courtCtx.arc(bx, by, 14 + ballPulse, 0, Math.PI * 2);
    courtCtx.fillStyle = "rgba(234, 179, 8, 0.38)";
    courtCtx.fill();

    // Animated expanding energy ripple ring
    const rippleRadius = (Date.now() / 20) % 26;
    const rippleAlpha = Math.max(0, 1 - rippleRadius / 26) * 0.45;
    courtCtx.beginPath();
    courtCtx.arc(bx, by, rippleRadius + 5, 0, Math.PI * 2);
    courtCtx.strokeStyle = `rgba(234, 179, 8, ${rippleAlpha})`;
    courtCtx.lineWidth = 1.5;
    courtCtx.stroke();

    // Ball node
    courtCtx.beginPath();
    courtCtx.arc(bx, by, 8, 0, Math.PI * 2);
    courtCtx.fillStyle = "#facc15";
    courtCtx.fill();
    courtCtx.strokeStyle = "#ffffff";
    courtCtx.lineWidth = 1.75;
    courtCtx.stroke();

    // Ball tag
    courtCtx.fillStyle = "#facc15";
    courtCtx.font = "bold 10px Inter, sans-serif";
    courtCtx.textAlign = "left";
    courtCtx.fillText("🏐 Ball", bx + 12, by + 3);
  }
}

// ---------------------------------------------------------------------------
// 3. 13-MODEL CONSENSUS & ENSEMBLE BENCHMARK
// ---------------------------------------------------------------------------

async function runConsensusInference() {
  if (!state.currentSampleId) return;

  try {
    const res = await fetch(apiUrl("/api/predict-all"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: state.currentSampleId })
    });
    const data = await res.json();
    renderConsensusView(data);
  } catch (err) {
    console.error("Failed to run consensus:", err);
  }
}

function renderConsensusView(data) {
  const consensus = data.consensus;
  const label = consensus.majority_label;
  const color = consensus.color || TACTIC_COLORS[label];

  document.getElementById("consensusLabel").innerText = label;
  document.getElementById("consensusLabel").style.color = color;
  document.getElementById("consensusPct").innerText = `${consensus.consensus_pct}%`;
  document.getElementById("consensusVotes").innerText = `${consensus.agreement_count} of ${consensus.total_models} Models`;

  // Render Vote Distribution Chart
  const chartCanvas = document.getElementById("consensusVoteChart");
  if (chartCanvas) {
    const ctx = chartCanvas.getContext("2d");
    const votes = consensus.votes_breakdown;
    const labels = Object.keys(votes);
    const counts = Object.values(votes);
    const bgColors = labels.map(l => TACTIC_COLORS[l] || "#94a3b8");

    const isDark = document.documentElement.getAttribute("data-theme") === "dark";

    if (state.consensusChart) state.consensusChart.destroy();
    state.consensusChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Model Votes",
          data: counts,
          backgroundColor: bgColors,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { ticks: { color: isDark ? "#94a3b8" : "#475569", font: { family: 'Outfit', size: 11, weight: '600' } }, grid: { display: false } },
          y: { ticks: { color: isDark ? "#94a3b8" : "#475569", stepSize: 1 }, grid: { color: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.06)" } }
        }
      }
    });
  }

  // Render 13 Model Cards Grid
  const grid = document.getElementById("modelsGridContainer");
  grid.innerHTML = "";

  data.models.forEach(m => {
    const card = document.createElement("div");
    card.className = "model-card";
    const mColor = m.color || TACTIC_COLORS[m.label];

    card.innerHTML = `
      <div class="model-card-top">
        <span class="model-rank-badge">Rank #${m.rank} • ${m.type}</span>
        <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">LOVO: ${m.test_acc}%</span>
      </div>
      <div class="model-name">${m.name}</div>
      <div class="model-meta">Macro-F1: ${m.test_f1} • Confidence: ${(m.confidence * 100).toFixed(1)}%</div>
      <div class="model-call-pill" style="background: ${mColor}1f; color: ${mColor}; border: 1px solid ${mColor}44;">
        <span>${m.label}</span>
        <span>${(m.confidence * 100).toFixed(0)}%</span>
      </div>
    `;
    grid.appendChild(card);
  });

  // GSAP 3D staggered card drop-in
  if (typeof gsap !== "undefined") {
    gsap.from(".consensus-hero", { scale: 0.96, opacity: 0, duration: 0.45, ease: "power2.out" });
    gsap.from(".model-card", {
      opacity: 0,
      y: 22,
      scale: 0.96,
      rotateX: 12,
      stagger: 0.04,
      duration: 0.45,
      ease: "back.out(1.2)"
    });
  }
}

// ---------------------------------------------------------------------------
// 4. OLYMPIC MATCH REPLAY (PARIS 2024 JAPAN VS ARGENTINA)
// ---------------------------------------------------------------------------

async function loadMatchTimeline() {
  try {
    const res = await fetch(apiUrl("/api/match-timeline"));
    const data = await res.json();
    state.matchTimeline = data;
    renderMatchTimeline(data);
  } catch (err) {
    console.error("Failed to load match timeline:", err);
  }
}

function renderMatchTimeline(data) {
  const strip = document.getElementById("matchTimelineStrip");
  strip.innerHTML = "";

  data.windows.forEach((win, idx) => {
    const seg = document.createElement("div");
    seg.className = `timeline-segment ${idx === 0 ? "active-window" : ""}`;
    seg.style.background = win.majority_color;
    seg.title = `Window #${win.seq} (${win.time_start.toFixed(1)}s - ${win.time_end.toFixed(1)}s): ${win.majority_label} (${win.agreement}/13 agree)`;

    seg.addEventListener("click", () => {
      document.querySelectorAll(".timeline-segment").forEach(s => s.classList.remove("active-window"));
      seg.classList.add("active-window");
      selectMatchWindow(win);
    });

    strip.appendChild(seg);
  });

  if (data.windows.length > 0) {
    selectMatchWindow(data.windows[0]);
  }
}

function selectMatchWindow(win) {
  state.activeMatchWindow = win;
  document.getElementById("matchWinSeq").innerText = `#${win.seq}`;
  document.getElementById("matchWinTime").innerText = `${win.time_start.toFixed(1)}s - ${win.time_end.toFixed(1)}s`;
  
  const majEl = document.getElementById("matchWinMajority");
  majEl.innerText = win.majority_label;
  majEl.style.color = win.majority_color;
  document.getElementById("matchWinAgreement").innerText = `${win.agreement} / 13 Models`;

  // HGB
  const hgbConfPct = (win.hgb_conf * 100).toFixed(1);
  const hgbColor = TACTIC_COLORS[win.hgb_label] || "#64748b";
  document.getElementById("matchWinHgbLabel").innerText = `${win.hgb_label} (${hgbConfPct}%)`;
  document.getElementById("matchWinHgbLabel").style.color = hgbColor;
  document.getElementById("matchWinHgbBar").style.width = `${hgbConfPct}%`;
  document.getElementById("matchWinHgbBar").style.background = hgbColor;

  // InceptionTime
  const dlConfPct = (win.dl_conf * 100).toFixed(1);
  const dlColor = TACTIC_COLORS[win.dl_label] || "#64748b";
  document.getElementById("matchWinDlLabel").innerText = `${win.dl_label} (${dlConfPct}%)`;
  document.getElementById("matchWinDlLabel").style.color = dlColor;
  document.getElementById("matchWinDlBar").style.width = `${dlConfPct}%`;
  document.getElementById("matchWinDlBar").style.background = dlColor;

  // Active indicator tag
  const indicator = document.getElementById("matchActiveWinIndicator");
  if (indicator && state.matchTimeline && state.matchTimeline.windows) {
    indicator.innerText = `Window #${win.seq} of ${state.matchTimeline.windows.length}`;
  }

  // GSAP animation on active window change
  if (typeof gsap !== "undefined") {
    gsap.fromTo(majEl, { scale: 0.9, opacity: 0.7 }, { scale: 1, opacity: 1, duration: 0.35, ease: "back.out(1.5)" });
    gsap.to("#matchWinHgbBar", { width: `${hgbConfPct}%`, duration: 0.55, ease: "power2.out" });
    gsap.to("#matchWinDlBar", { width: `${dlConfPct}%`, duration: 0.55, ease: "power2.out" });
  }

  // Flash animation & particle burst on window change
  if (typeof flashMatchWindow === "function") {
    flashMatchWindow();
  }
  if (typeof particleSystem !== "undefined" && particleSystem.burst) {
    const indicator = document.getElementById("matchActiveWinIndicator");
    if (indicator) {
      const rect = indicator.getBoundingClientRect();
      particleSystem.burst(rect.left + rect.width / 2, rect.top + rect.height / 2, win.majority_color || "#2563eb", 8);
    }
  }
}

function setupMatchReplayControls() {
  const playBtn = document.getElementById("matchReplayPlayBtn");
  const prevBtn = document.getElementById("matchReplayPrevBtn");
  const nextBtn = document.getElementById("matchReplayNextBtn");
  const speedBtn = document.getElementById("matchReplaySpeedBtn");

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (state.matchReplay.isPlaying) {
        stopMatchReplay();
      } else {
        startMatchReplay();
      }
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      stopMatchReplay();
      if (!state.matchTimeline || !state.matchTimeline.windows) return;
      const curIdx = state.activeMatchWindow ? (state.activeMatchWindow.seq - 1) : 0;
      if (curIdx > 0) {
        selectAndHighlightMatchWindow(curIdx - 1);
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      stopMatchReplay();
      if (!state.matchTimeline || !state.matchTimeline.windows) return;
      const curIdx = state.activeMatchWindow ? (state.activeMatchWindow.seq - 1) : 0;
      if (curIdx < state.matchTimeline.windows.length - 1) {
        selectAndHighlightMatchWindow(curIdx + 1);
      }
    });
  }

  if (speedBtn) {
    speedBtn.addEventListener("click", () => {
      if (state.matchReplay.speed === 1.0) state.matchReplay.speed = 2.0;
      else if (state.matchReplay.speed === 2.0) state.matchReplay.speed = 4.0;
      else state.matchReplay.speed = 1.0;
      speedBtn.innerText = `⚡ ${state.matchReplay.speed}x Speed`;
      if (state.matchReplay.isPlaying) {
        stopMatchReplay();
        startMatchReplay();
      }
    });
  }
}

function startMatchReplay() {
  state.matchReplay.isPlaying = true;
  const playBtn = document.getElementById("matchReplayPlayBtn");
  if (playBtn) playBtn.innerHTML = "<span>⏸</span> Pause Stream";

  const intervalMs = 1000 / state.matchReplay.speed;
  state.matchReplay.timer = setInterval(() => {
    if (!state.matchTimeline || !state.matchTimeline.windows) return;
    const curIdx = state.activeMatchWindow ? (state.activeMatchWindow.seq - 1) : 0;
    if (curIdx >= state.matchTimeline.windows.length - 1) {
      selectAndHighlightMatchWindow(0);
    } else {
      selectAndHighlightMatchWindow(curIdx + 1);
    }
  }, intervalMs);
}

function stopMatchReplay() {
  state.matchReplay.isPlaying = false;
  const playBtn = document.getElementById("matchReplayPlayBtn");
  if (playBtn) playBtn.innerHTML = "<span>▶</span> Auto-Play Match Stream";
  if (state.matchReplay.timer) clearInterval(state.matchReplay.timer);
}

function selectAndHighlightMatchWindow(index) {
  if (!state.matchTimeline || !state.matchTimeline.windows[index]) return;
  const win = state.matchTimeline.windows[index];
  const segs = document.querySelectorAll(".timeline-segment");
  segs.forEach(s => s.classList.remove("active-window"));
  if (segs[index]) {
    segs[index].classList.add("active-window");
    segs[index].scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }
  selectMatchWindow(win);
}

// ---------------------------------------------------------------------------
// 5. INTERACTIVE FORMATION SANDBOX
// ---------------------------------------------------------------------------

let sandboxCtx = null;
let sandboxThrottleTimer = null;

function setupSandboxCanvas() {
  const canvas = document.getElementById("sandboxCanvas");
  sandboxCtx = canvas.getContext("2d");

  // Mouse / Touch Interaction
  canvas.addEventListener("mousedown", handleSandboxPointerDown);
  canvas.addEventListener("mousemove", handleSandboxPointerMove);
  window.addEventListener("mouseup", handleSandboxPointerUp);

  canvas.addEventListener("touchstart", (e) => {
    const t = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    handleSandboxPointerDown({ clientX: t.clientX, clientY: t.clientY, target: canvas });
  });
  canvas.addEventListener("touchmove", (e) => {
    const t = e.touches[0];
    handleSandboxPointerMove({ clientX: t.clientX, clientY: t.clientY, target: canvas });
  });
  window.addEventListener("touchend", handleSandboxPointerUp);

  // Preset Buttons
  document.getElementById("sandboxAttackPresetBtn").addEventListener("click", () => {
    state.sandbox.ball = [1350, 180];
    state.sandbox.players = [
      [900, 720], [750, 640], [600, 620],
      [1020, 740], [1180, 640], [1320, 520] // Front attacker pushed forward
    ];
    renderSandboxCourt();
    triggerSandboxPrediction();
  });

  document.getElementById("sandboxDefensePresetBtn").addEventListener("click", () => {
    state.sandbox.ball = [1000, 300];
    state.sandbox.players = [
      [700, 700], [820, 700], [940, 700],
      [1060, 700], [1180, 700], [1300, 700] // Rigid defensive wall
    ];
    renderSandboxCourt();
    triggerSandboxPrediction();
  });

  document.getElementById("sandboxBreakdownPresetBtn").addEventListener("click", () => {
    state.sandbox.ball = [1100, 250];
    state.sandbox.players = [
      [450, 800], [500, 810], [550, 820], // Clustered in corner
      [1500, 800], [1550, 810], [1600, 820] // Dispersed opposite corner
    ];
    renderSandboxCourt();
    triggerSandboxPrediction();
  });

  const delayedBtn = document.getElementById("sandboxDelayedPresetBtn");
  if (delayedBtn) {
    delayedBtn.addEventListener("click", () => {
      state.sandbox.ball = [680, 410];
      state.sandbox.players = [
        [500, 780], [620, 810], [880, 820],
        [1120, 720], [1340, 680], [820, 590] // Scrambled dig formation
      ];
      playClickSound();
      renderSandboxCourt();
      triggerSandboxPrediction();
    });
  }

  // FIVB Auto-Rotate Button
  const rotateBtn = document.getElementById("sandboxRotateBtn");
  if (rotateBtn) {
    rotateBtn.addEventListener("click", () => {
      // Official FIVB clockwise rotational order:
      // P1 -> P6, P6 -> P5, P5 -> P4, P4 -> P3, P3 -> P2, P2 -> P1
      const p = state.sandbox.players;
      state.sandbox.players = [
        [p[1][0], p[1][1]],
        [p[2][0], p[2][1]],
        [p[3][0], p[3][1]],
        [p[4][0], p[4][1]],
        [p[5][0], p[5][1]],
        [p[0][0], p[0][1]]
      ];
      playWhistle();
      renderSandboxCourt();
      triggerSandboxPrediction();
    });
  }
}

function updateNodeInspector(idx, srcX, srcY) {
  const nodeEl = document.getElementById("inspNode");
  const zoneEl = document.getElementById("inspZone");
  const coordEl = document.getElementById("inspCoord");
  const netEl = document.getElementById("inspNetDist");
  if (!nodeEl || !zoneEl || !coordEl || !netEl) return;

  if (idx === 6) {
    nodeEl.innerText = "Volleyball (Target)";
    nodeEl.style.color = "#facc15";
  } else {
    nodeEl.innerText = `Player P${idx + 1} (${idx === 0 ? "Setter" : idx === 5 ? "Spiker" : srcY > 680 ? "Defender" : "Middle"})`;
    nodeEl.style.color = "var(--accent-blue)";
  }
  coordEl.innerText = `[${Math.round(srcX)}, ${Math.round(srcY)}]`;

  // Compute FIVB Zone
  let zoneName = "Midcourt";
  const isFront = srcY < 580;
  if (srcX < 850) {
    zoneName = isFront ? "Zone 4 (Front-Left)" : "Zone 5 (Back-Left)";
  } else if (srcX < 1250) {
    zoneName = isFront ? "Zone 3 (Front-Center)" : "Zone 6 (Back-Center)";
  } else {
    zoneName = isFront ? "Zone 2 (Front-Right)" : "Zone 1 (Back-Right)";
  }
  zoneEl.innerText = zoneName;

  const distMeters = Math.max(0, (srcY - 360) / 60).toFixed(2);
  netEl.innerText = `${distMeters} m`;
}

function handleSandboxPointerDown(e) {
  const canvas = document.getElementById("sandboxCanvas");
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  // Check collision with ball or 6 players
  const hitRadius = 18;

  // Check ball first
  const [bx, by] = mapCourtCoord(state.sandbox.ball[0], state.sandbox.ball[1], canvas.width, canvas.height);
  if (Math.hypot(mouseX - bx, mouseY - by) <= hitRadius) {
    state.sandbox.draggingIdx = 6;
    state.sandbox.isDragging = true;
    return;
  }

  // Check players 0..5
  for (let i = 0; i < 6; i++) {
    const p = state.sandbox.players[i];
    const [px, py] = mapCourtCoord(p[0], p[1], canvas.width, canvas.height);
    if (Math.hypot(mouseX - px, mouseY - py) <= hitRadius) {
      state.sandbox.draggingIdx = i;
      state.sandbox.isDragging = true;
      return;
    }
  }
}

function handleSandboxPointerMove(e) {
  if (!state.sandbox.isDragging) return;
  const canvas = document.getElementById("sandboxCanvas");
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  // Convert canvas mouseX/mouseY back to source coordinates
  const pad = 40;
  const cw = canvas.width - pad * 2;
  const ch = canvas.height - pad * 2;
  const minX = 400, maxX = 1700;
  const minY = 100, maxY = 860;

  const nx = (mouseX - pad) / cw;
  const ny = (mouseY - pad) / ch;

  const srcX = minX + Math.max(0, Math.min(1, nx)) * (maxX - minX);
  const srcY = minY + Math.max(0, Math.min(1, ny)) * (maxY - minY);

  if (state.sandbox.draggingIdx === 6) {
    state.sandbox.ball = [srcX, srcY];
    updateNodeInspector(6, srcX, srcY);
  } else if (state.sandbox.draggingIdx >= 0 && state.sandbox.draggingIdx < 6) {
    state.sandbox.players[state.sandbox.draggingIdx] = [srcX, srcY];
    updateNodeInspector(state.sandbox.draggingIdx, srcX, srcY);
  }

  renderSandboxCourt();

  // Throttled prediction
  if (sandboxThrottleTimer) clearTimeout(sandboxThrottleTimer);
  sandboxThrottleTimer = setTimeout(triggerSandboxPrediction, 120);
}

function handleSandboxPointerUp() {
  if (state.sandbox.isDragging) {
    if (state.sandbox.draggingIdx >= 0) {
      const isBall = state.sandbox.draggingIdx === 6;
      const pt = isBall ? state.sandbox.ball : state.sandbox.players[state.sandbox.draggingIdx];
      spawnSandboxRipple(pt[0], pt[1]);
    }
    state.sandbox.isDragging = false;
    state.sandbox.draggingIdx = -1;
    triggerSandboxPrediction();
  }
}

async function triggerSandboxPrediction() {
  try {
    const res = await fetch(apiUrl("/api/predict-custom-formation"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model_key: state.currentModel,
        ball: state.sandbox.ball,
        players: state.sandbox.players
      })
    });
    const data = await res.json();
    const pred = data.prediction;
    const label = pred.label;
    const color = pred.color || TACTIC_COLORS[label];

    const hero = document.getElementById("sandboxVerdictHero");
    hero.style.background = `${color}18`;
    hero.style.borderColor = `${color}66`;

    document.getElementById("sandboxLabel").innerText = label;
    document.getElementById("sandboxLabel").style.color = color;
    document.getElementById("sandboxCategory").innerText = pred.tactic_type;
    document.getElementById("sandboxCategory").style.color = color;

    const confPct = (pred.confidence * 100).toFixed(1);
    document.getElementById("sandboxConfVal").innerText = `${confPct}%`;
    document.getElementById("sandboxConfVal").style.color = color;
    document.getElementById("sandboxConfBar").style.width = `${confPct}%`;
    document.getElementById("sandboxConfBar").style.background = color;

    // GSAP bounce on prediction update
    if (typeof gsap !== "undefined") {
      gsap.fromTo(hero, { scale: 0.95 }, { scale: 1, duration: 0.35, ease: "back.out(1.5)" });
      gsap.to("#sandboxConfBar", { width: `${confPct}%`, duration: 0.5, ease: "power2.out" });
    }

    if (data.frames && data.frames[0]) {
      const geom = data.frames[0].geometry;
      document.getElementById("sandboxArea").innerText = `${geom.compactness} m²`;
      document.getElementById("sandboxSpread").innerText = `${geom.spread} m`;
    }
  } catch (err) {
    console.error("Failed to run sandbox prediction:", err);
  }
}

function spawnSandboxRipple(x, y) {
  if (!state.sandbox.ripples) state.sandbox.ripples = [];
  state.sandbox.ripples.push({ x, y, startTime: Date.now() });
}

function renderSandboxCourt() {
  if (!sandboxCtx) return;
  const canvas = sandboxCtx.canvas;
  const W = canvas.width;
  const H = canvas.height;

  sandboxCtx.clearRect(0, 0, W, H);

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";

  // Background
  sandboxCtx.fillStyle = isDark ? "#0c1626" : "#f1f5f9";
  sandboxCtx.fillRect(0, 0, W, H);

  const courtX = 40, courtY = 40, courtW = W - 80, courtH = H - 80;
  sandboxCtx.fillStyle = isDark ? "#14223c" : "#1d4ed8";
  sandboxCtx.fillRect(courtX, courtY, courtW, courtH);

  // Animated Holographic Radar Scan Line in Sandbox
  const scanProgress = (Date.now() / 2600) % 1;
  const scanX = courtX + scanProgress * courtW;
  const scanGrad = sandboxCtx.createLinearGradient(scanX - 40, 0, scanX + 40, 0);
  scanGrad.addColorStop(0, "rgba(56, 189, 248, 0)");
  scanGrad.addColorStop(0.5, isDark ? "rgba(56, 189, 248, 0.15)" : "rgba(255, 255, 255, 0.22)");
  scanGrad.addColorStop(1, "rgba(56, 189, 248, 0)");
  sandboxCtx.fillStyle = scanGrad;
  sandboxCtx.fillRect(scanX - 40, courtY, 80, courtH);

  // Shockwave ripples on node drop
  if (state.sandbox.ripples && state.sandbox.ripples.length > 0) {
    const now = Date.now();
    state.sandbox.ripples = state.sandbox.ripples.filter(r => now - r.startTime < 750);
    state.sandbox.ripples.forEach(r => {
      const progress = (now - r.startTime) / 750;
      const [rx, ry] = mapCourtCoord(r.x, r.y, W, H);
      sandboxCtx.beginPath();
      sandboxCtx.arc(rx, ry, progress * 45 + 10, 0, Math.PI * 2);
      sandboxCtx.strokeStyle = `rgba(56, 189, 248, ${1 - progress})`;
      sandboxCtx.lineWidth = 2.5 * (1 - progress);
      sandboxCtx.stroke();
    });
  }

  // Boundaries & Net
  sandboxCtx.strokeStyle = "rgba(255, 255, 255, 0.65)";
  sandboxCtx.lineWidth = 2.5;
  sandboxCtx.strokeRect(courtX, courtY, courtW, courtH);

  const netY = courtY + courtH * 0.35;
  sandboxCtx.strokeStyle = "#f43f5e";
  sandboxCtx.lineWidth = 3.5;
  sandboxCtx.beginPath();
  sandboxCtx.moveTo(courtX - 10, netY);
  sandboxCtx.lineTo(courtX + courtW + 10, netY);
  sandboxCtx.stroke();

  // Attack line
  const attackLineY = courtY + courtH * 0.60;
  sandboxCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
  sandboxCtx.lineWidth = 2;
  sandboxCtx.setLineDash([8, 8]);
  sandboxCtx.beginPath();
  sandboxCtx.moveTo(courtX, attackLineY);
  sandboxCtx.lineTo(courtX + courtW, attackLineY);
  sandboxCtx.stroke();
  sandboxCtx.setLineDash([]);

  // Mesh Lines between sandbox players
  sandboxCtx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  sandboxCtx.lineWidth = 1.5;
  for (let i = 0; i < 6; i++) {
    for (let j = i + 1; j < 6; j++) {
      const [x1, y1] = mapCourtCoord(state.sandbox.players[i][0], state.sandbox.players[i][1], W, H);
      const [x2, y2] = mapCourtCoord(state.sandbox.players[j][0], state.sandbox.players[j][1], W, H);
      const d = Math.hypot(x1 - x2, y1 - y2);
      if (d < 180) {
        sandboxCtx.beginPath();
        sandboxCtx.moveTo(x1, y1);
        sandboxCtx.lineTo(x2, y2);
        sandboxCtx.stroke();
      }
    }
  }

  // Draw 6 Players with pulsing aura
  state.sandbox.players.forEach((p, idx) => {
    const [px, py] = mapCourtCoord(p[0], p[1], W, H);

    const pulse = Math.sin(Date.now() / 220 + idx) * 3;
    sandboxCtx.beginPath();
    sandboxCtx.arc(px, py, 15 + pulse, 0, Math.PI * 2);
    sandboxCtx.fillStyle = state.sandbox.draggingIdx === idx ? "rgba(56, 189, 248, 0.45)" : "rgba(2, 132, 199, 0.25)";
    sandboxCtx.fill();

    sandboxCtx.beginPath();
    sandboxCtx.arc(px, py, 13, 0, Math.PI * 2);
    sandboxCtx.fillStyle = state.sandbox.draggingIdx === idx ? "#38bdf8" : "#0284c7";
    sandboxCtx.fill();
    sandboxCtx.strokeStyle = "#ffffff";
    sandboxCtx.lineWidth = 2.5;
    sandboxCtx.stroke();

    sandboxCtx.fillStyle = "#ffffff";
    sandboxCtx.font = "bold 11px Outfit, sans-serif";
    sandboxCtx.textAlign = "center";
    sandboxCtx.textBaseline = "middle";
    sandboxCtx.fillText(`P${idx + 1}`, px, py);
  });

  // Draw Ball with glowing ripple
  const [bx, by] = mapCourtCoord(state.sandbox.ball[0], state.sandbox.ball[1], W, H);
  const ballPulse = Math.sin(Date.now() / 150) * 3.5;
  sandboxCtx.beginPath();
  sandboxCtx.arc(bx, by, 14 + ballPulse, 0, Math.PI * 2);
  sandboxCtx.fillStyle = "rgba(234, 179, 8, 0.38)";
  sandboxCtx.fill();

  sandboxCtx.beginPath();
  sandboxCtx.arc(bx, by, 8, 0, Math.PI * 2);
  sandboxCtx.fillStyle = "#facc15";
  sandboxCtx.fill();
  sandboxCtx.strokeStyle = "#ffffff";
  sandboxCtx.lineWidth = 2;
  sandboxCtx.stroke();

  sandboxCtx.fillStyle = "#facc15";
  sandboxCtx.font = "bold 10px Inter, sans-serif";
  sandboxCtx.textAlign = "left";
  sandboxCtx.fillText("🏐 Ball (Drag)", bx + 12, by + 4);
}

// ---------------------------------------------------------------------------
// 6. MODEL LEADERBOARD POPULATION
// ---------------------------------------------------------------------------

function populateLeaderboard(data) {
  const tbody = document.getElementById("leaderboardTbody");
  tbody.innerHTML = "";

  data.forEach((item) => {
    const tr = document.createElement("tr");
    const isRank1 = item.rank === 1;
    tr.innerHTML = `
      <td><span class="model-rank-badge ${isRank1 ? 'gold-shimmer-badge' : ''}" style="font-size: 0.8rem;">#${item.rank}${isRank1 ? ' 🏆' : ''}</span></td>
      <td><strong>${item.model}</strong></td>
      <td><span style="color: ${item.family.includes('Deep') ? '#a855f7' : '#06b6d4'}; font-weight: 600;">${item.family}</span></td>
      <td><strong style="color: var(--text-primary); font-size: 0.95rem;">${item.test_accuracy}%</strong></td>
      <td>${item.test_macro_f1}</td>
      <td>${item.train_accuracy}%</td>
      <td>${item.f1_attack !== null ? item.f1_attack : '-'}</td>
      <td>${item.f1_defense !== null ? item.f1_defense : '-'}</td>
      <td>${item.f1_delayed !== null ? item.f1_delayed : '-'}</td>
      <td>${item.f1_spacing !== null ? item.f1_spacing : '-'}</td>
    `;
    tbody.appendChild(tr);
  });

  // GSAP staggered slide-in for leaderboard rows
  if (typeof gsap !== "undefined") {
    gsap.from("#leaderboardTbody tr", {
      opacity: 0,
      x: -20,
      stagger: 0.035,
      duration: 0.4,
      ease: "power2.out"
    });
  }
}

// ---------------------------------------------------------------------------
// 7. CSV FILE UPLOAD HANDLER
// ---------------------------------------------------------------------------

function setupUploadDropzone() {
  const dropzone = document.getElementById("csvDropzone");
  const fileInput = document.getElementById("csvFileInput");

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please upload a valid .csv file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_key", state.currentModel);

  try {
    const res = await fetch(apiUrl("/api/upload-csv"), {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }

    const data = await res.json();
    state.currentPrediction = data;
    state.currentSampleId = null;
    state.currentFrameIdx = 0;

    // Switch to Radar tab and show results
    document.getElementById("navRadarBtn").click();
    updateVerdictUI(data);
    updatePlaybackControls(data);
    renderCourt();
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// 8. RAW MATCH VIDEO UPLOAD & TACTICAL ANALYSIS
// ---------------------------------------------------------------------------

function setupVideoUpload() {
  const dropzone = document.getElementById("videoDropzone");
  const fileInput = document.getElementById("videoFileInput");
  const sampleBtn = document.getElementById("trySampleVideoBtn");

  if (dropzone && fileInput) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length > 0) {
        handleVideoUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleVideoUpload(e.target.files[0]);
      }
    });
  }

  if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
      analyzeSampleVideo("coord_attack_demo");
    });
  }
}

async function handleVideoUpload(file) {
  const validExts = [".mp4", ".mov", ".avi", ".webm", ".mkv"];
  const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
  if (!validExts.includes(ext)) {
    alert("Please upload a supported video format (.mp4, .mov, .avi, .webm).");
    return;
  }

  const modelKey = document.getElementById("videoModelSelect").value;
  const maxDuration = document.getElementById("videoDurationSelect").value;

  const progressBox = document.getElementById("videoProcessingProgress");
  const statusMsg = document.getElementById("videoStatusMessage");
  const outputPlaceholder = document.getElementById("videoOutputPlaceholder");
  const outputContainer = document.getElementById("videoOutputContainer");

  progressBox.style.display = "block";
  outputPlaceholder.style.display = "flex";
  outputContainer.style.display = "none";
  statusMsg.innerText = `Uploading ${file.name} and initializing YOLO detection + tactical model...`;

  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_key", modelKey);
  formData.append("max_duration_sec", maxDuration);

  try {
    const res = await fetch(apiUrl("/api/upload-video"), {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Video analysis failed");
    }

    const data = await res.json();
    displayVideoResults(data);
  } catch (err) {
    alert(`Video Processing Error: ${err.message}`);
  } finally {
    progressBox.style.display = "none";
  }
}

async function analyzeSampleVideo(sampleId) {
  const progressBox = document.getElementById("videoProcessingProgress");
  const statusMsg = document.getElementById("videoStatusMessage");
  const outputPlaceholder = document.getElementById("videoOutputPlaceholder");
  const outputContainer = document.getElementById("videoOutputContainer");

  progressBox.style.display = "block";
  outputPlaceholder.style.display = "flex";
  outputContainer.style.display = "none";
  statusMsg.innerText = "Analyzing rally footage (coordAttack.mp4) with HistGradientBoosting...";

  const modelKey = document.getElementById("videoModelSelect").value;
  const maxDuration = document.getElementById("videoDurationSelect").value;

  const formData = new FormData();
  formData.append("sample_id", sampleId);
  formData.append("model_key", modelKey);
  formData.append("max_duration_sec", maxDuration);

  try {
    const res = await fetch(apiUrl("/api/process-sample-video"), {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Sample video processing failed");
    }

    const data = await res.json();
    displayVideoResults(data);
  } catch (err) {
    alert(`Error: ${err.message}`);
  } finally {
    progressBox.style.display = "none";
  }
}

function displayVideoResults(data) {
  const outputPlaceholder = document.getElementById("videoOutputPlaceholder");
  const outputContainer = document.getElementById("videoOutputContainer");
  const player = document.getElementById("annotatedVideoPlayer");
  const dominantEl = document.getElementById("videoDominantTactic");
  const downloadBtn = document.getElementById("videoDownloadBtn");
  const durationVal = document.getElementById("videoDurationVal");
  const modelVal = document.getElementById("videoModelVal");
  const tbody = document.getElementById("videoWindowsTbody");

  outputPlaceholder.style.display = "none";
  outputContainer.style.display = "block";

  // Set video source and play
  player.src = data.output_video_url;
  player.load();
  player.play().catch(() => {});

  // Set dominant tactic
  dominantEl.innerText = data.dominant_tactic;
  dominantEl.style.color = data.dominant_color || TACTIC_COLORS[data.dominant_tactic] || "#f59e0b";

  // Set download button
  downloadBtn.href = data.output_video_url;
  downloadBtn.setAttribute("download", data.output_filename || "annotated_tactical_video.mp4");

  // Metadata
  durationVal.innerText = `${data.duration_sec}s (${data.processed_frames} frames) • ${data.fps} FPS`;
  modelVal.innerText = `${data.model_used} (${data.processing_time_sec}s analysis)`;

  // Keep class exploration inside the same result surface as the full video.
  const summary = data.analysis_summary || {};
  const playersEl = document.getElementById("playersTrackedVal");
  const coverageEl = document.getElementById("trackingCoverageVal");
  const movementEl = document.getElementById("teamMovementVal");
  const ralliesEl = document.getElementById("classifiedRalliesVal");
  if (playersEl) playersEl.innerText = summary.players_tracked || "6 / 6";
  if (coverageEl) coverageEl.innerText = summary.tracking_coverage || "Ready";
  if (movementEl) movementEl.innerText = summary.team_movement || "Formation tracked";
  if (ralliesEl) ralliesEl.innerText = `${(data.windows || []).length}`;
  renderClassClipCards(data.windows || [], player);

  // Populate timeline table
  tbody.innerHTML = "";
  if (data.windows && data.windows.length > 0) {
    data.windows.forEach(w => {
      const tr = document.createElement("tr");
      tr.className = "clickable-row";
      tr.style.cursor = "pointer";
      tr.title = `Click to seek video to ${w.second}s`;
      const wColor = w.color || TACTIC_COLORS[w.tactic] || "#64748b";
      tr.innerHTML = `
        <td><strong>${w.second}s</strong></td>
        <td><span style="color: ${wColor}; font-weight: 700;">${w.tactic}</span></td>
        <td>${(w.confidence * 100).toFixed(1)}%</td>
        <td><span style="font-size: 0.72rem; padding: 0.15rem 0.45rem; border-radius: 999px; background: ${wColor}22; color: ${wColor}; font-weight: 700;">VERIFIED</span></td>
      `;

      // Click to seek video player to that exact second!
      tr.addEventListener("click", () => {
        player.currentTime = w.second;
        player.play().catch(() => {});
        document.querySelectorAll("#videoWindowsTbody tr").forEach(r => r.style.background = "transparent");
        tr.style.background = `${wColor}18`;
      });

      tbody.appendChild(tr);
    });
  } else {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Sequence classified as ${data.dominant_tactic}</td></tr>`;
  }

  // GSAP animation for video output container and detection timeline
  if (typeof gsap !== "undefined") {
    gsap.from(outputContainer, {
      opacity: 0,
      y: 20,
      scale: 0.98,
      duration: 0.45,
      ease: "power2.out"
    });
    gsap.from("#videoWindowsTbody tr", {
      opacity: 0,
      x: -15,
      stagger: 0.04,
      duration: 0.35,
      ease: "power1.out"
    });
  }
}

function renderClassClipCards(windows, player) {
  const grid = document.getElementById("classClipGrid");
  if (!grid) return;
  const classes = [
    { name: "Coordinated Attack", color: "#f59e0b" },
    { name: "Coordinated Defense", color: "#10b981" },
    { name: "Delayed Support", color: "#ec4899" },
    { name: "Spacing Breakdown", color: "#a855f7" }
  ];
  const grouped = new Map(classes.map(item => [item.name, []]));
  windows.forEach(window => {
    if (!grouped.has(window.tactic)) grouped.set(window.tactic, []);
    grouped.get(window.tactic).push(window);
  });
  grid.innerHTML = classes.map(item => {
    const clips = grouped.get(item.name) || [];
    const first = clips[0];
    const label = clips.length ? `${clips.length} clip${clips.length === 1 ? "" : "s"}` : "No clip detected";
    return `<div class="class-clip-card" style="border-left: 3px solid ${item.color}">
      <div><div class="class-clip-name" style="color:${item.color}">${item.name}</div>
      <div class="class-clip-meta">${label}${first ? ` · first at ${first.second}s · ${(first.confidence * 100).toFixed(0)}% confidence` : ""}</div></div>
      ${first ? `<button class="class-clip-jump" data-second="${first.second}">View clip ↗</button>` : ""}
    </div>`;
  }).join("");
  grid.querySelectorAll(".class-clip-jump").forEach(button => {
    button.addEventListener("click", () => {
      player.currentTime = Number(button.dataset.second) || 0;
      player.play().catch(() => {});
    });
  });
}

// ---------------------------------------------------------------------------
// 9. SOUND EFFECTS, MODEL DUEL & REPORT EXPORTER
// ---------------------------------------------------------------------------

function setupSoundToggle() {
  const btn = document.getElementById("soundToggleBtn");
  const icon = document.getElementById("soundIcon");
  const text = document.getElementById("soundText");

  function updateSoundUI() {
    if (icon && text) {
      icon.innerText = state.audioEnabled ? "🔊" : "🔇";
      text.innerText = state.audioEnabled ? "Audio: On" : "Audio: Muted";
    }
  }
  updateSoundUI();

  if (btn) {
    btn.addEventListener("click", () => {
      state.audioEnabled = !state.audioEnabled;
      localStorage.setItem("volleytactix_audio", state.audioEnabled);
      updateSoundUI();
      if (state.audioEnabled) playClickSound();
    });
  }
}

function setupReportExporter() {
  const exportBtn = document.getElementById("exportReportBtn");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!state.currentPrediction) {
        alert("Please load a rally sample first.");
        return;
      }
      playClickSound();

      const reportData = {
        title: "VolleyTactix AI - Automated Tactical Coordination Intelligence Report",
        generation_time: new Date().toISOString(),
        clip_sample_id: state.currentSampleId,
        active_model: state.currentPrediction.model_name,
        active_model_rank: state.currentPrediction.model_rank,
        tactical_verdict: {
          label: state.currentPrediction.prediction.label,
          tactic_type: state.currentPrediction.prediction.tactic_type,
          confidence: state.currentPrediction.prediction.confidence,
          entropy_bits: state.currentPrediction.prediction.entropy,
          deployment_validity: !state.currentPrediction.prediction.is_anomaly,
          class_probabilities: state.currentPrediction.prediction.probabilities
        },
        spatial_telemetry: {
          total_frames: state.currentPrediction.n_frames,
          current_frame_geometry: state.currentPrediction.frames ? state.currentPrediction.frames[state.currentFrameIdx].geometry : null
        }
      };

      const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `volleytactix_report_${state.currentSampleId || 'rally'}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }
}

function setupModelDuel() {
  const runBtn = document.getElementById("runDuelBtn");
  if (runBtn) {
    runBtn.addEventListener("click", runModelDuel);
  }
}

async function runModelDuel() {
  if (!state.currentSampleId) return;
  const modelAKey = document.getElementById("duelModelA").value;
  const modelBKey = document.getElementById("duelModelB").value;
  const container = document.getElementById("duelResultsContainer");
  if (!container) return;

  container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 1rem; color: var(--text-muted);">Comparing Model A vs Model B on current sequence...</div>`;

  try {
    const [resA, resB] = await Promise.all([
      fetch(apiUrl(`/api/sample/${state.currentSampleId}?model_key=${modelAKey}`)).then(r => r.json()),
      fetch(apiUrl(`/api/sample/${state.currentSampleId}?model_key=${modelBKey}`)).then(r => r.json())
    ]);

    const predA = resA.prediction;
    const predB = resB.prediction;
    const agree = predA.label === predB.label;
    const colorA = predA.color || TACTIC_COLORS[predA.label];
    const colorB = predB.color || TACTIC_COLORS[predB.label];

    container.innerHTML = `
      <div class="duel-side-card">
        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin-bottom: 0.35rem;">Model A: ${resA.model_name}</div>
        <div style="font-family: 'Outfit'; font-size: 1.25rem; font-weight: 800; color: ${colorA};">${predA.label}</div>
        <div class="duel-metric-row">
          <span>Confidence</span>
          <strong style="color: ${colorA};">${(predA.confidence * 100).toFixed(1)}%</strong>
        </div>
        <div class="duel-metric-row">
          <span>Entropy</span>
          <span>${predA.entropy} bits</span>
        </div>
        <div class="duel-metric-row">
          <span>Architecture</span>
          <span>${resA.model_type}</span>
        </div>
      </div>

      <div class="duel-side-card">
        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: var(--text-muted); margin-bottom: 0.35rem;">Model B: ${resB.model_name}</div>
        <div style="font-family: 'Outfit'; font-size: 1.25rem; font-weight: 800; color: ${colorB};">${predB.label}</div>
        <div class="duel-metric-row">
          <span>Confidence</span>
          <strong style="color: ${colorB};">${(predB.confidence * 100).toFixed(1)}%</strong>
        </div>
        <div class="duel-metric-row">
          <span>Entropy</span>
          <span>${predB.entropy} bits</span>
        </div>
        <div class="duel-metric-row">
          <span>Architecture</span>
          <span>${resB.model_type}</span>
        </div>
      </div>

      <div style="grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center; background: var(--bg-secondary); padding: 0.65rem 1rem; border-radius: var(--radius-sm); border: 1px solid var(--border-glass); margin-top: 0.5rem;">
        <span style="font-size: 0.85rem; font-weight: 600;">Inter-Model Consensus Decision:</span>
        <span class="duel-winner-tag" style="background: ${agree ? 'rgba(5, 150, 105, 0.15)' : 'rgba(219, 39, 119, 0.15)'}; color: ${agree ? 'var(--defense-color)' : 'var(--delayed-color)'};">
          ${agree ? '✅ Consensus Match (Both Agree)' : '⚠️ Divergent Predictions (Tactical Disagreement)'}
        </span>
      </div>
    `;
    playTacticChime();

    if (typeof gsap !== "undefined") {
      gsap.from("#duelResultsContainer .duel-side-card", {
        opacity: 0,
        y: 18,
        scale: 0.95,
        duration: 0.4,
        stagger: 0.1,
        ease: "back.out(1.2)"
      });
    }
  } catch (err) {
    container.innerHTML = `<div style="grid-column: 1 / -1; color: #ef4444;">Failed to execute duel comparison: ${err.message}</div>`;
  }
}

// ---------------------------------------------------------------------------
// 10. LENIS SMOOTH SCROLLING, VANTA.JS 3D BACKGROUND & GSAP ANIMATIONS
// ---------------------------------------------------------------------------

let lenis = null;

function initLenis() {
  if (typeof Lenis !== "undefined") {
    try {
      lenis = new Lenis({
        duration: 1.1,
        easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: true,
        touchMultiplier: 1.8
      });

      if (typeof ScrollTrigger !== "undefined" && typeof gsap !== "undefined") {
        lenis.on('scroll', ScrollTrigger.update);
        gsap.ticker.add((time) => {
          lenis.raf(time * 1000);
        });
        gsap.ticker.lagSmoothing(0);
      } else {
        function raf(time) {
          lenis.raf(time);
          requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);
      }
    } catch (e) {
      console.warn("Lenis initialization skipped:", e);
    }
  }
}

function smoothScrollTo(targetSelector, offset = -70) {
  const target = document.querySelector(targetSelector);
  if (!target) return;
  if (lenis) {
    lenis.scrollTo(target, { offset, duration: 1.2 });
  } else {
    target.scrollIntoView({ behavior: "smooth" });
  }
}

let vantaEffect = null;

function initVanta(retries = 10) {
  const el = document.getElementById("vantaBackground");
  if (!el) return;

  if (typeof VANTA === "undefined" || typeof THREE === "undefined") {
    if (retries > 0) {
      setTimeout(() => initVanta(retries - 1), 250);
    }
    return;
  }

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";

  try {
    if (vantaEffect && vantaEffect.destroy) {
      vantaEffect.destroy();
    }
    vantaEffect = VANTA.NET({
      el: el,
      mouseControls: true,
      touchControls: true,
      gyroControls: false,
      minHeight: 200.0,
      minWidth: 200.0,
      scale: 1.0,
      scaleMobile: 1.0,
      color: isDark ? 0x38bdf8 : 0x0284c7,
      backgroundColor: isDark ? 0x090d16 : 0xf8fafc,
      points: 12.0,
      maxDistance: 24.0,
      spacing: 15.0,
      showDots: true
    });
  } catch (err) {
    console.warn("Vanta.js Net skipped:", err);
  }
}

function updateVantaTheme(theme) {
  if (vantaEffect && vantaEffect.setOptions) {
    const isDark = theme === "dark";
    vantaEffect.setOptions({
      color: isDark ? 0x38bdf8 : 0x0284c7,
      backgroundColor: isDark ? 0x090d16 : 0xf8fafc
    });
  }
}

// ---------------------------------------------------------------------------
// DYNAMIC HERO TACTICAL CONSTELLATION & MESH ANIMATION
// ---------------------------------------------------------------------------
function initHeroConstellation() {
  const canvas = document.getElementById("heroConstellationCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const hero = document.getElementById("heroLanding");

  function resize() {
    canvas.width = hero ? hero.clientWidth : window.innerWidth;
    canvas.height = hero ? hero.clientHeight : window.innerHeight * 0.85;
  }
  resize();
  window.addEventListener("resize", resize);

  const numNodes = Math.min(80, Math.floor((canvas.width * canvas.height) / 11000));
  const nodes = [];
  for (let i = 0; i < numNodes; i++) {
    nodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.65,
      vy: (Math.random() - 0.5) * 0.65,
      radius: Math.random() * 2.2 + 1.2,
      baseRadius: Math.random() * 2.2 + 1.2,
      pulse: Math.random() * Math.PI * 2,
      isSpecial: i % 7 === 0
    });
  }

  let mouse = { x: -9999, y: -9999, radius: 160 };
  if (hero) {
    hero.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    });
    hero.addEventListener("mouseleave", () => {
      mouse.x = -9999;
      mouse.y = -9999;
    });
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";

    const nodeColor = isDark ? "rgba(56, 189, 248, " : "rgba(2, 132, 199, ";
    const specialColor = "rgba(245, 158, 11, ";
    const lineColor = isDark ? "56, 189, 248" : "2, 132, 199";

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      n.x += n.vx;
      n.y += n.vy;
      n.pulse += 0.035;

      if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
      if (n.y < 0 || n.y > canvas.height) n.vy *= -1;

      // Magnetic mouse interaction
      const dx = mouse.x - n.x;
      const dy = mouse.y - n.y;
      const dist = Math.hypot(dx, dy);
      if (dist < mouse.radius) {
        const force = (1 - dist / mouse.radius) * 2.2;
        n.x -= (dx / dist) * force;
        n.y -= (dy / dist) * force;
      }

      // Draw tactical mesh connections
      for (let j = i + 1; j < nodes.length; j++) {
        const n2 = nodes[j];
        const distNodes = Math.hypot(n.x - n2.x, n.y - n2.y);
        if (distNodes < 120) {
          const alpha = (1 - distNodes / 120) * (isDark ? 0.38 : 0.25);
          ctx.beginPath();
          ctx.moveTo(n.x, n.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.strokeStyle = `rgba(${lineColor}, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Draw tactical node
      const pulseSize = n.baseRadius + Math.sin(n.pulse) * 0.8;
      ctx.beginPath();
      ctx.arc(n.x, n.y, pulseSize, 0, Math.PI * 2);
      ctx.fillStyle = n.isSpecial ? `${specialColor}0.95)` : `${nodeColor}0.9)`;
      ctx.fill();

      // Glowing aura
      if (n.isSpecial) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, pulseSize * 3.5, 0, Math.PI * 2);
        ctx.fillStyle = `${specialColor}0.2)`;
        ctx.fill();
      }
    }

    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
}

function initGSAPAnimations() {
  if (typeof gsap === "undefined") return;

  try {
    // 1. Hero Entrance Timeline (enhanced with elastic eases)
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.from(".hero-pill-badge", { y: -30, opacity: 0, scale: 0.85, duration: 0.8, delay: 0.1 })
      .from(".hero-title", { y: 45, opacity: 0, skewX: -3, duration: 0.95 }, "-=0.5")
      .from(".hero-subtitle", { y: 30, opacity: 0, duration: 0.85 }, "-=0.65")
      .from(".hero-actions > *", { y: 25, opacity: 0, stagger: 0.08, duration: 0.7 }, "-=0.55")
      .from(".hero-stat-card", {
        y: 40, opacity: 0, scale: 0.9, stagger: 0.1,
        duration: 0.7, ease: "back.out(1.4)"
      }, "-=0.4");

    // 2. Animated Number Counters
    setTimeout(() => {
      animateNumberCounter("#heroStatModels", 13, 1.4, "", "");
      animateNumberCounter("#heroStatClips", 945, 1.8, "", "");
      animateNumberCounter("#heroStatLatency", 18, 1.4, "< ", "ms");
      animateNumberCounter("#heroStatAgreement", 83.8, 1.6, "", "%", 1);
    }, 600);

    // 3. Brand logo spin-in
    gsap.from(".brand-logo", {
      rotateY: 360, scale: 0.5, opacity: 0,
      duration: 1.1, ease: "elastic.out(1, 0.5)", delay: 0.1
    });

    // 4. Header actions slide in from right
    gsap.from(".header-actions > *", {
      x: 20, opacity: 0, stagger: 0.06, duration: 0.5, ease: "power2.out", delay: 0.3
    });

    // 5. Feature Showcase Cards Entrance via ScrollTrigger
    if (typeof ScrollTrigger !== "undefined") {
      gsap.registerPlugin(ScrollTrigger);
      gsap.from(".feature-card", {
        scrollTrigger: {
          trigger: ".features-grid",
          start: "top 82%"
        },
        y: 50, opacity: 0, scale: 0.94,
        duration: 0.7, stagger: 0.1, ease: "power2.out"
      });

      // Section tag animation
      gsap.from(".section-tag, .section-heading", {
        scrollTrigger: { trigger: ".features-section", start: "top 88%" },
        y: 20, opacity: 0, stagger: 0.15, duration: 0.65, ease: "power2.out"
      });
    }
  } catch (err) {
    console.warn("GSAP animations skipped:", err);
  }
}

function animateNumberCounter(selector, endVal, duration, prefix = "", suffix = "", decimals = 0) {
  const el = document.querySelector(selector);
  if (!el || typeof gsap === "undefined") return;
  const obj = { val: 0 };
  gsap.to(obj, {
    val: endVal,
    duration: duration,
    ease: "power2.out",
    onUpdate: () => {
      el.innerText = `${prefix}${decimals > 0 ? obj.val.toFixed(decimals) : Math.round(obj.val)}${suffix}`;
    }
  });
}

function setupLandingHeroInteractions() {
  // Hero CTAs
  const launchBtn = document.getElementById("heroLaunchBtn");
  if (launchBtn) {
    launchBtn.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#workspaceSection");
      const videoTab = document.getElementById("navVideoBtn");
      if (videoTab) videoTab.click();
    });
  }

  const videoBtn = document.getElementById("heroVideoBtn");
  if (videoBtn) {
    videoBtn.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#workspaceSection");
      const vTab = document.getElementById("navVideoBtn");
      if (vTab) vTab.click();
    });
  }

  const consensusBtn = document.getElementById("heroConsensusBtn");
  if (consensusBtn) {
    consensusBtn.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#workspaceSection");
      const videoTab = document.getElementById("navVideoBtn");
      if (videoTab) videoTab.click();
    });
  }

  const sandboxBtn = document.getElementById("heroSandboxBtn");
  if (sandboxBtn) {
    sandboxBtn.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#workspaceSection");
      const sTab = document.getElementById("navSandboxBtn");
      if (sTab) sTab.click();
    });
  }

  // Header Back to Overview Links
  const brandHome = document.getElementById("brandHomeLink");
  if (brandHome) {
    brandHome.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#heroLanding", -100);
    });
  }

  const navHero = document.getElementById("navHeroBtn");
  if (navHero) {
    navHero.addEventListener("click", () => {
      playClickSound();
      smoothScrollTo("#heroLanding", -100);
    });
  }

  // Feature Cards Click-to-Jump
  document.querySelectorAll(".feature-card").forEach(card => {
    card.addEventListener("click", () => {
      const tabId = card.dataset.jump;
      if (!tabId) return;
      playClickSound();
      smoothScrollTo("#workspaceSection");
      const targetNav = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
      if (targetNav) targetNav.click();
    });
  });
}

// ---------------------------------------------------------------------------
// 11. AMBIENT ANIMATION LOOP & INTERACTIVE 3D TILT
// ---------------------------------------------------------------------------

let ambientAnimationTimer = null;

function startAmbientCanvasLoop() {
  if (ambientAnimationTimer) return;
  function tick() {
    // Re-render radar canvas for continuous beam sweep & glowing halos when paused
    if (!state.isPlaying && state.currentPrediction) {
      const radarPane = document.getElementById("tab-radar");
      if (radarPane && radarPane.classList.contains("active")) {
        renderCourt();
      }
    }
    // Re-render sandbox canvas for shockwave ripples & scanning lines
    const sandboxPane = document.getElementById("tab-sandbox");
    if (sandboxPane && sandboxPane.classList.contains("active") && !state.sandbox.isDragging) {
      renderSandboxCourt();
    }
    ambientAnimationTimer = requestAnimationFrame(tick);
  }
  ambientAnimationTimer = requestAnimationFrame(tick);
}

function setup3DCardTilt() {
  const cards = document.querySelectorAll(".feature-card");
  cards.forEach(card => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = ((y - centerY) / centerY) * -6;
      const rotateY = ((x - centerX) / centerX) * 6;

      if (typeof gsap !== "undefined") {
        gsap.to(card, {
          rotateX,
          rotateY,
          transformPerspective: 800,
          scale: 1.02,
          duration: 0.25,
          ease: "power1.out"
        });
      }
    });

    card.addEventListener("mouseleave", () => {
      if (typeof gsap !== "undefined") {
        gsap.to(card, {
          rotateX: 0,
          rotateY: 0,
          scale: 1,
          duration: 0.45,
          ease: "power2.out"
        });
      }
    });
  });
}

// ---------------------------------------------------------------------------
// 12. PRODUCT SHOWCASE & DOWNLOAD MODAL CONTROLLER
// ---------------------------------------------------------------------------

function setupProductShowcaseModal() {
  const modal = document.getElementById("productShowcaseModal");
  const closeBtn = document.getElementById("modalCloseBtn");
  const headerBtn = document.getElementById("headerDownloadBtn");
  const heroBtn = document.getElementById("heroDownloadBtn");
  const floatBtn = document.getElementById("floatingDownloadBtn");
  const prevBtn = document.getElementById("galleryPrevBtn");
  const nextBtn = document.getElementById("galleryNextBtn");
  const carousel = document.getElementById("modalPhotoCarousel");
  const downloadZipBtn = document.getElementById("modalDownloadMainBtn");
  const quickExportBtn = document.getElementById("modalQuickExportBtn");

  function openModal() {
    if (!modal) return;
    playClickSound();
    modal.classList.add("active");
    modal.setAttribute("aria-hidden", "false");

    // Ensure download button is immediately 100% visible
    const dlBtn = document.getElementById("modalDownloadMainBtn");
    if (dlBtn) {
      dlBtn.style.opacity = "1";
      dlBtn.style.visibility = "visible";
    }

    // Modal card entrance
    if (typeof gsap !== "undefined") {
      gsap.fromTo(".product-modal-card",
        { scale: 0.95, y: 15 },
        { scale: 1, y: 0, duration: 0.35, ease: "back.out(1.2)" }
      );
    }
  }

  function closeModal() {
    if (!modal) return;
    playClickSound();
    modal.classList.remove("active");
    modal.setAttribute("aria-hidden", "true");
  }

  // Open triggers
  if (headerBtn) headerBtn.addEventListener("click", openModal);
  if (heroBtn) heroBtn.addEventListener("click", openModal);
  if (floatBtn) floatBtn.addEventListener("click", openModal);

  // Close triggers
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.classList.contains("active")) {
      closeModal();
    }
  });

  // Carousel Horizontal Scroll Controls
  if (carousel && prevBtn && nextBtn) {
    const scrollStep = 290;
    prevBtn.addEventListener("click", () => {
      playClickSound();
      carousel.scrollBy({ left: -scrollStep, behavior: "smooth" });
    });

    nextBtn.addEventListener("click", () => {
      playClickSound();
      carousel.scrollBy({ left: scrollStep, behavior: "smooth" });
    });
  }

  // Download Main Action (ZIP)
  if (downloadZipBtn) {
    downloadZipBtn.addEventListener("click", () => {
      playTacticChime();
      const badge = downloadZipBtn.querySelector(".dl-badge-chip");
      if (badge) {
        badge.innerText = "✓ DOWNLOADING...";
        badge.style.background = "var(--defense-color)";
        setTimeout(() => {
          badge.innerText = "FREE • 1.2 MB";
          badge.style.background = "rgba(255, 255, 255, 0.2)";
        }, 2500);
      }
    });
  }

  // Quick JSON Report Export Action
  if (quickExportBtn) {
    quickExportBtn.addEventListener("click", () => {
      playClickSound();
      const exportReportBtn = document.getElementById("exportReportBtn");
      if (exportReportBtn) {
        exportReportBtn.click();
      } else {
      window.location.href = apiUrl("/api/download-suite");
    });
  }

  // Set modal download button URL dynamically
  const modalDownloadMainBtn = document.getElementById("modalDownloadMainBtn");
  if (modalDownloadMainBtn) {
    modalDownloadMainBtn.href = apiUrl("/api/download-suite");
  }

  // System Status Badge: Click to configure Render Backend URL
  const statusBadge = document.getElementById("systemStatusBadge");
  if (statusBadge) {
    const currentBackend = getApiBaseUrl();
    const statusText = document.getElementById("systemStatusText");
    if (statusText) {
      if (currentBackend) {
        statusText.innerText = "Render API Connected";
      } else {
        statusText.innerText = "Pipeline Online";
      }
    }
    statusBadge.addEventListener("click", () => {
      const current = localStorage.getItem(RENDER_BACKEND_KEY) || "";
      const input = prompt(
        "VolleyTactix AI Backend API URL:\n\nLeave empty to use current domain/localhost, or enter your Render backend URL (e.g., https://volleyball-tactical-intelligence-backend.onrender.com):",
        current
      );
      if (input !== null) {
        const val = input.trim();
        if (val) {
          localStorage.setItem(RENDER_BACKEND_KEY, val);
        } else {
          localStorage.removeItem(RENDER_BACKEND_KEY);
        }
        window.location.reload();
      }
    });
  }
}


// ===========================================================================
// 13. AMBIENT ORB BACKGROUND CANVAS
// ===========================================================================

function initAmbientOrbCanvas() {
  const canvas = document.getElementById("animOrbCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  const orbs = Array.from({ length: 6 }, (_, i) => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: 180 + Math.random() * 220,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    hue: [210, 240, 270, 200, 220, 260][i],
    alpha: 0.06 + Math.random() * 0.08
  }));

  function drawOrbs() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    orbs.forEach(orb => {
      orb.x += orb.vx;
      orb.y += orb.vy;
      if (orb.x < -orb.r) orb.x = canvas.width + orb.r;
      if (orb.x > canvas.width + orb.r) orb.x = -orb.r;
      if (orb.y < -orb.r) orb.y = canvas.height + orb.r;
      if (orb.y > canvas.height + orb.r) orb.y = -orb.r;

      const grad = ctx.createRadialGradient(orb.x, orb.y, 0, orb.x, orb.y, orb.r);
      grad.addColorStop(0, `hsla(${orb.hue}, 80%, 60%, ${orb.alpha})`);
      grad.addColorStop(1, `hsla(${orb.hue}, 60%, 40%, 0)`);
      ctx.beginPath();
      ctx.arc(orb.x, orb.y, orb.r, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    });
    requestAnimationFrame(drawOrbs);
  }
  drawOrbs();
}

// ===========================================================================
// 14. PARTICLE BURST FX SYSTEM
// ===========================================================================

const particleSystem = {
  canvas: null,
  ctx: null,
  particles: [],

  init() {
    this.canvas = document.getElementById("particleBurstCanvas");
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext("2d");
    const resize = () => {
      this.canvas.width = window.innerWidth;
      this.canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);
    this.loop();
  },

  burst(x, y, color = "#2563eb", count = 18) {
    for (let i = 0; i < count; i++) {
      const angle = (Math.PI * 2 * i) / count + Math.random() * 0.5;
      const speed = 2 + Math.random() * 4;
      this.particles.push({
        x, y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 2,
        r: 3 + Math.random() * 4,
        alpha: 1,
        color,
        decay: 0.02 + Math.random() * 0.02,
        gravity: 0.12
      });
    }
  },

  loop() {
    if (!this.ctx) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.particles = this.particles.filter(p => p.alpha > 0.01);
    this.particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      p.vy += p.gravity;
      p.vx *= 0.97;
      p.alpha -= p.decay;
      this.ctx.save();
      this.ctx.globalAlpha = Math.max(0, p.alpha);
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      this.ctx.fillStyle = p.color;
      this.ctx.shadowColor = p.color;
      this.ctx.shadowBlur = 6;
      this.ctx.fill();
      this.ctx.restore();
    });
    requestAnimationFrame(() => this.loop());
  }
};

// ===========================================================================
// 15. CURSOR TRAIL EFFECT
// ===========================================================================

function initCursorTrail() {
  const trail = document.getElementById("cursorTrail");
  if (!trail || window.matchMedia("(pointer: coarse)").matches) return;

  let mouseX = -100, mouseY = -100;
  let trailX = -100, trailY = -100;
  const LERP = 0.14;

  document.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  function updateTrail() {
    trailX += (mouseX - trailX) * LERP;
    trailY += (mouseY - trailY) * LERP;
    trail.style.left = trailX + "px";
    trail.style.top  = trailY + "px";
    requestAnimationFrame(updateTrail);
  }
  updateTrail();

  // Scale up on interactive elements
  document.querySelectorAll("button, a, [data-jump], .nav-tab, .sample-chip").forEach(el => {
    el.addEventListener("mouseenter", () => {
      trail.style.width = "22px";
      trail.style.height = "22px";
      trail.style.opacity = "0.7";
    });
    el.addEventListener("mouseleave", () => {
      trail.style.width = "10px";
      trail.style.height = "10px";
      trail.style.opacity = "1";
    });
  });
}

// ===========================================================================
// 16. RIPPLE CLICK EFFECTS ON ALL BUTTONS
// ===========================================================================

function initRippleEffects() {
  const targets = document.querySelectorAll(
    ".btn-hero-primary, .btn-hero-secondary, .btn-hero-download, " +
    ".btn.btn-primary, .btn.btn-secondary, .nav-tab, .control-btn, " +
    ".header-download-btn, .floating-download-widget, .btn-modal-download-primary"
  );

  targets.forEach(btn => {
    btn.classList.add("ripple-btn");
    btn.addEventListener("click", (e) => {
      const rect = btn.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;

      const ripple = document.createElement("span");
      ripple.className = "ripple-effect";
      ripple.style.cssText = `width:${size}px;height:${size}px;left:${x}px;top:${y}px;`;
      btn.appendChild(ripple);

      // Particle burst from button center
      particleSystem.burst(
        e.clientX, e.clientY,
        btn.classList.contains("btn-hero-primary") ? "#2563eb" :
        btn.classList.contains("header-download-btn") ? "#10b981" : "#2563eb",
        12
      );

      setTimeout(() => ripple.remove(), 700);
    });
  });
}

// ===========================================================================
// 17. SCROLL-TRIGGERED GSAP ANIMATIONS FOR EVERY PAGE SECTION
// ===========================================================================

function initScrollAnimations() {
  if (typeof gsap === "undefined" || typeof ScrollTrigger === "undefined") return;
  gsap.registerPlugin(ScrollTrigger);

  // Hero stats section
  gsap.from(".hero-stat-card", {
    scrollTrigger: { trigger: ".hero-stats-grid", start: "top 90%" },
    y: 30, opacity: 0, stagger: 0.12, duration: 0.7, ease: "back.out(1.4)"
  });

  // Features section heading
  gsap.from(".section-heading", {
    scrollTrigger: { trigger: ".features-section", start: "top 85%" },
    y: 25, opacity: 0, duration: 0.7, ease: "power2.out"
  });

  // Feature cards with alternating directions
  gsap.from(".feature-card", {
    scrollTrigger: { trigger: ".features-grid", start: "top 80%" },
    y: 50, opacity: 0, stagger: { each: 0.1, from: "start" },
    duration: 0.65, ease: "power2.out"
  });

  // KPI ticker ribbon
  gsap.from(".kpi-ticker-item", {
    scrollTrigger: { trigger: ".kpi-ticker-ribbon", start: "top 95%" },
    x: -30, opacity: 0, stagger: 0.1, duration: 0.5, ease: "power2.out"
  });

  // Workspace header anchor
  gsap.from(".workspace-header-anchor", {
    scrollTrigger: { trigger: ".workspace-header-anchor", start: "top 90%" },
    y: -20, opacity: 0, duration: 0.5, ease: "power2.out"
  });
}

// ===========================================================================
// 18. PER-TAB ANIMATION TRIGGERS (fire on tab open)
// ===========================================================================

function runTabEntranceAnimation(tabId) {
  if (typeof gsap === "undefined") return;

  switch (tabId) {
    case "tab-radar":
      // Verdict hero pop in
      gsap.fromTo("#verdictHero",
        { scale: 0.93, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.5, ease: "back.out(1.6)", delay: 0.1 }
      );
      // Probability bars wipe in
      gsap.from(".prob-fill", { width: 0, duration: 0.85, stagger: 0.08, ease: "power2.out", delay: 0.2 });
      // Telemetry tiles cascade
      gsap.from(".telemetry-tile", {
        y: 20, opacity: 0, stagger: 0.06, duration: 0.4, ease: "back.out(1.4)", delay: 0.15
      });
      // Explainability card slide up
      gsap.from(".explainability-card", { y: 30, opacity: 0, duration: 0.55, ease: "power2.out", delay: 0.4 });
      // Tag pills stagger
      gsap.from(".tag-pill", { scale: 0.7, opacity: 0, stagger: 0.05, duration: 0.3, ease: "back.out(1.8)", delay: 0.55 });
      break;

    case "tab-consensus":
      // Consensus hero banner
      gsap.fromTo(".consensus-hero",
        { scaleX: 0.95, scaleY: 0.9, opacity: 0 },
        { scaleX: 1, scaleY: 1, opacity: 1, duration: 0.6, ease: "elastic.out(1, 0.6)", delay: 0.1 }
      );
      // Model cards cascade from bottom
      gsap.from(".model-card", {
        y: 40, opacity: 0, rotateX: 8, stagger: 0.04,
        duration: 0.5, ease: "power2.out", delay: 0.2
      });
      break;

    case "tab-match":
      // Timeline strip reveal
      gsap.from(".timeline-strip", {
        scaleX: 0, opacity: 0, duration: 0.7, ease: "power2.out",
        transformOrigin: "left center"
      });
      // Control buttons drop in
      gsap.from(".match-stream-controls .btn", {
        y: -15, opacity: 0, stagger: 0.08, duration: 0.4, ease: "back.out(1.4)", delay: 0.3
      });
      // Dashboard cards
      gsap.from("#tab-match .glass-card", {
        x: -30, opacity: 0, stagger: 0.15, duration: 0.5, ease: "power2.out", delay: 0.4
      });
      break;

    case "tab-sandbox":
      // Sandbox canvas pop
      gsap.fromTo("#sandboxCanvas",
        { scale: 0.92, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.6, ease: "back.out(1.4)", delay: 0.1 }
      );
      // Control buttons slide in from right
      gsap.from("#tab-sandbox .btn", {
        x: 20, opacity: 0, stagger: 0.07, duration: 0.4, ease: "power2.out", delay: 0.3
      });
      break;

    case "tab-leaderboard":
      // Leaderboard rows cascade
      gsap.from(".leaderboard-row", {
        x: -40, opacity: 0, stagger: 0.04, duration: 0.45, ease: "power2.out", delay: 0.1
      });
      break;

    case "tab-video":
      // Upload dropzone bounce
      gsap.fromTo(".upload-dropzone, .upload-area",
        { scale: 0.93, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.55, ease: "back.out(1.5)", delay: 0.1 }
      );
      break;

    case "tab-upload":
      // Upload form elements stagger
      gsap.from("#tab-upload .glass-card > *", {
        y: 20, opacity: 0, stagger: 0.08, duration: 0.5, ease: "power2.out", delay: 0.1
      });
      break;
  }
}

// ===========================================================================
// 19. ENHANCED VERDICT ANIMATION (particle burst on new result)
// ===========================================================================

function burstOnVerdict(colorHex) {
  // Find verdict hero position
  const hero = document.getElementById("verdictHero");
  if (!hero) return;
  const rect = hero.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  particleSystem.burst(cx, cy, colorHex || "#2563eb", 24);

  // Flash the confidence value
  const confEl = document.getElementById("confidenceValue");
  if (confEl && typeof gsap !== "undefined") {
    gsap.fromTo(confEl,
      { scale: 1.3, color: "#ffffff" },
      { scale: 1, color: colorHex || "#2563eb", duration: 0.55, ease: "power2.out" }
    );
  }
}

// ===========================================================================
// 20. ANIMATED PROGRESS BAR COUNTER (telemetry tiles with counting number)
// ===========================================================================

function animateTelemetryTile(id, toVal, suffix = "") {
  const el = document.getElementById(id);
  if (!el || typeof gsap === "undefined") return;
  const obj = { v: 0 };
  gsap.to(obj, {
    v: parseFloat(toVal) || 0,
    duration: 1.2,
    ease: "power2.out",
    onUpdate: () => {
      const raw = obj.v;
      el.innerText = (raw % 1 === 0 ? Math.round(raw) : raw.toFixed(1)) + suffix;
    }
  });
}

// ===========================================================================
// 21. MATCH WINDOW CHANGE FLASH ANIMATION
// ===========================================================================

function flashMatchWindow() {
  const panels = document.querySelectorAll("#tab-match .glass-card");
  panels.forEach(panel => {
    panel.classList.remove("match-window-flash");
    void panel.offsetWidth; // reflow
    panel.classList.add("match-window-flash");
  });
  if (typeof gsap !== "undefined") {
    gsap.fromTo("#matchWinSeq, #matchWinTime, #matchWinMajority",
      { scale: 1.12, opacity: 0.6 },
      { scale: 1, opacity: 1, duration: 0.3, ease: "power2.out" }
    );
  }
}

// ===========================================================================
// 22. HEADER SCROLL HIDE / REVEAL WITH GSAP
// ===========================================================================

function initHeaderScrollBehavior() {
  if (typeof gsap === "undefined") return;
  let lastY = 0;
  const header = document.querySelector(".app-header");
  if (!header) return;

  window.addEventListener("scroll", () => {
    const currentY = window.scrollY;
    if (currentY > 80 && currentY > lastY) {
      gsap.to(header, { y: -80, duration: 0.35, ease: "power2.in" });
    } else {
      gsap.to(header, { y: 0, duration: 0.35, ease: "power2.out" });
    }
    lastY = currentY;
  });
}

// ===========================================================================
// 23. BOOT ALL ANIMATION SYSTEMS
// ===========================================================================

document.addEventListener("DOMContentLoaded", () => {
  // These run AFTER the existing DOMContentLoaded handler
  setTimeout(() => {
    initAmbientOrbCanvas();
    particleSystem.init();
    initCursorTrail();
    initRippleEffects();
    initScrollAnimations();
    initHeaderScrollBehavior();

    // Patch tab navigation to also fire per-tab entrance animations
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
      tab.addEventListener("click", () => {
        setTimeout(() => runTabEntranceAnimation(tab.dataset.tab), 50);
      });
    });

    // Patch verdict update to fire particle burst
    const origUpdateVerdictUI = window.__origUpdateVerdictUI;
    if (!origUpdateVerdictUI) {
      // We hook into verdict UI updates by observing DOM changes on verdictLabel
      const verdictLabel = document.getElementById("verdictLabel");
      if (verdictLabel) {
        const mo = new MutationObserver(() => {
          const colorMap = {
            "Coordinated Attack": "#d97706",
            "Coordinated Defense": "#059669",
            "Delayed Support": "#db2777",
            "Spacing Breakdown": "#7c3aed"
          };
          const color = colorMap[verdictLabel.innerText] || "#2563eb";
          setTimeout(() => burstOnVerdict(color), 80);
        });
        mo.observe(verdictLabel, { childList: true, characterData: true, subtree: true });
      }
    }

    // Trigger initial radar tab animation
    runTabEntranceAnimation("tab-radar");
  }, 300);
});

