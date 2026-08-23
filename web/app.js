/**
 * app.js - Pusher-v5 Live Simulation & Control Center (UI-UX Pro Max Zero-Scroll Mentor Edition)
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const canvas = document.getElementById("simCanvas");
    const ctx = canvas.getContext("2d");
    const placeholder = document.getElementById("viewportPlaceholder");

    const connStatus = document.getElementById("connStatus");
    const connDot = connStatus.querySelector(".status-indicator");
    const connLabel = connStatus.querySelector(".status-label");

    const fpsDisplay = document.getElementById("fpsDisplay");
    const epDisplay = document.getElementById("epDisplay");
    const stepDisplay = document.getElementById("stepDisplay");
    const policyBadge = document.getElementById("currentPolicyBadge");

    // Mode Switcher
    const btnModeSim = document.getElementById("btnModeSim");
    const btnModeTrain = document.getElementById("btnModeTrain");

    // KPI Readouts
    const telReturn = document.getElementById("telReturn");
    const telStepReward = document.getElementById("telStepReward");
    const telDistGoal = document.getElementById("telDistGoal");
    const telDistArm = document.getElementById("telDistArm");

    // 3D Coordinates Vector Elements
    const coordTipX = document.getElementById("coordTipX");
    const coordTipY = document.getElementById("coordTipY");
    const coordTipZ = document.getElementById("coordTipZ");

    const coordObjX = document.getElementById("coordObjX");
    const coordObjY = document.getElementById("coordObjY");
    const coordObjZ = document.getElementById("coordObjZ");

    const coordGoalX = document.getElementById("coordGoalX");
    const coordGoalY = document.getElementById("coordGoalY");
    const coordGoalZ = document.getElementById("coordGoalZ");

    const distGoalProgress = document.getElementById("distGoalProgress");
    const distGoalPercent = document.getElementById("distGoalPercent");
    const torqueGrid = document.getElementById("torqueGrid");

    // Simulation Action Buttons
    const btnStartSim = document.getElementById("btnStartSim");
    const btnPauseSim = document.getElementById("btnPauseSim");
    const btnStepSim = document.getElementById("btnStepSim");
    const btnResetSim = document.getElementById("btnResetSim");
    const btnTogglePolicy = document.getElementById("btnTogglePolicy");
    const btnToggleHud = document.getElementById("btnToggleHud");
    const speedChips = document.querySelectorAll(".chip");

    // Training Controls & Preset Selector
    const trainPresetSelect = document.getElementById("trainPresetSelect");
    const customInputRow = document.getElementById("customInputRow");
    const trainTimestepsInput = document.getElementById("trainTimesteps");
    const trainEvalFreqInput = document.getElementById("trainEvalFreq");
    const btnStartTrain = document.getElementById("btnStartTrain");
    const btnQuickTrain = document.getElementById("btnQuickTrain");
    const trainStatusMsg = document.getElementById("trainStatusMsg");
    const trainProgressFill = document.getElementById("trainProgressFill");

    // Tabs & Terminal Logs
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const terminalLogBody = document.getElementById("terminalLogBody");
    const btnRefreshGallery = document.getElementById("btnRefreshGallery");

    // State Variables
    let ws = null;
    let isSimRunning = false;
    let isSimPaused = false;
    let currentPolicy = "trained"; // 'trained' or 'random'
    let frameCount = 0;
    let telemetryChart = null;
    let checkpointsData = [];
    let lastLogCount = 0;
    let recentRewardsBuffer = [];

    const jointNames = [
        "J1 (Shoulder Pan)",
        "J2 (Shoulder Lift)",
        "J3 (Arm Roll)",
        "J4 (Elbow Flex)",
        "J5 (Forearm Roll)",
        "J6 (Wrist Flex)",
        "J7 (Wrist Roll)"
    ];

    // 1. Initialize 7-DOF Bipolar Action Torque Grid
    function initTorqueGrid() {
        torqueGrid.innerHTML = "";
        for (let i = 1; i <= 7; i++) {
            const card = document.createElement("div");
            card.className = "torque-card";
            card.setAttribute("data-tooltip", `${jointNames[i-1]} torque control (Nm)`);
            card.innerHTML = `
                <div class="torque-header">
                    <span>J${i}</span>
                    <span class="torque-reading text-cyan" id="torqueVal_${i}">+0.0</span>
                </div>
                <div class="bipolar-rail">
                    <div class="bipolar-half-left">
                        <div class="bipolar-bar-left" id="torqueLeft_${i}"></div>
                    </div>
                    <div class="bipolar-half-right">
                        <div class="bipolar-bar-right" id="torqueRight_${i}"></div>
                    </div>
                </div>
            `;
            torqueGrid.appendChild(card);
        }
    }

    // 2. Initialize Real-Time Dynamic Chart with Moving Average (Chart.js)
    function initTelemetryChart() {
        const chartCtx = document.getElementById("liveTelemetryChart").getContext("2d");
        chartInstance = new Chart(chartCtx, {
            type: "line",
            data: {
                labels: [],
                datasets: [
                    {
                        label: "Step Reward (Raw)",
                        borderColor: "rgba(56, 189, 248, 0.45)",
                        backgroundColor: "transparent",
                        data: [],
                        borderWidth: 1.5,
                        borderDash: [3, 3],
                        pointRadius: 0,
                        tension: 0.1,
                        yAxisID: "yReward",
                    },
                    {
                        label: "20-Ep Moving Avg",
                        borderColor: "#38bdf8",
                        backgroundColor: "rgba(56, 189, 248, 0.12)",
                        data: [],
                        borderWidth: 2.5,
                        pointRadius: 0,
                        tension: 0.25,
                        yAxisID: "yReward",
                    },
                    {
                        label: "Dist to Goal (m)",
                        borderColor: "#f59e0b",
                        backgroundColor: "rgba(245, 158, 11, 0.08)",
                        data: [],
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2,
                        yAxisID: "yDist",
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        grid: { color: "rgba(51, 65, 85, 0.25)" },
                        ticks: { color: "#64748b", font: { family: "Fira Code", size: 9 } }
                    },
                    yReward: {
                        position: "left",
                        grid: { color: "rgba(51, 65, 85, 0.25)" },
                        ticks: { color: "#38bdf8", font: { family: "Fira Code", size: 9 } },
                        title: { display: true, text: "Return (Reward)", color: "#38bdf8", font: { size: 9.5, weight: 'bold' } }
                    },
                    yDist: {
                        position: "right",
                        grid: { drawOnChartArea: false },
                        ticks: { color: "#f59e0b", font: { family: "Fira Code", size: 9 } },
                        title: { display: true, text: "Dist to Goal (m)", color: "#f59e0b", font: { size: 9.5, weight: 'bold' } }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#cbd5e1", font: { family: "Plus Jakarta Sans", size: 10 }, boxWidth: 10 }
                    },
                    tooltip: {
                        backgroundColor: "rgba(15, 23, 42, 0.95)",
                        borderColor: "rgba(56, 189, 248, 0.4)",
                        borderWidth: 1,
                        titleFont: { family: "Fira Code", size: 9.5 },
                        bodyFont: { family: "Fira Code", size: 9.5 },
                        padding: 6,
                    }
                }
            }
        });
    }

    // 3. Tab Switching
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            tabBtns.forEach((b) => b.classList.remove("active"));
            tabPanes.forEach((p) => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
            if (targetId === "tabChart" && chartInstance) {
                setTimeout(() => chartInstance.resize(), 50);
            }
        });
    });

    // Mode Switcher Handlers
    btnModeSim.addEventListener("click", () => {
        btnModeSim.classList.add("active");
        btnModeTrain.classList.remove("active");
        // Focus chart tab
        tabBtns[0].click();
    });

    btnModeTrain.addEventListener("click", () => {
        btnModeTrain.classList.add("active");
        btnModeSim.classList.remove("active");
        // Focus logs tab
        tabBtns[2].click();
    });

    // 4. Training Preset Selector
    trainPresetSelect.addEventListener("change", () => {
        const val = trainPresetSelect.value;
        if (val === "custom") {
            customInputRow.style.display = "grid";
        } else {
            customInputRow.style.display = "none";
            const [steps, evalFreq] = val.split("_");
            trainTimestepsInput.value = steps;
            trainEvalFreqInput.value = evalFreq;
        }
    });

    // 5. WebSocket Connection
    function connectWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/simulation`;

        connLabel.textContent = "CONNECTING";
        connDot.className = "status-indicator disconnected";

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            connLabel.textContent = "LIVE (CONNECTED)";
            connDot.className = "status-indicator connected";
            console.log("[WebSocket] Connected successfully.");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "sim_frame") {
                renderSimFrame(data);
            }
        };

        ws.onclose = () => {
            connLabel.textContent = "DISCONNECTED";
            connDot.className = "status-indicator disconnected";
            console.log("[WebSocket] Disconnected. Reconnecting in 2s...");
            setTimeout(connectWebSocket, 2000);
        };

        ws.onerror = (err) => {
            console.error("[WebSocket Error]", err);
            ws.close();
        };
    }

    // 6. Render Simulation Frame & Update Telemetry
    const simImg = new Image();
    function renderSimFrame(data) {
        if (placeholder && !placeholder.classList.contains("hidden")) {
            placeholder.classList.add("hidden");
        }

        // Draw image frame
        simImg.onload = () => {
            ctx.drawImage(simImg, 0, 0, canvas.width, canvas.height);
        };
        simImg.src = `data:image/jpeg;base64,${data.frame}`;

        // Header Meta
        epDisplay.textContent = `Ep #${data.episode}`;
        stepDisplay.textContent = `Step ${data.step}`;

        // KPI Numbers
        telReturn.textContent = (data.ep_reward >= 0 ? "+" : "") + data.ep_reward.toFixed(2);
        telStepReward.textContent = (data.step_reward >= 0 ? "+" : "") + data.step_reward.toFixed(3);
        telDistGoal.textContent = `${data.dist_goal.toFixed(3)} m`;
        telDistArm.textContent = `${data.dist_arm.toFixed(3)} m`;

        // 3D Spatial Vector Coordinates
        if (data.tip_pos && data.tip_pos.length >= 3) {
            coordTipX.textContent = (data.tip_pos[0] >= 0 ? "+" : "") + data.tip_pos[0].toFixed(2);
            coordTipY.textContent = (data.tip_pos[1] >= 0 ? "+" : "") + data.tip_pos[1].toFixed(2);
            coordTipZ.textContent = (data.tip_pos[2] >= 0 ? "+" : "") + data.tip_pos[2].toFixed(2);
        }
        if (data.obj_pos && data.obj_pos.length >= 3) {
            coordObjX.textContent = (data.obj_pos[0] >= 0 ? "+" : "") + data.obj_pos[0].toFixed(2);
            coordObjY.textContent = (data.obj_pos[1] >= 0 ? "+" : "") + data.obj_pos[1].toFixed(2);
            coordObjZ.textContent = (data.obj_pos[2] >= 0 ? "+" : "") + data.obj_pos[2].toFixed(2);
        }
        if (data.goal_pos && data.goal_pos.length >= 3) {
            coordGoalX.textContent = (data.goal_pos[0] >= 0 ? "+" : "") + data.goal_pos[0].toFixed(2);
            coordGoalY.textContent = (data.goal_pos[1] >= 0 ? "+" : "") + data.goal_pos[1].toFixed(2);
            coordGoalZ.textContent = (data.goal_pos[2] >= 0 ? "+" : "") + data.goal_pos[2].toFixed(2);
        }

        // Target Alignment Gauge (0.05m = success threshold)
        const dGoal = data.dist_goal;
        const progressPct = Math.max(5, Math.min(100, (1.0 - (dGoal / 0.5)) * 100));
        distGoalProgress.style.width = `${progressPct}%`;
        if (dGoal <= 0.08) {
            distGoalPercent.textContent = `${dGoal.toFixed(3)} m (Target Reached)`;
            distGoalPercent.className = "dist-num text-emerald";
        } else {
            distGoalPercent.textContent = `${dGoal.toFixed(3)} m (Pushing)`;
            distGoalPercent.className = "dist-num text-amber";
        }

        // 7-DOF Bipolar Action Torque Bars ([-2.0, +2.0] Nm)
        if (data.actions && data.actions.length >= 7) {
            for (let i = 0; i < 7; i++) {
                const torque = data.actions[i];
                const valElem = document.getElementById(`torqueVal_${i+1}`);
                const leftBar = document.getElementById(`torqueLeft_${i+1}`);
                const rightBar = document.getElementById(`torqueRight_${i+1}`);
                if (valElem && leftBar && rightBar) {
                    valElem.textContent = (torque >= 0 ? "+" : "") + torque.toFixed(1);
                    if (torque >= 0) {
                        valElem.className = "torque-reading text-cyan";
                        leftBar.style.width = "0%";
                        const pct = Math.min(100, (torque / 2.0) * 100);
                        rightBar.style.width = `${pct}%`;
                    } else {
                        valElem.className = "torque-reading text-rose";
                        rightBar.style.width = "0%";
                        const pct = Math.min(100, (Math.abs(torque) / 2.0) * 100);
                        leftBar.style.width = `${pct}%`;
                    }
                }
            }
        }

        // Dynamic Chart Update with Moving Average
        if (chartInstance && data.step % 2 === 0) {
            const labels = chartInstance.data.labels;
            const rawRewData = chartInstance.data.datasets[0].data;
            const maRewData = chartInstance.data.datasets[1].data;
            const distData = chartInstance.data.datasets[2].data;

            recentRewardsBuffer.push(data.step_reward);
            if (recentRewardsBuffer.length > 20) recentRewardsBuffer.shift();
            const maVal = recentRewardsBuffer.reduce((a, b) => a + b, 0) / recentRewardsBuffer.length;

            labels.push(`S${data.step}`);
            rawRewData.push(data.step_reward);
            maRewData.push(maVal);
            distData.push(data.dist_goal);

            if (labels.length > 35) {
                labels.shift();
                rawRewData.shift();
                maRewData.shift();
                distData.shift();
            }
            chartInstance.update();
        }

        // FPS Calculation
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
            fpsDisplay.textContent = `${fps} FPS`;
            frameCount = 0;
            lastFpsTime = now;
        }
    }

    // 7. Action Functions
    function toggleSimulation() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!isSimRunning) {
            isSimRunning = true;
            isSimPaused = false;
            ws.send(JSON.stringify({ command: "start" }));
            btnStartSim.classList.remove("btn-primary");
            btnStartSim.classList.add("btn-emerald");
            btnStartSim.querySelector("span").textContent = "Running...";
        } else {
            isSimPaused = !isSimPaused;
            ws.send(JSON.stringify({ command: "pause" }));
            btnPauseSim.querySelector("span").textContent = isSimPaused ? "Resume" : "Pause";
        }
    }

    function stepOnce() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ command: "step" }));
    }

    function resetEnv() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ command: "reset" }));
        if (chartInstance) {
            chartInstance.data.labels = [];
            chartInstance.data.datasets[0].data = [];
            chartInstance.data.datasets[1].data = [];
            chartInstance.data.datasets[2].data = [];
            chartInstance.update();
        }
    }

    function toggleHUD() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({ command: "toggle_hud" }));
    }

    // Button Click Listeners
    btnStartSim.addEventListener("click", toggleSimulation);
    btnPauseSim.addEventListener("click", () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        isSimPaused = !isSimPaused;
        ws.send(JSON.stringify({ command: "pause" }));
        btnPauseSim.querySelector("span").textContent = isSimPaused ? "Resume" : "Pause";
    });
    btnStepSim.addEventListener("click", stepOnce);
    btnResetSim.addEventListener("click", resetEnv);
    btnToggleHud.addEventListener("click", toggleHUD);

    btnTogglePolicy.addEventListener("click", () => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (currentPolicy === "trained") {
            currentPolicy = "random";
            policyBadge.textContent = "Random Exploration";
            policyBadge.className = "pill-bright text-amber";
            btnTogglePolicy.querySelector("span").textContent = "Set Trained PPO";
        } else {
            currentPolicy = "trained";
            policyBadge.textContent = "Trained PPO";
            policyBadge.className = "pill-bright text-cyan";
            btnTogglePolicy.querySelector("span").textContent = "Set Random Mode";
        }
        ws.send(JSON.stringify({ command: "set_policy", policy: currentPolicy }));
    });

    // Speed Chips
    speedChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            speedChips.forEach((c) => c.classList.remove("active"));
            chip.classList.add("active");
            const speed = parseFloat(chip.getAttribute("data-speed"));
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: "set_speed", speed: speed }));
            }
        });
    });

    // 8. Keyboard Shortcuts (Space, R, S, H)
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

        if (e.code === "Space") {
            e.preventDefault();
            toggleSimulation();
        } else if (e.key === "r" || e.key === "R") {
            resetEnv();
        } else if (e.key === "s" || e.key === "S") {
            stepOnce();
        } else if (e.key === "h" || e.key === "H") {
            toggleHUD();
        }
    });

    // 9. Training Controller & Live Log Streaming
    async function triggerTraining(timesteps, evalFreq) {
        btnStartTrain.disabled = true;
        btnQuickTrain.disabled = true;
        trainStatusMsg.textContent = "PPO Training in progress...";
        trainProgressFill.style.width = "10%";

        // Auto-switch to Live PPO Logs tab
        tabBtns[2].click();
        appendTerminalLog(`[LAUNCH] PPO Training initiated: ${timesteps.toLocaleString()} Timesteps, Eval every ${evalFreq.toLocaleString()} steps.`);

        try {
            const resp = await fetch(`/api/train/start?timesteps=${timesteps}&eval_freq=${evalFreq}`, {
                method: "POST"
            });
            const res = await resp.json();
            trainStatusMsg.textContent = res.message;

            const interval = setInterval(async () => {
                const sResp = await fetch("/api/train/status");
                const sData = await sResp.json();
                trainStatusMsg.textContent = `Status: ${sData.status} (${sData.progress}%)`;
                trainProgressFill.style.width = `${Math.max(5, sData.progress)}%`;

                // Stream backend logs
                if (sData.logs && sData.logs.length > lastLogCount) {
                    for (let i = lastLogCount; i < sData.logs.length; i++) {
                        appendTerminalLog(sData.logs[i]);
                    }
                    lastLogCount = sData.logs.length;
                }

                if (!sData.is_training) {
                    clearInterval(interval);
                    btnStartTrain.disabled = false;
                    btnQuickTrain.disabled = false;
                    trainProgressFill.style.width = "100%";
                    loadMilestones();
                }
            }, 1000);
        } catch (e) {
            trainStatusMsg.textContent = `Error: ${e.message}`;
            appendTerminalLog(`[ERROR] ${e.message}`);
            btnStartTrain.disabled = false;
            btnQuickTrain.disabled = false;
        }
    }

    function appendTerminalLog(msg) {
        const line = document.createElement("div");
        line.className = "log-line";
        if (msg.includes("Error")) line.classList.add("rose");
        else if (msg.includes("Checkpoint") || msg.includes("Complete")) line.classList.add("emerald");
        else if (msg.includes("Step")) line.classList.add("cyan");
        line.textContent = msg;
        terminalLogBody.appendChild(line);
        terminalLogBody.scrollTop = terminalLogBody.scrollHeight;
    }

    btnStartTrain.addEventListener("click", () => {
        const timesteps = parseInt(trainTimestepsInput.value) || 500000;
        const evalFreq = parseInt(trainEvalFreqInput.value) || 50000;
        triggerTraining(timesteps, evalFreq);
    });

    btnQuickTrain.addEventListener("click", () => {
        triggerTraining(10000, 5000);
    });

    // 10. Milestone Replay Deck Loader (Widescreen Multi-Video Card Gallery)
    const milestoneGalleryContainer = document.getElementById("milestoneGalleryContainer");

    async function loadMilestones() {
        if (!milestoneGalleryContainer) return;
        milestoneGalleryContainer.innerHTML = `<div class="gallery-loading-card">Loading milestone checkpoint videos...</div>`;
        try {
            const resp = await fetch("/api/checkpoints");
            const data = await resp.json();
            if (data.checkpoints && data.checkpoints.length > 0) {
                checkpointsData = data.checkpoints;
                milestoneGalleryContainer.innerHTML = "";
                checkpointsData.forEach((cp, idx) => {
                    const isBase = cp.step === 0;
                    const isFinal = idx === checkpointsData.length - 1 && cp.step > 0;
                    let tagText = isBase ? "Step 0 (Baseline)" : (isFinal ? `Step ${(cp.step/1000).toFixed(0)}k (Final)` : `Step ${(cp.step/1000).toFixed(0)}k`);
                    
                    let statusLabel = "Mid-Training";
                    if (isBase) statusLabel = "Random Explore";
                    else if (cp.reward > -50) statusLabel = "Target Pushed";

                    const card = document.createElement("div");
                    card.className = "checkpoint-card";
                    card.innerHTML = `
                        <div class="cp-card-head">
                            <span class="cp-card-title text-cyan">${tagText}</span>
                            <span class="cp-card-badge ${cp.reward > -50 ? 'text-emerald' : 'text-amber'}">${cp.reward !== null ? (cp.reward >= 0 ? '+' : '') + cp.reward.toFixed(1) + ' pts' : 'N/A'}</span>
                        </div>
                        <div class="cp-video-box">
                            <video autoplay loop muted playsinline poster="${cp.gif || ''}">
                                <source src="${cp.mp4}" type="video/mp4">
                            </video>
                        </div>
                        <div class="cp-meta-body">
                            <div class="cp-meta-row">
                                <span class="cp-meta-label">Goal Distance</span>
                                <span class="cp-meta-val text-cyan">${cp.dist_goal !== null ? cp.dist_goal.toFixed(3) + ' m' : 'N/A'}</span>
                            </div>
                            <div class="cp-meta-row">
                                <span class="cp-meta-label">Policy Status</span>
                                <span class="cp-meta-val ${cp.reward > -50 ? 'text-emerald' : 'text-purple'}">${statusLabel}</span>
                            </div>
                        </div>
                        <div class="cp-action-row">
                            <a href="${cp.mp4}" download class="btn btn-outline-cyan cp-btn" data-tooltip="Download MP4 video">
                                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                                <span>MP4</span>
                            </a>
                            <a href="${cp.gif || cp.mp4}" download class="btn btn-secondary cp-btn" data-tooltip="Download GIF animation">
                                <span>GIF</span>
                            </a>
                        </div>
                    `;
                    milestoneGalleryContainer.appendChild(card);
                });
            } else {
                milestoneGalleryContainer.innerHTML = `<div class="gallery-loading-card">No checkpoints recorded yet. Run PPO training to generate videos.</div>`;
            }
        } catch (e) {
            milestoneGalleryContainer.innerHTML = `<div class="gallery-loading-card">Error loading milestones: ${e.message}</div>`;
        }
    }

    btnRefreshGallery.addEventListener("click", loadMilestones);

    // Enable smooth mousewheel horizontal scrolling on milestone gallery
    if (milestoneGalleryContainer) {
        milestoneGalleryContainer.addEventListener("wheel", (evt) => {
            if (evt.deltaY !== 0) {
                evt.preventDefault();
                milestoneGalleryContainer.scrollLeft += evt.deltaY * 1.5;
            }
        }, { passive: false });
    }

    // Initialize App
    initTorqueGrid();
    initTelemetryChart();
    connectWebSocket();
    loadMilestones();
});
