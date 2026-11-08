/**
 * ============================================================================
 * HANDWRITTEN DIGIT RECOGNITION - FRONTEND CONTROLLER
 * ============================================================================
 * Handles canvas interaction, smoothing, brush sizing, preset drawing,
 * API communication with the PyTorch Flask backend, probability visualization,
 * debug inspector, and prediction history in localStorage.
 */

(() => {
    "use strict";

    // ------------------------------------------------------------------------
    // DOM ELEMENTS
    // ------------------------------------------------------------------------
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const canvasContainer = document.querySelector(".canvas-container");
    const canvasOverlay = document.getElementById("canvasOverlay");

    const clearBtn = document.getElementById("clearBtn");
    const predictBtn = document.getElementById("predictBtn");
    const predictBtnText = document.getElementById("predictBtnText");
    const alertBox = document.getElementById("alertBox");

    const predictionDisplay = document.getElementById("predictionDisplay");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBarFill = document.getElementById("confidenceBarFill");
    const inferenceBadge = document.getElementById("inferenceBadge");
    const probabilityList = document.getElementById("probabilityList");

    const debugImage = document.getElementById("debugImage");
    const debugPlaceholder = document.getElementById("debugPlaceholder");

    const historyList = document.getElementById("historyList");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");

    const modelStatus = document.getElementById("modelStatus");
    const statusText = document.getElementById("statusText");
    const modelStatusDot = document.querySelector(".pulse-dot");

    const brushButtons = document.querySelectorAll(".brush-btn");
    const presetButtons = document.querySelectorAll(".preset-btn");

    // ------------------------------------------------------------------------
    // API & BACKEND CONFIGURATION
    // ------------------------------------------------------------------------
    // Automatically routes requests:
    // 1. If page is served directly through Flask on port 5000 -> uses relative paths ("").
    // 2. If page is served via VS Code Live Server (port 5500) or any other host -> targets http://127.0.0.1:5000.
    const backendHost = window.location.hostname === "localhost" ? "localhost" : "127.0.0.1";
    const API_BASE_URL = (window.location.port === "5000")
        ? ""
        : `http://${backendHost}:5000`;

    // ------------------------------------------------------------------------
    // APPLICATION STATE
    // ------------------------------------------------------------------------
    let isDrawing = false;
    let hasStrokes = false;
    let currentBrushSize = 20; // Default brush stroke width
    let lastPoint = null;
    let midPoint = null;
    let strokeCount = 0;

    const STORAGE_KEY = "mnist_digit_history_v1";

    // ------------------------------------------------------------------------
    // INITIALIZATION
    // ------------------------------------------------------------------------
    function init() {
        initCanvas();
        initProbabilityBars();
        setupBrushControls();
        setupPresetControls();
        setupCanvasListeners();
        setupActionListeners();
        loadHistory();
        setBackendStatus("checking", "Connecting to Backend...");
        checkBackendHealth();
        // Periodic health check every 5 seconds
        setInterval(checkBackendHealth, 5000);
    }

    /**
     * Initializes the canvas buffer and sets up drawing context.
     */
    function initCanvas() {
        canvas.width = 280;
        canvas.height = 280;
        resetCanvasContext();
        clearCanvas();
    }

    function resetCanvasContext() {
        ctx.fillStyle = "#000000";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = currentBrushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
    }

    /**
     * Fills the canvas with black background.
     */
    function clearCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasStrokes = false;
        strokeCount = 0;
        lastPoint = null;
        midPoint = null;
        if (canvasOverlay) {
            canvasOverlay.classList.remove("hidden");
        }
        hideAlert();
    }

    /**
     * Clears all predictions and resets the UI display.
     */
    function resetResults() {
        predictionDisplay.textContent = "?";
        confidenceValue.textContent = "--";
        confidenceBarFill.style.width = "0%";
        inferenceBadge.classList.add("hidden");

        // Reset all probability bars to 0%
        for (let i = 0; i <= 9; i++) {
            const fill = document.getElementById(`prob-fill-${i}`);
            const pct = document.getElementById(`prob-pct-${i}`);
            const row = document.getElementById(`prob-row-${i}`);
            if (fill) fill.style.width = "0%";
            if (pct) pct.textContent = "0.00%";
            if (row) row.classList.remove("is-winner");
        }

        // Reset debug image
        debugImage.src = "";
        debugImage.classList.add("hidden");
        debugPlaceholder.classList.remove("hidden");
    }

    /**
     * Generates horizontal progress bars for classes 0 to 9.
     */
    function initProbabilityBars() {
        probabilityList.innerHTML = "";
        for (let i = 0; i <= 9; i++) {
            const row = document.createElement("div");
            row.className = "prob-row";
            row.id = `prob-row-${i}`;

            row.innerHTML = `
                <span class="prob-digit">${i}</span>
                <div class="prob-track">
                    <div class="prob-fill" id="prob-fill-${i}"></div>
                </div>
                <span class="prob-pct" id="prob-pct-${i}">0.00%</span>
            `;
            probabilityList.appendChild(row);
        }
    }

    // ------------------------------------------------------------------------
    // BRUSH SIZE CONTROLS
    // ------------------------------------------------------------------------
    function setupBrushControls() {
        brushButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                brushButtons.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                currentBrushSize = parseInt(btn.dataset.size, 10) || 20;
                ctx.lineWidth = currentBrushSize;
            });
        });
    }

    // ------------------------------------------------------------------------
    // SMOOTH DRAWING ENGINE (Pointer Events with Quadratic Curves)
    // ------------------------------------------------------------------------
    function getCanvasCoordinates(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    function onPointerDown(event) {
        event.preventDefault();
        canvas.setPointerCapture(event.pointerId);
        isDrawing = true;
        hasStrokes = true;
        strokeCount++;
        hideAlert();

        if (canvasOverlay) {
            canvasOverlay.classList.add("hidden");
        }
        canvasContainer.classList.add("drawing-active");

        const pt = getCanvasCoordinates(event);
        lastPoint = pt;
        midPoint = pt;

        // Draw an initial round dot for click/tap
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, currentBrushSize / 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#000000";
    }

    function onPointerMove(event) {
        if (!isDrawing) return;
        event.preventDefault();

        const currentPoint = getCanvasCoordinates(event);
        // Calculate midpoint for quadratic bezier curve smoothing
        const newMidPoint = {
            x: (lastPoint.x + currentPoint.x) / 2,
            y: (lastPoint.y + currentPoint.y) / 2
        };

        ctx.beginPath();
        ctx.moveTo(midPoint.x, midPoint.y);
        ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, newMidPoint.x, newMidPoint.y);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = currentBrushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();

        lastPoint = currentPoint;
        midPoint = newMidPoint;
    }

    function onPointerUp(event) {
        if (!isDrawing) return;
        isDrawing = false;
        canvasContainer.classList.remove("drawing-active");
        try {
            canvas.releasePointerCapture(event.pointerId);
        } catch (_) {}
    }

    function setupCanvasListeners() {
        canvas.addEventListener("pointerdown", onPointerDown);
        canvas.addEventListener("pointermove", onPointerMove);
        canvas.addEventListener("pointerup", onPointerUp);
        canvas.addEventListener("pointercancel", onPointerUp);
        canvas.addEventListener("lostpointercapture", onPointerUp);
    }

    // ------------------------------------------------------------------------
    // PRESET CANONICAL DIGIT GENERATOR
    // ------------------------------------------------------------------------
    /**
     * Stroke blueprints for digits 0-9 to allow instant 1-click verification.
     */
    const DIGIT_PRESETS = {
        0: (c) => {
            c.beginPath();
            c.ellipse(140, 140, 58, 85, 0, 0, Math.PI * 2);
            c.stroke();
        },
        1: (c) => {
            c.beginPath();
            c.moveTo(140, 50);
            c.lineTo(140, 230);
            c.stroke();
        },
        2: (c) => {
            c.beginPath();
            c.moveTo(85, 95);
            c.quadraticCurveTo(140, 50, 195, 95);
            c.quadraticCurveTo(195, 145, 85, 230);
            c.lineTo(195, 230);
            c.stroke();
        },
        3: (c) => {
            c.beginPath();
            c.moveTo(90, 65);
            c.lineTo(190, 65);
            c.lineTo(140, 135);
            c.stroke();
            c.beginPath();
            c.arc(140, 175, 48, -Math.PI / 3, Math.PI / 1.5, false);
            c.stroke();
        },
        4: (c) => {
            c.beginPath();
            c.moveTo(175, 50);
            c.lineTo(175, 230);
            c.stroke();
            c.beginPath();
            c.moveTo(175, 50);
            c.lineTo(75, 170);
            c.lineTo(205, 170);
            c.stroke();
        },
        5: (c) => {
            c.beginPath();
            c.moveTo(190, 65);
            c.lineTo(95, 65);
            c.lineTo(90, 135);
            c.stroke();
            c.beginPath();
            c.arc(140, 175, 48, -Math.PI / 3, Math.PI / 1.5, false);
            c.stroke();
        },
        6: (c) => {
            c.beginPath();
            c.moveTo(175, 65);
            c.quadraticCurveTo(90, 80, 95, 170);
            c.stroke();
            c.beginPath();
            c.ellipse(140, 175, 48, 50, 0, 0, Math.PI * 2);
            c.stroke();
        },
        7: (c) => {
            c.beginPath();
            c.moveTo(85, 65);
            c.lineTo(195, 65);
            c.lineTo(115, 225);
            c.stroke();
        },
        8: (c) => {
            c.beginPath();
            c.ellipse(140, 95, 38, 38, 0, 0, Math.PI * 2);
            c.stroke();
            c.beginPath();
            c.ellipse(140, 175, 48, 48, 0, 0, Math.PI * 2);
            c.stroke();
        },
        9: (c) => {
            c.beginPath();
            c.ellipse(140, 105, 48, 48, 0, 0, Math.PI * 2);
            c.stroke();
            c.beginPath();
            c.moveTo(188, 105);
            c.quadraticCurveTo(188, 205, 120, 225);
            c.stroke();
        }
    };

    function setupPresetControls() {
        presetButtons.forEach((btn) => {
            btn.addEventListener("click", () => {
                const digit = parseInt(btn.dataset.digit, 10);
                drawPresetDigit(digit);
            });
        });
    }

    function drawPresetDigit(digit) {
        clearCanvas();
        resetResults();

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 22;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        const drawFn = DIGIT_PRESETS[digit];
        if (drawFn) {
            drawFn(ctx);
            hasStrokes = true;
            strokeCount = 5;
            if (canvasOverlay) {
                canvasOverlay.classList.add("hidden");
            }
            // Trigger prediction automatically for smooth testing
            predictDigit();
        }
    }

    // ------------------------------------------------------------------------
    // ACTION LISTENERS (Clear & Predict)
    // ------------------------------------------------------------------------
    function setupActionListeners() {
        clearBtn.addEventListener("click", () => {
            clearCanvas();
            resetResults();
        });

        predictBtn.addEventListener("click", () => {
            predictDigit();
        });

        clearHistoryBtn.addEventListener("click", () => {
            clearHistory();
        });
    }

    // ------------------------------------------------------------------------
    // PREDICTION CONTROLLER
    // ------------------------------------------------------------------------
    async function predictDigit() {
        // Prevent prediction if canvas is blank
        if (!hasStrokes || strokeCount === 0) {
            showAlert("Canvas is empty. Please draw a digit before predicting.");
            return;
        }

        // Quick client-side pixel check to ensure actual strokes exist
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        let whitePixels = 0;
        for (let i = 0; i < data.length; i += 4) {
            if (data[i] > 30) whitePixels++;
        }

        if (whitePixels < 25) {
            showAlert("Drawing is too faint. Please draw a clearer digit.");
            return;
        }

        hideAlert();
        setLoadingState(true);

        const startTime = performance.now();
        const base64Data = canvas.toDataURL("image/png");

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ image: base64Data })
            });

            const elapsed = Math.round(performance.now() - startTime);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server returned HTTP ${response.status}`);
            }

            const result = await response.json();
            displayPredictionResults(result, elapsed, base64Data);

        } catch (error) {
            console.error("Prediction request failed:", error);
            showAlert(error.message || "Prediction failed. Check backend connection.");
            resetResults();
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(loading) {
        predictBtn.disabled = loading;
        if (loading) {
            predictBtnText.textContent = "Processing...";
            predictBtn.style.opacity = "0.75";
        } else {
            predictBtnText.textContent = "Predict Digit";
            predictBtn.style.opacity = "1";
        }
    }

    function displayPredictionResults(result, elapsedMs, originalThumb) {
        const { prediction, confidence, probabilities, debug_image } = result;

        // Display hero prediction
        predictionDisplay.textContent = prediction;
        confidenceValue.textContent = `${confidence.toFixed(2)}%`;
        confidenceBarFill.style.width = `${Math.min(100, Math.max(0, confidence))}%`;

        // Latency
        inferenceBadge.textContent = `Inference: ${elapsedMs} ms`;
        inferenceBadge.classList.remove("hidden");

        // Display probabilities for all 10 classes
        for (let i = 0; i <= 9; i++) {
            const prob = probabilities && probabilities[i.toString()] !== undefined
                ? probabilities[i.toString()]
                : 0.0;

            const fill = document.getElementById(`prob-fill-${i}`);
            const pct = document.getElementById(`prob-pct-${i}`);
            const row = document.getElementById(`prob-row-${i}`);

            if (fill) fill.style.width = `${Math.min(100, Math.max(0, prob))}%`;
            if (pct) pct.textContent = `${prob.toFixed(2)}%`;

            if (row) {
                if (i === prediction) {
                    row.classList.add("is-winner");
                } else {
                    row.classList.remove("is-winner");
                }
            }
        }

        // Display preprocessed 28x28 debug image
        if (debug_image) {
            debugImage.src = debug_image;
            debugImage.classList.remove("hidden");
            debugPlaceholder.classList.add("hidden");
        }

        // Save to history
        addHistoryItem({
            prediction,
            confidence,
            debug_image: debug_image || originalThumb,
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
        });
    }

    // ------------------------------------------------------------------------
    // PREDICTION HISTORY (Browser Memory / LocalStorage)
    // ------------------------------------------------------------------------
    function getStoredHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (_) {
            return [];
        }
    }

    function saveStoredHistory(history) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 10)));
        } catch (_) {}
    }

    function addHistoryItem(item) {
        const history = getStoredHistory();
        history.unshift(item);
        saveStoredHistory(history);
        renderHistory();
    }

    function clearHistory() {
        try {
            localStorage.removeItem(STORAGE_KEY);
        } catch (_) {}
        renderHistory();
    }

    function loadHistory() {
        renderHistory();
    }

    function renderHistory() {
        const history = getStoredHistory();
        if (history.length === 0) {
            historyList.innerHTML = `<div class="history-empty">No predictions recorded yet. Draw a digit above and click Predict!</div>`;
            return;
        }

        historyList.innerHTML = "";
        history.forEach((item) => {
            const el = document.createElement("div");
            el.className = "history-item";
            el.innerHTML = `
                <img src="${item.debug_image}" alt="Digit ${item.prediction}" class="history-thumb">
                <div class="history-info">
                    <div class="history-digit-row">
                        <span class="history-digit">${item.prediction}</span>
                        <span class="history-conf">${item.confidence.toFixed(1)}%</span>
                    </div>
                    <span class="history-time">${item.timestamp}</span>
                </div>
            `;
            historyList.appendChild(el);
        });
    }

    // ------------------------------------------------------------------------
    // BACKEND HEALTH & STATUS MONITOR
    // ------------------------------------------------------------------------
    function setBackendStatus(state, message) {
        if (!modelStatus) return;
        modelStatus.classList.remove("status-checking", "status-online", "status-offline");

        if (state === "checking") {
            modelStatus.classList.add("status-checking");
            if (statusText) statusText.textContent = message || "Connecting...";
            if (modelStatusDot) {
                modelStatusDot.style.background = "#f59e0b";
                modelStatusDot.style.boxShadow = "0 0 8px #f59e0b";
            }
        } else if (state === "online") {
            modelStatus.classList.add("status-online");
            if (statusText) statusText.textContent = message || "Backend Online (PyTorch CNN)";
            if (modelStatusDot) {
                modelStatusDot.style.background = "#10b981";
                modelStatusDot.style.boxShadow = "0 0 8px #10b981";
            }
        } else {
            modelStatus.classList.add("status-offline");
            if (statusText) statusText.textContent = message || "Backend Unavailable";
            if (modelStatusDot) {
                modelStatusDot.style.background = "#ef4444";
                modelStatusDot.style.boxShadow = "0 0 8px #ef4444";
            }
        }
    }

    async function checkBackendHealth() {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3500);

            const res = await fetch(`${API_BASE_URL}/health`, {
                method: "GET",
                signal: controller.signal
            });
            clearTimeout(timeoutId);

            if (res.ok) {
                const data = await res.json();
                if (data.status === "online" && data.model_loaded) {
                    setBackendStatus(
                        "online",
                        `Backend Online (${data.model || "CNN"} • ${(data.device || "CPU").toUpperCase()})`
                    );
                    return;
                }
            }
            setBackendStatus("offline", "Backend Error");
        } catch (_) {
            setBackendStatus("offline", "Backend Unavailable (Port 5000)");
        }
    }

    // ------------------------------------------------------------------------
    // ALERT BANNER HELPERS
    // ------------------------------------------------------------------------
    function showAlert(msg) {
        alertBox.textContent = msg;
        alertBox.classList.remove("hidden");
    }

    function hideAlert() {
        alertBox.textContent = "";
        alertBox.classList.add("hidden");
    }

    // Start application
    init();
})();
