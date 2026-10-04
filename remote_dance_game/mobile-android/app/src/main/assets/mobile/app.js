(function() {
    const POSE_VERSION = '0.5.1675469404';
    const LOCAL_POSE_BASE = './vendor/pose';
    const POSE_SCRIPT_FALLBACKS = [
        `${LOCAL_POSE_BASE}/pose.js`,
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/pose.js`,
        `https://unpkg.com/@mediapipe/pose@${POSE_VERSION}/pose.js`,
    ];
    const POSE_ASSET_FALLBACKS = [
        LOCAL_POSE_BASE,
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}`,
        `https://unpkg.com/@mediapipe/pose@${POSE_VERSION}`,
    ];

    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session');
    const serverUrl = params.get('server') || '';
    
    let ws = null;
    let pose = null;
    let seq = 0;
    let isConnected = false;
    let isTracking = false;
    let isCalibrating = true;
    let isPlaying = false;
    let lastPoseResult = null;
    let lastSendMs = 0;
    let lastInferenceCaptureMs = 0;
    const TARGET_SEND_INTERVAL_MS = 24;
    const MAX_WS_BUFFERED_BYTES = 120000;
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 10;
    let gradeTimeout = null;
    let holdTimeout = null;
    let yeahTimeout = null;

    const connDot = document.getElementById('conn-dot');
    const connText = document.getElementById('conn-text');
    const trackDot = document.getElementById('track-dot');
    const trackText = document.getElementById('track-text');
    const latencyText = document.getElementById('latency-text');
    const gradeDisplay = document.getElementById('grade-display');
    const scoreDisplay = document.getElementById('score-display');
    const comboDisplay = document.getElementById('combo-display');
    const holdIndicator = document.getElementById('hold-indicator');
    const yeahIndicator = document.getElementById('yeah-indicator');
    const calPanel = document.getElementById('calibration-panel');
    const calChecks = document.getElementById('calibration-checks');
    const calMessage = document.getElementById('calibration-message');
    const hintText = document.getElementById('hint-text');
    const startBtn = document.getElementById('start-btn');
    const skeletonCanvas = document.getElementById('skeleton-overlay');
    const skeletonCtx = skeletonCanvas.getContext('2d');
    const videoEl = document.getElementById('camera-preview');

    if (!sessionId) {
        connText.textContent = 'No session ID';
        return;
    }

    function loadScript(url, timeoutMs = 8000) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            const timer = setTimeout(() => {
                script.remove();
                reject(new Error(`Timeout loading script: ${url}`));
            }, timeoutMs);

            script.async = true;
            script.crossOrigin = 'anonymous';
            script.src = url;
            script.onload = () => {
                clearTimeout(timer);
                resolve();
            };
            script.onerror = () => {
                clearTimeout(timer);
                script.remove();
                reject(new Error(`Failed to load script: ${url}`));
            };

            document.head.appendChild(script);
        });
    }

    async function loadFirstAvailableScript(urls) {
        let lastErr = null;
        for (const url of urls) {
            try {
                await loadScript(url);
                return url;
            } catch (err) {
                lastErr = err;
            }
        }
        throw lastErr || new Error('No script source available');
    }

    async function ensurePoseLibrary() {
        if (typeof window.Pose === 'function') {
            return 'already-loaded';
        }
        const loadedUrl = await loadFirstAvailableScript(POSE_SCRIPT_FALLBACKS);
        if (typeof window.Pose !== 'function') {
            throw new Error('Pose constructor not found after script load');
        }
        return loadedUrl;
    }

    function getWsUrl() {
        if (serverUrl) {
            const normalized = /^[a-z]+:\/\//i.test(serverUrl) ? serverUrl : `${location.protocol}//${serverUrl}`;
            const base = new URL(normalized);
            const proto = base.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${proto}//${base.host}/ws/phone/${sessionId}`;
        }
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${location.host}/ws/phone/${sessionId}`;
    }

    function connect() {
        const url = getWsUrl();
        console.log('Connecting to', url);
        ws = new WebSocket(url);

        ws.onopen = () => {
            isConnected = true;
            reconnectAttempts = 0;
            connDot.className = 'dot green';
            connText.textContent = 'Connected';
            ws.send(JSON.stringify({ type: 'ping', timestamp_ms: Date.now() }));
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        };

        ws.onclose = () => {
            isConnected = false;
            connDot.className = 'dot red';
            connText.textContent = 'Disconnected';
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.error('WebSocket error', err);
        };
    }

    function scheduleReconnect() {
        if (reconnectAttempts >= MAX_RECONNECT) {
            connText.textContent = 'Connection lost';
            return;
        }
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
        reconnectAttempts++;
        connText.textContent = `Reconnecting (${reconnectAttempts})...`;
        connDot.className = 'dot yellow';
        setTimeout(connect, delay);
    }

    function handleMessage(msg) {
        switch (msg.type) {
            case 'start':
                isPlaying = true;
                startBtn.textContent = 'Dancing!';
                startBtn.disabled = true;
                break;
            case 'pause':
                startBtn.textContent = 'Paused';
                break;
            case 'resume':
                startBtn.textContent = 'Dancing!';
                break;
            case 'game_over':
                startBtn.textContent = 'Finished';
                break;
            case 'score_feedback':
                if (msg.is_move_grade !== false) {
                    showGrade(msg.grade, msg.score, msg.combo);
                }
                if (msg.hold_state === 'entering' || msg.hold_state === 'holding') {
                    showHold();
                }
                if (msg.hold_state === 'yeah') {
                    showYeah();
                }
                break;
            case 'calibration_result':
                showCalibration(msg.checks, msg.ready);
                break;
            case 'pong':
                const clientTs = msg.client_timestamp_ms || 0;
                const latency = clientTs > 0 ? Math.max(0, Math.round((Date.now() - clientTs) / 2)) : 0;
                latencyText.textContent = `${latency}ms`;
                break;
            case 'error':
                console.error('Server error:', msg.message);
                break;
        }
    }

    function showGrade(grade, score, combo) {
        if (gradeTimeout) clearTimeout(gradeTimeout);
        
        gradeDisplay.className = '';
        gradeDisplay.classList.remove('hidden');
        gradeDisplay.textContent = grade.toUpperCase();
        gradeDisplay.classList.add(grade.toLowerCase() === 'x' ? 'x' : grade.toLowerCase());

        if (score !== undefined) {
            scoreDisplay.classList.remove('hidden');
            scoreDisplay.textContent = `+${score}`;
        }

        if (combo !== undefined && combo > 1) {
            comboDisplay.classList.remove('hidden');
            comboDisplay.textContent = `${combo}x Combo!`;
        }

        gradeTimeout = setTimeout(() => {
            gradeDisplay.classList.add('hidden');
            scoreDisplay.classList.add('hidden');
            comboDisplay.classList.add('hidden');
        }, 800);
    }

    function showHold() {
        holdIndicator.classList.remove('hidden');
        if (holdTimeout) clearTimeout(holdTimeout);
        holdTimeout = setTimeout(() => holdIndicator.classList.add('hidden'), 2000);
    }

    function showYeah() {
        holdIndicator.classList.add('hidden');
        yeahIndicator.classList.remove('hidden');
        if (yeahTimeout) clearTimeout(yeahTimeout);
        yeahTimeout = setTimeout(() => yeahIndicator.classList.add('hidden'), 1500);
    }

    function showCalibration(checks, ready) {
        calPanel.classList.remove('hidden');
        calChecks.innerHTML = '';
        checks.forEach(check => {
            const div = document.createElement('div');
            div.className = `cal-check ${check.passed ? 'pass' : 'fail'}`;
            div.textContent = `${check.passed ? '✓' : '✗'} ${check.message}`;
            calChecks.appendChild(div);
        });
        calMessage.textContent = ready ? 'Calibration complete! Ready to dance!' : 'Adjust position and wait...';
        
        if (ready) {
            isCalibrating = false;
            startBtn.disabled = false;
            startBtn.textContent = 'Ready!';
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'control', action: 'ready' }));
            }
        }
    }

    function isLocalHost(hostname) {
        return ['localhost', '127.0.0.1', '::1'].includes(hostname);
    }

    function requireSecureCameraContext() {
        if (window.isSecureContext || isLocalHost(location.hostname)) {
            return;
        }
        throw new Error(
            'Phone browser camera requires HTTPS. Open the mobile page through an HTTPS tunnel/domain, not http://<pc-ip>:8000.'
        );
    }

    function waitForVideoReady(timeoutMs = 8000) {
        if (videoEl.readyState >= 2 && videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
            return Promise.resolve();
        }
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                cleanup();
                reject(new Error('Camera video stream did not become ready in time'));
            }, timeoutMs);

            function cleanup() {
                clearTimeout(timer);
                videoEl.removeEventListener('loadedmetadata', onReady);
                videoEl.removeEventListener('canplay', onReady);
            }

            function onReady() {
                if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
                    cleanup();
                    resolve();
                }
            }

            videoEl.addEventListener('loadedmetadata', onReady);
            videoEl.addEventListener('canplay', onReady);
        });
    }

    function setHint(message) {
        hintText.classList.remove('hidden');
        hintText.textContent = message;
    }

    async function initPose() {
        try {
            await waitForVideoReady();
            const loadedScriptUrl = await ensurePoseLibrary();
            const preferredAssetIndex = Math.max(
                0,
                POSE_SCRIPT_FALLBACKS.findIndex((url) => loadedScriptUrl !== 'already-loaded' && loadedScriptUrl.includes(url.replace('/pose.js', '')))
            );

            let lastErr = null;
            for (let assetBaseIndex = preferredAssetIndex; assetBaseIndex < POSE_ASSET_FALLBACKS.length; assetBaseIndex++) {
                try {
                    const candidate = new window.Pose({
                        locateFile: (file) => `${POSE_ASSET_FALLBACKS[assetBaseIndex]}/${file}`,
                    });

                    candidate.setOptions({
                        modelComplexity: 1,
                        smoothLandmarks: true,
                        enableSegmentation: false,
                        smoothSegmentation: false,
                        minDetectionConfidence: 0.58,
                        minTrackingConfidence: 0.60,
                    });
                    candidate.onResults(onPoseResults);

                    // Warm up once to make sure wasm/model assets are really reachable.
                    await candidate.send({ image: videoEl });
                    pose = candidate;
                    console.log('MediaPipe Pose initialized with assets:', POSE_ASSET_FALLBACKS[assetBaseIndex]);
                    return true;
                } catch (err) {
                    lastErr = err;
                    console.warn('MediaPipe asset source failed:', POSE_ASSET_FALLBACKS[assetBaseIndex], err);
                }
            }
            throw lastErr || new Error('No MediaPipe asset source worked');
        } catch (e) {
            console.error('Failed to init MediaPipe Pose:', e);
            setHint(`MediaPipe failed to load: ${e.message || e}. If you use the phone browser, open the QR link through HTTPS and make sure the phone has internet access or local vendor/pose files are installed.`);
            return false;
        }
    }

    function onPoseResults(results) {
        skeletonCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);

        if (results.poseLandmarks) {
            isTracking = true;
            trackDot.className = 'dot green';
            trackText.textContent = 'Tracking';

            const w = skeletonCanvas.width;
            const h = skeletonCanvas.height;
            
            drawSkeleton(results.poseLandmarks, w, h);

            const landmarks = results.poseLandmarks.map(lm => ({
                x: parseFloat(lm.x.toFixed(4)),
                y: parseFloat(lm.y.toFixed(4)),
                z: parseFloat(lm.z.toFixed(4)),
                v: parseFloat((lm.visibility || 0).toFixed(4)),
            }));

            const worldLandmarks = results.poseWorldLandmarks ? 
                results.poseWorldLandmarks.map(lm => ({
                    x: parseFloat(lm.x.toFixed(4)),
                    y: parseFloat(lm.y.toFixed(4)),
                    z: parseFloat(lm.z.toFixed(4)),
                    v: parseFloat((lm.visibility || 0).toFixed(4)),
                })) : [];

            const trackingScore = landmarks.reduce((s, l) => s + l.v, 0) / landmarks.length;

            const frame = {
                type: isCalibrating ? 'calibration_frame' : 'pose_frame',
                session_id: sessionId,
                seq: seq++,
                timestamp_ms: lastInferenceCaptureMs || Date.now(),
                landmarks: landmarks,
                world_landmarks: worldLandmarks,
                tracking_score: parseFloat(trackingScore.toFixed(3)),
                device_rotation: 0,
            };

            if (ws && ws.readyState === WebSocket.OPEN) {
                const now = Date.now();
                if (now - lastSendMs >= TARGET_SEND_INTERVAL_MS && ws.bufferedAmount < MAX_WS_BUFFERED_BYTES) {
                    ws.send(JSON.stringify(frame));
                    lastSendMs = now;
                }
            }

            lastPoseResult = results;
        } else {
            isTracking = false;
            trackDot.className = 'dot red';
            trackText.textContent = 'No tracking';
        }
    }

    const CONNECTIONS = [
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
        [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
        [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32],
        [15, 17], [16, 18], [15, 19], [16, 20], [17, 19], [18, 20],
    ];

    function drawSkeleton(landmarks, w, h) {
        skeletonCtx.strokeStyle = '#00e5ff';
        skeletonCtx.lineWidth = 2;

        CONNECTIONS.forEach(([a, b]) => {
            if (a < landmarks.length && b < landmarks.length) {
                const la = landmarks[a];
                const lb = landmarks[b];
                if (la.visibility > 0.3 && lb.visibility > 0.3) {
                    skeletonCtx.beginPath();
                    skeletonCtx.moveTo(la.x * w, la.y * h);
                    skeletonCtx.lineTo(lb.x * w, lb.y * h);
                    skeletonCtx.stroke();
                }
            }
        });

        skeletonCtx.fillStyle = '#ff0';
        landmarks.forEach((lm, i) => {
            if (lm.visibility > 0.3 && i < 33) {
                skeletonCtx.beginPath();
                skeletonCtx.arc(lm.x * w, lm.y * h, 3, 0, 2 * Math.PI);
                skeletonCtx.fill();
            }
        });
    }

    async function initCamera() {
        try {
            requireSecureCameraContext();
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('getUserMedia is not available in this browser');
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 60, min: 30 },
                },
                audio: false,
            });
            videoEl.srcObject = stream;
            videoEl.muted = true;
            videoEl.playsInline = true;
            await videoEl.play();
            await waitForVideoReady();

            const track = stream.getVideoTracks()[0];
            const settings = track.getSettings();
            skeletonCanvas.width = videoEl.videoWidth || settings.width || 1280;
            skeletonCanvas.height = videoEl.videoHeight || settings.height || 720;

            console.log('Camera initialized', skeletonCanvas.width, 'x', skeletonCanvas.height);
            return true;
        } catch (e) {
            console.error('Camera initialization failed:', e);
            setHint(`Camera failed: ${e.message || e}`);
            trackDot.className = 'dot red';
            trackText.textContent = 'Camera error';
            return false;
        }
    }

    let frameCount = 0;
    async function processFrame(now, metadata) {
        if (pose && videoEl.readyState >= 2) {
            try {
                lastInferenceCaptureMs = Date.now();
                await pose.send({ image: videoEl });
            } catch (e) {
                // Skip only this camera frame. Keeping inference serial avoids
                // a growing latency queue on slower phones.
            }
        }
        frameCount++;
        if (typeof videoEl.requestVideoFrameCallback === 'function') {
            videoEl.requestVideoFrameCallback(processFrame);
        } else {
            requestAnimationFrame(processFrame);
        }
    }

    function startFrameLoop() {
        if (typeof videoEl.requestVideoFrameCallback === 'function') {
            videoEl.requestVideoFrameCallback(processFrame);
        } else {
            requestAnimationFrame(processFrame);
        }
    }

    startBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            isCalibrating = false;
            calPanel.classList.add('hidden');
            ws.send(JSON.stringify({ type: 'control', action: 'ready' }));
            startBtn.textContent = 'Waiting for start...';
            startBtn.disabled = true;
        }
    });

    async function init() {
        connDot.className = 'dot yellow';
        connText.textContent = 'Preparing camera...';
        startBtn.disabled = true;
        startBtn.textContent = 'Preparing camera...';

        const cameraOk = await initCamera();
        if (!cameraOk) {
            connText.textContent = 'Camera blocked';
            startBtn.textContent = 'Camera error';
            return;
        }

        connText.textContent = 'Loading MediaPipe...';
        startBtn.textContent = 'Loading MediaPipe...';
        const poseOk = await initPose();
        if (!poseOk) {
            connText.textContent = 'MediaPipe error';
            startBtn.textContent = 'MediaPipe error';
            return;
        }

        connText.textContent = 'Connecting...';
        connect();
        startBtn.textContent = 'Calibrating...';

        startFrameLoop();

        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping', timestamp_ms: Date.now() }));
            }
        }, 5000);
    }

    init();
})();
