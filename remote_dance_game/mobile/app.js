(function() {
    const LEGACY_POSE_VERSION = '0.5.1675469404';
    const LOCAL_POSE_BASE = './vendor/pose';
    const POSE_SCRIPT_FALLBACKS = [
        `${LOCAL_POSE_BASE}/pose.js`,
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${LEGACY_POSE_VERSION}/pose.js`,
        `https://unpkg.com/@mediapipe/pose@${LEGACY_POSE_VERSION}/pose.js`,
    ];
    const POSE_ASSET_FALLBACKS = [
        LOCAL_POSE_BASE,
        `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${LEGACY_POSE_VERSION}`,
        `https://unpkg.com/@mediapipe/pose@${LEGACY_POSE_VERSION}`,
    ];

    const TASKS_VERSION = '0.10.14';
    const TASKS_MODULE_FALLBACKS = [
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/+esm`,
        `https://esm.sh/@mediapipe/tasks-vision@${TASKS_VERSION}`,
    ];
    const TASKS_WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`;
    const TASKS_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';
    const MAX_PEOPLE = 4;

    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get('session');
    const serverUrl = params.get('server') || '';

    let ws = null;
    let legacyPose = null;
    let multiPose = null;
    let trackerMode = 'loading';
    let seq = 0;
    let isConnected = false;
    let isCalibrating = true;
    let isPlaying = false;
    let lastSendMs = 0;
    let lastInferenceCaptureMs = 0;
    let estimatedLatencyMs = 0;
    let stableCalibrationFrames = 0;
    const TARGET_SEND_INTERVAL_MS = 24;
    const MAX_WS_BUFFERED_BYTES = 120000;
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 10;
    let gradeTimeout = null;
    let holdTimeout = null;
    let yeahTimeout = null;
    let nextTrackId = 0;
    const personTracks = new Map();

    const COLORS = ['#6cff55', '#bc67ff', '#ffcc42', '#46e5ff', '#ff6b9d', '#ff8c42'];

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

    function setHint(message) {
        hintText.classList.remove('hidden');
        hintText.textContent = message;
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
        ws = new WebSocket(getWsUrl());
        ws.onopen = () => {
            isConnected = true;
            reconnectAttempts = 0;
            connDot.className = 'dot green';
            connText.textContent = 'Connected';
            ws.send(JSON.stringify({ type: 'ping', timestamp_ms: Date.now() }));
        };
        ws.onmessage = event => handleMessage(JSON.parse(event.data));
        ws.onclose = () => {
            isConnected = false;
            connDot.className = 'dot red';
            connText.textContent = 'Disconnected';
            scheduleReconnect();
        };
        ws.onerror = err => console.error('WebSocket error', err);
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
                isCalibrating = false;
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
                if (msg.is_move_grade !== false) showGrade(msg.grade, msg.score, msg.combo, msg.player_slot || 0);
                if (msg.hold_state === 'entering' || msg.hold_state === 'holding') showHold();
                if (msg.hold_state === 'yeah') showYeah();
                break;
            case 'pong': {
                const clientTs = msg.client_timestamp_ms || 0;
                const latency = clientTs > 0 ? Math.max(0, (Date.now() - clientTs) / 2) : 0;
                estimatedLatencyMs = estimatedLatencyMs > 0 ? estimatedLatencyMs * 0.78 + latency * 0.22 : latency;
                latencyText.textContent = `${Math.round(estimatedLatencyMs)}ms`;
                break;
            }
            case 'error':
                console.error('Server error:', msg.message);
                break;
        }
    }

    function showGrade(grade, score, combo, slot) {
        if (gradeTimeout) clearTimeout(gradeTimeout);
        gradeDisplay.className = '';
        gradeDisplay.classList.remove('hidden');
        gradeDisplay.textContent = `P${slot + 1}  ${String(grade || '').toUpperCase()}`;
        gradeDisplay.classList.add(String(grade || '').toLowerCase() === 'x' ? 'x' : String(grade || '').toLowerCase());
        if (score !== undefined) {
            scoreDisplay.classList.remove('hidden');
            scoreDisplay.textContent = `+${score}`;
        }
        if (combo !== undefined && combo > 1) {
            comboDisplay.classList.remove('hidden');
            comboDisplay.textContent = `${combo}x Combo`;
        }
        gradeTimeout = setTimeout(() => {
            gradeDisplay.classList.add('hidden');
            scoreDisplay.classList.add('hidden');
            comboDisplay.classList.add('hidden');
        }, 650);
    }

    function showHold() {
        holdIndicator.classList.remove('hidden');
        if (holdTimeout) clearTimeout(holdTimeout);
        holdTimeout = setTimeout(() => holdIndicator.classList.add('hidden'), 1600);
    }

    function showYeah() {
        holdIndicator.classList.add('hidden');
        yeahIndicator.classList.remove('hidden');
        if (yeahTimeout) clearTimeout(yeahTimeout);
        yeahTimeout = setTimeout(() => yeahIndicator.classList.add('hidden'), 1200);
    }

    function isLocalHost(hostname) {
        return ['localhost', '127.0.0.1', '::1'].includes(hostname);
    }

    function requireSecureCameraContext() {
        if (window.isSecureContext || isLocalHost(location.hostname)) return;
        throw new Error('Phone camera requires HTTPS. Open the QR link through HTTPS.');
    }

    function waitForVideoReady(timeoutMs = 8000) {
        if (videoEl.readyState >= 2 && videoEl.videoWidth > 0 && videoEl.videoHeight > 0) return Promise.resolve();
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

    async function importFirst(urls) {
        let lastError = null;
        for (const url of urls) {
            try {
                return await import(url);
            } catch (error) {
                lastError = error;
            }
        }
        throw lastError || new Error('No MediaPipe Tasks module source worked');
    }

    async function initMultiPose() {
        const visionModule = await importFirst(TASKS_MODULE_FALLBACKS);
        const FilesetResolver = visionModule.FilesetResolver;
        const PoseLandmarker = visionModule.PoseLandmarker;
        if (!FilesetResolver || !PoseLandmarker) throw new Error('PoseLandmarker module unavailable');

        const fileset = await FilesetResolver.forVisionTasks(TASKS_WASM_BASE);
        multiPose = await PoseLandmarker.createFromOptions(fileset, {
            baseOptions: {
                modelAssetPath: TASKS_MODEL,
                delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numPoses: MAX_PEOPLE,
            minPoseDetectionConfidence: 0.50,
            minPosePresenceConfidence: 0.50,
            minTrackingConfidence: 0.58,
        });
        trackerMode = 'multipose';
        return true;
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
            script.onload = () => { clearTimeout(timer); resolve(); };
            script.onerror = () => {
                clearTimeout(timer);
                script.remove();
                reject(new Error(`Failed to load script: ${url}`));
            };
            document.head.appendChild(script);
        });
    }

    async function initLegacyPose() {
        if (typeof window.Pose !== 'function') {
            let loaded = false;
            for (const url of POSE_SCRIPT_FALLBACKS) {
                try {
                    await loadScript(url);
                    loaded = typeof window.Pose === 'function';
                    if (loaded) break;
                } catch (_) {}
            }
            if (!loaded) throw new Error('Legacy Pose library unavailable');
        }

        let lastErr = null;
        for (const base of POSE_ASSET_FALLBACKS) {
            try {
                const candidate = new window.Pose({ locateFile: file => `${base}/${file}` });
                candidate.setOptions({
                    modelComplexity: 1,
                    smoothLandmarks: true,
                    enableSegmentation: false,
                    smoothSegmentation: false,
                    minDetectionConfidence: 0.58,
                    minTrackingConfidence: 0.60,
                });
                candidate.onResults(results => {
                    if (!results.poseLandmarks) {
                        handleMultiResults([], []);
                        return;
                    }
                    handleMultiResults(
                        [results.poseLandmarks],
                        results.poseWorldLandmarks ? [results.poseWorldLandmarks] : []
                    );
                });
                await candidate.send({ image: videoEl });
                legacyPose = candidate;
                trackerMode = 'legacy';
                return true;
            } catch (err) {
                lastErr = err;
            }
        }
        throw lastErr || new Error('Legacy pose assets unavailable');
    }

    async function initPose() {
        await waitForVideoReady();
        try {
            await initMultiPose();
            trackText.textContent = 'Multi-person tracking';
            return true;
        } catch (multiError) {
            console.warn('MultiPose init failed, using single-person fallback', multiError);
            try {
                await initLegacyPose();
                trackText.textContent = 'Single-person fallback';
                setHint('Multi-person model could not load. Single-player tracking is active.');
                return true;
            } catch (legacyError) {
                console.error('Pose initialization failed', legacyError);
                setHint(`Pose tracking failed: ${legacyError.message || legacyError}`);
                return false;
            }
        }
    }

    function normalizeLandmarks(raw) {
        return (raw || []).slice(0, 33).map(lm => ({
            x: Number(Number(lm.x || 0).toFixed(4)),
            y: Number(Number(lm.y || 0).toFixed(4)),
            z: Number(Number(lm.z || 0).toFixed(4)),
            v: Number(Number(lm.visibility ?? lm.presence ?? 1).toFixed(4)),
        }));
    }

    function centerOf(lm) {
        if (!lm || lm.length < 25) return {x:.5,y:.5,scale:.1,signature:1};
        const hipX = (lm[23].x + lm[24].x) * .5;
        const hipY = (lm[23].y + lm[24].y) * .5;
        const shX = (lm[11].x + lm[12].x) * .5;
        const shY = (lm[11].y + lm[12].y) * .5;
        const torso = Math.max(.05, Math.hypot(shX - hipX, shY - hipY));
        const shoulderWidth = Math.hypot(lm[11].x - lm[12].x, lm[11].y - lm[12].y);
        const hipWidth = Math.hypot(lm[23].x - lm[24].x, lm[23].y - lm[24].y);
        const signature = (shoulderWidth + hipWidth * .7) / torso;
        return {x: hipX, y: (hipY + shY) * .5, scale:torso, signature};
    }

    function trackMatchCost(track, det, now) {
        const dt = Math.min(.45, Math.max(.016, now - track.lastSeen) / 1000);
        const predictedX = track.x + (track.vx || 0) * dt;
        const predictedY = track.y + (track.vy || 0) * dt;
        const spatial = Math.hypot(predictedX - det.center.x, predictedY - det.center.y);
        const scaleDelta = Math.abs(track.scale - det.center.scale) /
            Math.max(track.scale, det.center.scale, .05);
        const signatureDelta = Math.abs((track.signature || det.center.signature) - det.center.signature) /
            Math.max(Math.abs(track.signature || 1), Math.abs(det.center.signature || 1), .25);

        // Prediction is dominant, while body proportions and apparent size keep
        // two dancers from swapping identities when their paths cross.
        return spatial + .12 * scaleDelta + .085 * signatureDelta;
    }

    function bestGlobalAssignment(tracks, detections, now) {
        let bestPairs = [];
        let bestCost = Infinity;
        const used = new Set();

        function visit(trackIndex, pairs, cost) {
            if (trackIndex >= tracks.length) {
                if (pairs.length > bestPairs.length ||
                    (pairs.length === bestPairs.length && cost < bestCost)) {
                    bestPairs = pairs.slice();
                    bestCost = cost;
                }
                return;
            }

            // A temporarily occluded player may legitimately have no detection.
            visit(trackIndex + 1, pairs, cost);

            const track = tracks[trackIndex];
            for (let di = 0; di < detections.length; di++) {
                if (used.has(di)) continue;
                const matchCost = trackMatchCost(track, detections[di], now);
                if (matchCost > .38) continue;
                used.add(di);
                pairs.push({track, det:detections[di], di});
                visit(trackIndex + 1, pairs, cost + matchCost);
                pairs.pop();
                used.delete(di);
            }
        }

        visit(0, [], 0);
        return bestPairs;
    }

    function associatePeople(poseLandmarks, worldLandmarks) {
        const now = Date.now();

        // Remove truly expired identities before considering new dancers.
        for (const [id, track] of personTracks.entries()) {
            if (now - track.lastSeen > 3500) personTracks.delete(id);
        }

        const detections = poseLandmarks.map((raw, index) => {
            const landmarks = normalizeLandmarks(raw);
            const world = normalizeLandmarks(worldLandmarks[index] || []);
            const center = centerOf(landmarks);
            const trackingScore = landmarks.length
                ? landmarks.reduce((sum, landmark) => sum + (landmark.v || 0), 0) / landmarks.length
                : 0;
            return {index, landmarks, world, center, trackingScore};
        }).filter(det =>
            det.landmarks.length >= 29 &&
            det.trackingScore >= .30 &&
            det.center.scale >= .045
        );

        const activeTracks = [...personTracks.values()]
            .filter(track => now - track.lastSeen < 1400)
            .sort((a,b) => a.id - b.id);
        const assigned = bestGlobalAssignment(activeTracks, detections, now);
        const usedDetections = new Set(assigned.map(pair => pair.di));

        detections.forEach((det, di) => {
            if (usedDetections.has(di)) return;
            if (personTracks.size >= MAX_PEOPLE) {
                const oldest = [...personTracks.values()].sort((a,b) => a.lastSeen - b.lastSeen)[0];
                if (!oldest || now - oldest.lastSeen < 1400) return;
                personTracks.delete(oldest.id);
            }
            const id = nextTrackId++;
            const track = {
                id,
                x:det.center.x,
                y:det.center.y,
                scale:det.center.scale,
                signature:det.center.signature,
                vx:0,
                vy:0,
                lastSeen:now,
                seenFrames:0,
            };
            personTracks.set(id, track);
            assigned.push({track, det, di});
        });

        const output = [];
        for (const {track, det} of assigned) {
            const previousSeen = track.lastSeen;
            const dt = Math.max(.016, Math.min(.35, (now - previousSeen) / 1000));
            const measuredVx = (det.center.x - track.x) / dt;
            const measuredVy = (det.center.y - track.y) / dt;
            track.vx = (track.vx || 0) * .76 + Math.max(-1.8, Math.min(1.8, measuredVx)) * .24;
            track.vy = (track.vy || 0) * .76 + Math.max(-1.8, Math.min(1.8, measuredVy)) * .24;
            track.x = track.x * .62 + det.center.x * .38;
            track.y = track.y * .62 + det.center.y * .38;
            track.scale = track.scale * .76 + det.center.scale * .24;
            track.signature = (track.signature || det.center.signature) * .86 + det.center.signature * .14;
            track.lastSeen = now;
            track.seenFrames = (track.seenFrames || 0) + 1;

            // One-frame detections are usually passers-by or model noise. A
            // dancer becomes a player after two consecutive observations.
            if (track.seenFrames < 2) continue;
            output.push({
                player_id: `p${track.id}`,
                tracking_score: Number(det.trackingScore.toFixed(3)),
                landmarks: det.landmarks,
                world_landmarks: det.world,
                color: COLORS[track.id % COLORS.length],
            });
        }

        return output.sort((a,b) => Number(a.player_id.slice(1)) - Number(b.player_id.slice(1)));
    }

    const CONNECTIONS = [
        [11,12],[11,13],[13,15],[12,14],[14,16],
        [11,23],[12,24],[23,24],[23,25],[25,27],
        [24,26],[26,28],[27,29],[29,31],[28,30],[30,32],
    ];

    function drawPeople(people) {
        skeletonCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
        const w = skeletonCanvas.width;
        const h = skeletonCanvas.height;
        people.forEach((person, personIndex) => {
            const lm = person.landmarks;
            const color = person.color || COLORS[personIndex % COLORS.length];
            skeletonCtx.strokeStyle = color;
            skeletonCtx.fillStyle = color;
            skeletonCtx.lineWidth = 4;
            CONNECTIONS.forEach(([a,b]) => {
                if (!lm[a] || !lm[b] || Math.min(lm[a].v, lm[b].v) < .25) return;
                skeletonCtx.beginPath();
                skeletonCtx.moveTo(lm[a].x * w, lm[a].y * h);
                skeletonCtx.lineTo(lm[b].x * w, lm[b].y * h);
                skeletonCtx.stroke();
            });
            [11,12,13,14,15,16,23,24,25,26,27,28].forEach(idx => {
                if (!lm[idx] || lm[idx].v < .25) return;
                skeletonCtx.beginPath();
                skeletonCtx.arc(lm[idx].x * w, lm[idx].y * h, 5, 0, Math.PI * 2);
                skeletonCtx.fill();
            });
            const head = lm[0];
            if (head) {
                skeletonCtx.font = 'bold 22px sans-serif';
                skeletonCtx.fillText(`P${Number(person.player_id.slice(1)) + 1}`, head.x * w + 12, head.y * h - 12);
            }
        });
    }

    function handleMultiResults(poseLandmarks, worldLandmarks) {
        const people = associatePeople(poseLandmarks || [], worldLandmarks || []);
        drawPeople(people);

        if (people.length) {
            trackDot.className = 'dot green';
            trackText.textContent = `${people.length} player${people.length === 1 ? '' : 's'} tracked`;
            if (isCalibrating && people.every(p => p.tracking_score >= .45)) stableCalibrationFrames++;
            else if (isCalibrating) stableCalibrationFrames = Math.max(0, stableCalibrationFrames - 1);
        } else {
            trackDot.className = 'dot red';
            trackText.textContent = 'No full body';
            stableCalibrationFrames = 0;
        }

        if (isCalibrating && stableCalibrationFrames >= 10) {
            isCalibrating = false;
            calPanel.classList.remove('hidden');
            calChecks.innerHTML = `<div class="cal-check pass">✓ ${people.length} player${people.length === 1 ? '' : 's'} separated</div><div class="cal-check pass">✓ Full body tracking stable</div>`;
            calMessage.textContent = 'Ready to dance';
            startBtn.disabled = false;
            startBtn.textContent = 'Ready!';
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:'control', action:'ready'}));
        }

        if (!ws || ws.readyState !== WebSocket.OPEN || !people.length) return;
        const now = Date.now();
        if (now - lastSendMs < TARGET_SEND_INTERVAL_MS || ws.bufferedAmount >= MAX_WS_BUFFERED_BYTES) return;

        ws.send(JSON.stringify({
            type: 'multi_pose_frame',
            session_id: sessionId,
            seq: seq++,
            timestamp_ms: lastInferenceCaptureMs || now,
            estimated_latency_ms: Math.round(estimatedLatencyMs),
            poses: people.map(p => ({
                player_id: p.player_id,
                tracking_score: p.tracking_score,
                landmarks: p.landmarks,
                world_landmarks: p.world_landmarks,
            })),
        }));
        lastSendMs = now;
    }

    async function initCamera() {
        try {
            requireSecureCameraContext();
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('getUserMedia unavailable');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: {ideal: 1920},
                    height: {ideal: 1080},
                    frameRate: {ideal: 60, min: 30},
                },
                audio: false,
            });
            videoEl.srcObject = stream;
            videoEl.muted = true;
            videoEl.playsInline = true;
            await videoEl.play();
            await waitForVideoReady();

            const settings = stream.getVideoTracks()[0].getSettings();
            skeletonCanvas.width = videoEl.videoWidth || settings.width || 1280;
            skeletonCanvas.height = videoEl.videoHeight || settings.height || 720;
            return true;
        } catch (e) {
            console.error('Camera initialization failed', e);
            setHint(`Camera failed: ${e.message || e}`);
            trackDot.className = 'dot red';
            trackText.textContent = 'Camera error';
            return false;
        }
    }

    async function processFrame() {
        if (videoEl.readyState >= 2) {
            try {
                lastInferenceCaptureMs = Date.now();
                if (trackerMode === 'multipose' && multiPose) {
                    const results = multiPose.detectForVideo(videoEl, performance.now());
                    handleMultiResults(results.landmarks || [], results.worldLandmarks || []);
                } else if (trackerMode === 'legacy' && legacyPose) {
                    await legacyPose.send({image: videoEl});
                }
            } catch (_) {}
        }
        if (typeof videoEl.requestVideoFrameCallback === 'function') videoEl.requestVideoFrameCallback(processFrame);
        else requestAnimationFrame(processFrame);
    }

    function startFrameLoop() {
        if (typeof videoEl.requestVideoFrameCallback === 'function') videoEl.requestVideoFrameCallback(processFrame);
        else requestAnimationFrame(processFrame);
    }

    startBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            isCalibrating = false;
            calPanel.classList.add('hidden');
            ws.send(JSON.stringify({type:'control', action:'ready'}));
            startBtn.textContent = 'Waiting for start...';
            startBtn.disabled = true;
        }
    });

    async function init() {
        connDot.className = 'dot yellow';
        connText.textContent = 'Preparing camera...';
        startBtn.disabled = true;
        startBtn.textContent = 'Preparing camera...';

        if (!await initCamera()) {
            connText.textContent = 'Camera blocked';
            startBtn.textContent = 'Camera error';
            return;
        }

        connText.textContent = 'Loading multi-person tracker...';
        startBtn.textContent = 'Loading tracker...';
        if (!await initPose()) {
            connText.textContent = 'Tracking error';
            startBtn.textContent = 'Tracking error';
            return;
        }

        connText.textContent = 'Connecting...';
        connect();
        startBtn.textContent = 'Stand in frame...';
        startFrameLoop();

        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type:'ping', timestamp_ms:Date.now()}));
        }, 4000);
    }

    init();
})();
