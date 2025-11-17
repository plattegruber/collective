<script>
  import { onMount, onDestroy, tick } from 'svelte';
  import LoadingOverlay from './components/LoadingOverlay.svelte';
  import InstructionOverlay from './components/InstructionOverlay.svelte';
  import ArtworkOverlay from './components/ArtworkOverlay.svelte';
  import ReactionConfetti from './components/ReactionConfetti.svelte';
  import SplashScreen from './components/SplashScreen.svelte';
  import ToastHost from './components/ToastHost.svelte';
  import { fetchCountsForPieces, REACTION_EMOJIS } from './lib/reactions';

  const BASE_URL = import.meta.env.BASE_URL ?? '/';
  const CONFIDENCE_THRESHOLD = 0.7;
  const DISPLAY_CONFIDENCE = 0.78;
  const HIDE_CONFIDENCE = 0.6;
  const SMOOTHING_FACTOR = 0.55;
  const DECAY_FACTOR = 0.6;
  const SWITCH_MARGIN = 0.1;
  const MIN_BUFFER_CONFIDENCE = 0.05;
  const REACTION_REFRESH_INTERVAL_MS = 10_000;

  const phrases = ['Point me toward the art.'];

  const isBrowser = typeof window !== 'undefined';
  const resolvedPath = isBrowser ? window.location.pathname.toLowerCase() : '';
  const IS_TEST_MODE = isBrowser && (resolvedPath.endsWith('/test') || resolvedPath.endsWith('/test/'));

  const TEST_ARTWORK = {
    id: 'test:pollock_full_fathom_five',
    title: 'Full Fathom Five',
    byline: 'Jackson Pollock, 1947',
    materials: 'Oil on canvas',
    description:
      'Test mode mock entry so you can try reactions without the detector. Imagine you are standing before Pollock\'s Full Fathom Five.',
  };

  let artContent = $state({});
  let labels = $state({});
  let currentArtwork = $state(null);
  let session = null;
  let isOnnxLoaded = false;
  const detectionBuffer = new Map();
  let permissionState = $state('unknown');
  let permissionMessage = $state('');
  let showPermissionPrompt = $state(false);
  let isRequestingCamera = $state(false);
  let detectionStarted = false;
  let lastCameraError = $state('');
  let permissionStatusHandle;

  let videoEl;
  let stream;
  let animationFrameId;
  let phraseTimeout;

  let overlayVisible = $state(true);
  let shouldShimmer = $state(false);
  let phraseOpacity = $state(1);
  let phraseText = $state('');

  let showLoading = $state(true);
  let loadingMessage = $state('Loading detector model...');

  let artworkVisible = $state(false);
  const EMPTY_ARTWORK = {
    id: '',
    title: '',
    byline: '',
    materials: '',
    description: '',
  };

  let displayedArtwork = $state({ ...EMPTY_ARTWORK });

  const canRequestPermission = $derived(['prompt', 'denied', 'error'].includes(permissionState));

  let showSplash = $state(!IS_TEST_MODE);
  let hasStarted = false;
  let splashReady = $state(false);
  const DEFAULT_SPLASH_SUBTITLE = 'Point • Discover • Remember';
  let splashSubtitle = $state(DEFAULT_SPLASH_SUBTITLE);
  let splashStatus = $state('Preparing experience...');
  let cameraReady = $state(false);

  let reactionCounts = $state({});
  let countsRefreshTimer = null;
  let hasStartedCountsLoop = false;

  const CONFETTI_DURATION_MS = 3_200;
  let confettiVisible = $state(false);
  let confettiSeed = $state(0);
  let confettiCounts = $state([]);
  let confettiTimer = null;

  const getFrameCanvas = (() => {
    let canvasRef = null;
    let contextRef = null;
    let currentSize = 0;
    return (size) => {
      if (!canvasRef || currentSize !== size) {
        currentSize = size;
        if (typeof OffscreenCanvas === 'function') {
          canvasRef = new OffscreenCanvas(size, size);
        } else {
          canvasRef = document.createElement('canvas');
          canvasRef.width = size;
          canvasRef.height = size;
        }
        contextRef = canvasRef.getContext('2d');
      }
      return { canvas: canvasRef, context: contextRef };
    };
  })();

  function normalizeReactionCounts(list) {
    const map = new Map();
    if (Array.isArray(list)) {
      list.forEach((item) => {
        if (item && typeof item.emoji === 'string') {
          const count = Number(item.count ?? 0);
          map.set(item.emoji, Number.isFinite(count) ? count : 0);
        }
      });
    }
    return REACTION_EMOJIS.map((emoji) => ({
      emoji,
      count: map.get(emoji) ?? 0,
    }));
  }

  function getCountsForPiece(pieceId) {
    if (!pieceId) return normalizeReactionCounts(null);
    const entry = reactionCounts[pieceId];
    return normalizeReactionCounts(entry);
  }

  async function refreshAllReactionCounts() {
    if (!artContent || typeof artContent !== 'object') return;
    const pieceIds = Object.keys(artContent ?? {});
    if (pieceIds.length === 0) return;

    try {
      const countsMap = await fetchCountsForPieces(pieceIds, 4);
      if (!countsMap || countsMap.size === 0) {
        return;
      }

      const next = { ...reactionCounts };
      countsMap.forEach((value, key) => {
        next[key] = normalizeReactionCounts(value);
      });
      reactionCounts = next;
    } catch (error) {
      console.warn('Reaction counts refresh failed:', error);
    }
  }

  function startCountsRefreshLoop() {
    if (hasStartedCountsLoop) return;
    hasStartedCountsLoop = true;
    countsRefreshTimer = setInterval(() => {
      refreshAllReactionCounts();
    }, REACTION_REFRESH_INTERVAL_MS);
  }

  function handleReaction(event) {
    const detail = event?.detail;
    if (!detail || detail.throttled) return;
    const emoji = detail.emoji;
    if (typeof emoji !== 'string' || !emoji) return;

    const pieceId = displayedArtwork?.id;
    if (!pieceId) return;

    const nextCounts = getCountsForPiece(pieceId).map((item) =>
      item.emoji === emoji ? { ...item, count: item.count + 1 } : item,
    );

    reactionCounts = { ...reactionCounts, [pieceId]: nextCounts };

    if (confettiVisible) {
      confettiCounts = nextCounts;
      confettiSeed = Date.now();
    }
  }

  function triggerConfettiFor(pieceId) {
    if (!pieceId) return;
    confettiCounts = getCountsForPiece(pieceId);
    confettiSeed = Date.now();
    confettiVisible = true;
    if (confettiTimer) {
      clearTimeout(confettiTimer);
    }
    confettiTimer = setTimeout(() => {
      confettiVisible = false;
      confettiTimer = null;
    }, CONFETTI_DURATION_MS);
  }

  function setStatus(message) {
    loadingMessage = message;
    if (showSplash) {
      splashStatus = message;
    }
  }

  function pickNewPhrase() {
    const options = phraseText ? phrases.filter((phrase) => phrase !== phraseText) : phrases;
    const next = options[Math.floor(Math.random() * options.length)] ?? phraseText;
    shouldShimmer = false;
    phraseOpacity = 0;
    if (phraseTimeout) clearTimeout(phraseTimeout);
    phraseTimeout = setTimeout(() => {
      phraseText = next;
      phraseOpacity = 1;
      shouldShimmer = true;
    }, 140);
  }

  function hideOverlay() {
    overlayVisible = false;
  }

  function showInstructionOverlay() {
    overlayVisible = true;
  }

  async function loadArtworkContent() {
    const response = await fetch(`${BASE_URL}data/art-content.v2.json`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to load artwork content (${response.status})`);
    }
    try {
      artContent = await response.json();
    } catch (error) {
      throw new Error('Artwork content response is not valid JSON.');
    }
    console.log('Loaded artwork content:', Object.keys(artContent).length, 'pieces');

    await refreshAllReactionCounts();
    startCountsRefreshLoop();
  }

  async function loadModel() {
    const manifestResponse = await fetch(`${BASE_URL}model-manifest.json`, { cache: 'no-store' });
    if (!manifestResponse.ok) {
      throw new Error(`Failed to load model manifest (${manifestResponse.status})`);
    }

    let manifest;
    try {
      manifest = await manifestResponse.json();
    } catch (error) {
      throw new Error('Model manifest is not valid JSON.');
    }

    if (!manifest || typeof manifest !== 'object') {
      throw new Error('Model manifest is empty or invalid.');
    }

    console.log('Model manifest:', manifest);

    const labelsPath = manifest.labels
      ? `${BASE_URL}${manifest.labels}`
      : `${BASE_URL}models/detector/labels.json`;
    const labelsResponse = await fetch(labelsPath, { cache: 'no-store' });
    if (!labelsResponse.ok) {
      throw new Error(`Failed to load detector labels (${labelsResponse.status})`);
    }
    try {
      labels = await labelsResponse.json();
    } catch (error) {
      throw new Error('Detector labels file is not valid JSON.');
    }
    console.log('Loaded labels:', labels);

    if (!isOnnxLoaded) {
      await new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.min.js';
        script.onload = () => {
          isOnnxLoaded = true;
          resolve();
        };
        script.onerror = reject;
        document.head.appendChild(script);
      });
    }

    const modelPath = `${BASE_URL}${manifest.path ?? 'models/detector/model.onnx'}`;
    session = await window.ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm', 'webgl'],
    });
    console.log('Model loaded successfully');
  }

  function updateDetectionBuffer(detections) {
    const seenThisFrame = new Set();

    detections.forEach((box) => {
      const labelText = labels[box.label];
      if (!labelText) return;
      const currentValue = detectionBuffer.get(labelText) ?? 0;
      const smoothed = currentValue * (1 - SMOOTHING_FACTOR) + SMOOTHING_FACTOR * box.score;
      detectionBuffer.set(labelText, smoothed);
      seenThisFrame.add(labelText);
    });

    detectionBuffer.forEach((confidence, label) => {
      if (seenThisFrame.has(label)) return;
      const decayed = confidence * DECAY_FACTOR;
      if (decayed <= MIN_BUFFER_CONFIDENCE) {
        detectionBuffer.delete(label);
      } else {
        detectionBuffer.set(label, decayed);
      }
    });
  }

  function findArtworkByLabel(label) {
    return artContent[label] ? label : null;
  }

  function displayArtwork(artworkId) {
    const artwork = artContent[artworkId];
    if (!artwork) return;

    const bylineParts = [];
    if (artwork.artist) bylineParts.push(artwork.artist);
    if (artwork.year) bylineParts.push(artwork.year);

    displayedArtwork = {
      id: artworkId,
      title: artwork.title ?? '',
      byline: bylineParts.join(', '),
      materials: artwork.materials ?? '',
      description: artwork.description ?? '',
    };

    currentArtwork = artworkId;
    artworkVisible = true;
    hideOverlay();
    triggerConfettiFor(artworkId);
  }

  function clearArtwork() {
    if (!currentArtwork) return;
    artworkVisible = false;
    currentArtwork = null;
    displayedArtwork = { ...EMPTY_ARTWORK };
    confettiVisible = false;
    if (confettiTimer) {
      clearTimeout(confettiTimer);
      confettiTimer = null;
    }
    showInstructionOverlay();
    pickNewPhrase();
  }

  function updateArtworkOverlay(detections) {
    updateDetectionBuffer(detections);

    let bestLabel = null;
    let bestConfidence = 0;
    detectionBuffer.forEach((confidence, label) => {
      if (confidence > bestConfidence) {
        bestConfidence = confidence;
        bestLabel = label;
      }
    });

    const currentConfidence = currentArtwork ? detectionBuffer.get(currentArtwork) ?? 0 : 0;

    if (currentArtwork && currentConfidence < HIDE_CONFIDENCE) {
      clearArtwork();
    }

    if (!bestLabel) {
      return;
    }

    const candidateId = findArtworkByLabel(bestLabel);
    if (!candidateId) return;

    if (!currentArtwork) {
      if (bestConfidence >= DISPLAY_CONFIDENCE) {
        displayArtwork(candidateId);
      }
      return;
    }

    if (candidateId === currentArtwork) {
      return;
    }

    if (bestConfidence >= DISPLAY_CONFIDENCE && bestConfidence >= currentConfidence + SWITCH_MARGIN) {
      displayArtwork(candidateId);
    }
  }

  function updatePermissionMessage(state) {
    switch (state) {
      case 'granted':
        permissionMessage = '';
        break;
      case 'denied':
        permissionMessage = 'Camera access is blocked in your browser. Enable it in settings, then refresh to continue.';
        break;
      case 'prompt':
      case 'unknown':
        permissionMessage = '';
        break;
      case 'unsupported':
        permissionMessage = 'This browser cannot share the camera. Try opening the page in Safari or Chrome.';
        break;
      case 'error':
        permissionMessage = 'We could not open the camera. Make sure it isn\'t already in use, then try again.';
        break;
      default:
        permissionMessage = 'Camera access is required to continue.';
    }
  }

  async function checkCameraPermission() {
    if (!navigator?.mediaDevices?.getUserMedia) {
      permissionState = 'unsupported';
      updatePermissionMessage('unsupported');
      showPermissionPrompt = true;
      return;
    }

    if (navigator.permissions?.query) {
      try {
        permissionStatusHandle = await navigator.permissions.query({ name: 'camera' });
        permissionState = permissionStatusHandle.state;
        updatePermissionMessage(permissionState);
        if (permissionState === 'granted') {
          await requestCameraAccess(true);
        } else {
          showPermissionPrompt = true;
        }
        permissionStatusHandle.onchange = async () => {
          permissionState = permissionStatusHandle.state;
          updatePermissionMessage(permissionState);
          if (permissionStatusHandle.state === 'granted') {
            showPermissionPrompt = false;
            await requestCameraAccess(true);
          } else {
            showPermissionPrompt = true;
          }
        };
        return;
      } catch (error) {
        console.warn('Unable to query camera permissions', error);
      }
    }

    permissionState = 'prompt';
    updatePermissionMessage('prompt');
    showPermissionPrompt = true;
  }

  async function requestCameraAccess(auto = false) {
    if (isRequestingCamera && !auto) return;
    if (!navigator?.mediaDevices?.getUserMedia) {
      permissionState = 'unsupported';
      updatePermissionMessage('unsupported');
      showPermissionPrompt = true;
      return;
    }

    try {
      if (!auto) {
        isRequestingCamera = true;
      }
      showLoading = true;
      setStatus('Starting camera...');
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      permissionState = 'granted';
      updatePermissionMessage('granted');
      await handleCameraStream(mediaStream);
    } catch (error) {
      handleCameraError(error);
    } finally {
      if (!auto) {
        isRequestingCamera = false;
      }
      if (permissionState !== 'granted') {
        showLoading = false;
      }
    }
  }

  async function handleCameraStream(mediaStream) {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    stream = mediaStream;
    videoEl.srcObject = stream;

    await new Promise((resolve) => {
      const finalize = () => resolve();
      videoEl.onloadedmetadata = () => {
        const playPromise = videoEl.play();
        if (playPromise?.catch) {
          playPromise.catch((error) => {
            console.warn('Video playback error', error);
          }).finally(finalize);
        } else {
          finalize();
        }
      };
    });

    showPermissionPrompt = false;
    showLoading = false;
    lastCameraError = '';
    setStatus('Ready');
    cameraReady = true;
    splashReady = true;

    if (!detectionStarted) {
      detectionStarted = true;
      detectLoop();
    }
  }

  function handleCameraError(error) {
    console.error('Camera access error:', error);
    const name = error?.name;
    lastCameraError = error?.message ?? '';

    if (name === 'NotAllowedError' || name === 'SecurityError') {
      permissionState = 'denied';
    } else if (name === 'NotFoundError') {
      permissionState = 'error';
      lastCameraError = 'No camera device was found.';
    } else if (name === 'NotReadableError') {
      permissionState = 'error';
      lastCameraError = 'The camera is already in use by another application.';
    } else {
      permissionState = 'error';
    }

    updatePermissionMessage(permissionState);
    showPermissionPrompt = true;
    setStatus(lastCameraError || 'Camera access blocked.');
    splashReady = false;
  }

  async function processFrame() {
    if (!session || !videoEl || videoEl.readyState < 2) return [];

    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const size = 320;

    const scale = Math.min(size / vh, size / vw);
    const newWidth = Math.floor(vw * scale);
    const newHeight = Math.floor(vh * scale);
    const top = Math.floor((size - newHeight) / 2);
    const left = Math.floor((size - newWidth) / 2);

    const { context } = getFrameCanvas(size);
    context.fillStyle = 'black';
    context.fillRect(0, 0, size, size);
    context.drawImage(videoEl, 0, 0, vw, vh, left, top, newWidth, newHeight);

    const imageData = context.getImageData(0, 0, size, size);
    const chw = new Float32Array(3 * size * size);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const i = y * size + x;
        const j = i * 4;
        chw[0 * size * size + i] = imageData.data[j] / 255;
        chw[1 * size * size + i] = imageData.data[j + 1] / 255;
        chw[2 * size * size + i] = imageData.data[j + 2] / 255;
      }
    }

    const input = new window.ort.Tensor('float32', chw, [1, 3, size, size]);
    const output = await session.run({ images: input });
    return decodeOutputs(output, vw, vh, { left, top, scale });
  }

  function decodeOutputs(output, vw, vh, pad) {
    const names = Object.keys(output);
    const boxes = [];

    if (names.includes('boxes') && names.includes('scores') && names.includes('labels')) {
      const bxT = output.boxes;
      const scT = output.scores;
      const lbT = output.labels;
      const total = bxT.dims[1] ?? bxT.dims[0];

      for (let i = 0; i < total; i += 1) {
        const score = scT.data[i];
        const label = lbT.data[i];
        if (score >= CONFIDENCE_THRESHOLD && label > 0) {
          const offset = i * 4;
          let [x1, y1, x2, y2] = [
            bxT.data[offset],
            bxT.data[offset + 1],
            bxT.data[offset + 2],
            bxT.data[offset + 3],
          ];

          x1 = (x1 - pad.left) / pad.scale;
          y1 = (y1 - pad.top) / pad.scale;
          x2 = (x2 - pad.left) / pad.scale;
          y2 = (y2 - pad.top) / pad.scale;

          x1 = Math.max(0, Math.min(vw, x1));
          y1 = Math.max(0, Math.min(vh, y1));
          x2 = Math.max(0, Math.min(vw, x2));
          y2 = Math.max(0, Math.min(vh, y2));

          if (x2 > x1 && y2 > y1) {
            boxes.push({ x1, y1, x2, y2, score, label });
          }
        }
      }
    }

    return boxes;
  }

  async function detectLoop() {
    try {
      const boxes = await processFrame();
      updateArtworkOverlay(boxes);
    } catch (error) {
      console.error('Detection error:', error);
    }
    animationFrameId = requestAnimationFrame(detectLoop);
  }

  async function init() {
    let artworkLoaded = false;
    let modelLoaded = false;

    setStatus('Loading detector assets...');

    try {
      const [artworkResult, modelResult] = await Promise.allSettled([
        loadArtworkContent(),
        loadModel(),
      ]);

      if (artworkResult.status === 'rejected') {
        console.error('Artwork content failed to load:', artworkResult.reason);
      } else {
        artworkLoaded = true;
      }

      if (modelResult.status === 'rejected') {
        console.error('Model failed to load:', modelResult.reason);
        setStatus('Detector model failed to load. Refresh and try again.');
      } else {
        modelLoaded = true;
      }

      showLoading = false;
      setStatus('Checking camera access...');

      await checkCameraPermission();
      if (permissionState === 'granted') {
        await requestCameraAccess(true);
        setStatus('Enjoy!');
        splashReady = true;
      } else {
        showPermissionPrompt = true;
        splashReady = false;
      }

      if (!artworkLoaded || !modelLoaded) {
        console.warn('Application initialized with missing resources.', {
          artworkLoaded,
          modelLoaded,
        });
      }
    } catch (error) {
      console.error('Initialization failed unexpectedly:', error);
      setStatus('Something went wrong during startup. Refresh and try again.');
      showLoading = false;
      await checkCameraPermission();
      if (permissionState !== 'granted') {
        showPermissionPrompt = true;
      }
      splashReady = permissionState === 'granted';
    }
  }

  async function beginExperience() {
    if (hasStarted) return;
    hasStarted = true;
    splashReady = false;
    splashStatus = 'Preparing experience...';
    cameraReady = false;
    pickNewPhrase();
    await init();
  }

  async function handleSplashDone() {
    showSplash = false;
    splashSubtitle = DEFAULT_SPLASH_SUBTITLE;
    splashStatus = '';
    if (!hasStarted && !IS_TEST_MODE) {
      await beginExperience();
    }
  }

  onMount(async () => {
    await tick();

    if (IS_TEST_MODE) {
      showSplash = false;
      showLoading = false;
      showPermissionPrompt = false;
      overlayVisible = false;
      displayedArtwork = { ...TEST_ARTWORK };
      currentArtwork = TEST_ARTWORK.id;
      artworkVisible = true;
      triggerConfettiFor(TEST_ARTWORK.id);
      splashReady = true;
      splashSubtitle = DEFAULT_SPLASH_SUBTITLE;
      splashStatus = 'Enjoy!';
      cameraReady = true;
      return;
    }

    beginExperience();
  });

  onDestroy(() => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
    detectionBuffer.clear();
    if (permissionStatusHandle) {
      permissionStatusHandle.onchange = null;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (phraseTimeout) {
      clearTimeout(phraseTimeout);
    }
    if (countsRefreshTimer) {
      clearInterval(countsRefreshTimer);
    }
    if (confettiTimer) {
      clearTimeout(confettiTimer);
    }
  });
</script>

<svelte:window on:keydown={(event) => {
  if (event.key === 'Escape') {
    showInstructionOverlay();
  }
}} />

<div class="app">
  <div class="video-container">
    <div class={`camera-placeholder ${cameraReady ? 'camera-placeholder--hidden' : ''}`}></div>
    <video bind:this={videoEl} autoplay muted playsinline></video>
  </div>

  <LoadingOverlay visible={showLoading && !showSplash} message={loadingMessage} />

  <InstructionOverlay
    visible={overlayVisible}
    phrase={phraseText}
    shimmer={shouldShimmer}
    phraseOpacity={phraseOpacity}
  />

  <ReactionConfetti active={confettiVisible} counts={confettiCounts} seed={confettiSeed} />

  <ArtworkOverlay
    visible={artworkVisible}
    artwork={displayedArtwork}
    on:reacted={handleReaction}
  />

  <ToastHost placement="bottom-center" />

  {#if showSplash}
    <SplashScreen
      status={splashStatus}
      ready={splashReady}
      minDurationMs={2500}
      fadeMs={350}
      awaitingPermission={showPermissionPrompt}
      permissionMessage={lastCameraError || permissionMessage}
      permissionCopy="We need camera access for this to work."
      permissionDisabled={!canRequestPermission || isRequestingCamera}
      on:cameraRequest={() => requestCameraAccess(false)}
      on:done={handleSplashDone}
    />
  {/if}
</div>

<style>
  :global(:root) {
    --ui-fg: #ffffff;
    --ui-fg-dim: #ffffffcc;
    --ui-shadow: 0 10px 40px rgba(0, 0, 0, 0.22);
    --bracket-thickness: 2.4;
  }

  :global(html),
  :global(body) {
    height: 100%;
    margin: 0;
    background: #111;
    color: var(--ui-fg);
    font: 16px/1.2 ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto,
      'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji';
  }

  .app {
    position: relative;
    inline-size: 100%;
    block-size: 100vh;
    overflow: hidden;
    touch-action: manipulation;
  }

  .video-container {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }

  .camera-placeholder {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 30%, rgba(99, 102, 241, 0.18), transparent 55%),
      radial-gradient(circle at 70% 70%, rgba(16, 185, 129, 0.16), transparent 52%),
      #0b0f18;
    transition: opacity 0.35s ease;
    pointer-events: none;
    opacity: 1;
  }

  .camera-placeholder.camera-placeholder--hidden {
    opacity: 0;
  }

  video {
    position: absolute;
    inset: 0;
    inline-size: 100%;
    block-size: 100%;
    object-fit: cover;
    transform: translateZ(0);
    background: #000;
  }
</style>
