<script>
  import { onMount, onDestroy, tick } from 'svelte';

  const BASE_URL = import.meta.env.BASE_URL ?? '/';
  const CONFIDENCE_THRESHOLD = 0.7;
  const DISPLAY_CONFIDENCE = 0.78;
  const HIDE_CONFIDENCE = 0.6;
  const SMOOTHING_FACTOR = 0.55;
  const DECAY_FACTOR = 0.6;
  const SWITCH_MARGIN = 0.1;
  const MIN_BUFFER_CONFIDENCE = 0.05;

  const phrases = ['Point me toward the art.'];

  let artContent = {};
  let labels = {};
  let currentArtwork = null;
  let session = null;
  let isOnnxLoaded = false;
  const detectionBuffer = new Map();

  let videoEl;
  let canvasEl;
  let ctx;
  let stream;
  let animationFrameId;
  let phraseTimeout;

  let overlayVisible = true;
  let shouldShimmer = false;
  let phraseOpacity = 1;
  let phraseText = '';

  let showLoading = true;
  let loadingMessage = 'Loading detector model...';

  let artworkVisible = false;
  let displayedArtwork = {
    title: '',
    byline: '',
    materials: '',
    description: '',
  };

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

  function handlePhraseKey(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pickNewPhrase();
    }
  }

  function hideOverlay() {
    overlayVisible = false;
  }

  function showInstructionOverlay() {
    overlayVisible = true;
  }

  async function loadArtworkContent() {
    const response = await fetch(`${BASE_URL}data/art-content.v1.json`);
    artContent = await response.json();
    console.log('Loaded artwork content:', Object.keys(artContent).length, 'pieces');
  }

  async function loadModel() {
    const manifestResponse = await fetch(`${BASE_URL}model-manifest.json`);
    const manifest = await manifestResponse.json();
    console.log('Model manifest:', manifest);

    const labelsPath = manifest.labels
      ? `${BASE_URL}${manifest.labels}`
      : `${BASE_URL}models/detector/labels.json`;
    const labelsResponse = await fetch(labelsPath);
    labels = await labelsResponse.json();
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

  function drawBoxes(detections) {
    if (!ctx || !videoEl || !canvasEl) return;
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    ctx.setLineDash([8, 6]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#00e0ff';
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.font = '16px system-ui';

    detections.forEach((box) => {
      const x = box.x1;
      const y = box.y1;
      const w = box.x2 - box.x1;
      const h = box.y2 - box.y1;
      ctx.strokeRect(x, y, w, h);

      const labelText = labels[box.label] ?? `class_${box.label}`;
      const confidence = `${(box.score * 100).toFixed(0)}%`;
      const text = `${labelText} ${confidence}`;

      const metrics = ctx.measureText(text);
      ctx.fillRect(x, y - 25, metrics.width + 10, 22);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, x + 5, y - 8);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
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
      title: artwork.title ?? '',
      byline: bylineParts.join(', '),
      materials: artwork.materials ?? '',
      description: artwork.description ?? '',
    };

    currentArtwork = artworkId;
    artworkVisible = true;
    hideOverlay();
  }

  function clearArtwork() {
    if (!currentArtwork) return;
    artworkVisible = false;
    currentArtwork = null;
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
      drawBoxes(boxes);
      updateArtworkOverlay(boxes);
    } catch (error) {
      console.error('Detection error:', error);
    }
    animationFrameId = requestAnimationFrame(detectLoop);
  }

  async function init() {
    try {
      await Promise.all([loadArtworkContent(), loadModel()]);

      loadingMessage = 'Starting camera...';
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      videoEl.srcObject = stream;

      await new Promise((resolve) => {
        videoEl.onloadedmetadata = () => {
          videoEl.play().catch(() => {});
          resolve();
        };
      });

      showLoading = false;
      detectLoop();
    } catch (error) {
      console.error('Initialization failed:', error);
      loadingMessage = 'Failed to initialize. Please check camera permissions and refresh.';
      showLoading = true;
    }
  }

  onMount(async () => {
    await tick();
    if (canvasEl) {
      ctx = canvasEl.getContext('2d');
    }
    pickNewPhrase();
    await init();
  });

  onDestroy(() => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
    detectionBuffer.clear();
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    if (phraseTimeout) {
      clearTimeout(phraseTimeout);
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
    <video bind:this={videoEl} autoplay muted playsinline></video>
    <canvas bind:this={canvasEl}></canvas>
  </div>

  {#if showLoading}
    <div id="loading" class="loading">{loadingMessage}</div>
  {/if}

  <div
    class={`overlay ${overlayVisible ? 'is-visible' : 'is-hidden'}`}
    aria-hidden={!overlayVisible}
    on:click={hideOverlay}
  >
    <div class="hud">
      <svg class="brackets pulse" viewBox="0 0 100 64" role="img" aria-label="framing guides">
        <path d="M8 18 L8 8 L28 8" />
        <path d="M92 18 L92 8 L72 8" />
        <path d="M8 46 L8 56 L28 56" />
        <path d="M92 46 L92 56 L72 56" />
      </svg>
      <div
        class={`phrase ${shouldShimmer ? 'shimmer' : ''}`}
        tabindex="0"
        aria-live="polite"
        style={`opacity: ${phraseOpacity};`}
        on:click={pickNewPhrase}
        on:keydown={handlePhraseKey}
      >
        {phraseText}
      </div>
    </div>
  </div>

  <div class={`artwork-overlay ${artworkVisible ? 'visible' : ''}`}>
    <h2>{displayedArtwork.title}</h2>
    <p>{displayedArtwork.byline}</p>
    <p>{displayedArtwork.materials}</p>
    <p>{displayedArtwork.description}</p>
  </div>
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

  video {
    position: absolute;
    inset: 0;
    inline-size: 100%;
    block-size: 100%;
    object-fit: cover;
    transform: translateZ(0);
    background: #000;
  }

  canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 1;
    pointer-events: none;
  }

  .loading {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    color: #fff;
    font-family: system-ui, sans-serif;
    z-index: 50;
    background: rgba(0, 0, 0, 0.6);
  }

  .overlay {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: clamp(12px, 4vmin, 28px);
    pointer-events: none;
  }

  .overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(60% 60% at 50% 45%, rgba(0, 0, 0, 0) 0%, rgba(0, 0, 0, 0.35) 100%);
    pointer-events: none;
  }

  .hud {
    position: relative;
    display: grid;
    place-items: center;
    filter: drop-shadow(var(--ui-shadow));
  }

  .brackets {
    inline-size: min(70vmin, 560px);
    block-size: auto;
    opacity: 0.95;
  }

  .brackets line,
  .brackets path {
    stroke: var(--ui-fg);
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: var(--bracket-thickness);
    vector-effect: non-scaling-stroke;
  }

  @media (prefers-reduced-motion: no-preference) {
    .pulse {
      animation: pulse 2.4s ease-in-out infinite;
    }

    @keyframes pulse {
      0%,
      100% {
        opacity: 0.85;
      }

      50% {
        opacity: 1;
      }
    }
  }

  .phrase {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    text-align: center;
    letter-spacing: 0.01em;
    font-weight: 600;
    font-size: clamp(18px, 3.6vmin, 28px);
    color: var(--ui-fg-dim);
    line-height: 1.25;
    max-inline-size: min(80vw, 26ch);
    padding: 0 clamp(8px, 2vw, 16px);
    transition: opacity 0.2s ease;
  }

  .phrase.shimmer {
    background: linear-gradient(
      90deg,
      rgba(255, 255, 255, 0.5) 0%,
      rgba(255, 255, 255, 1) 20%,
      rgba(255, 255, 255, 0.6) 40%,
      rgba(255, 255, 255, 0.85) 60%,
      rgba(255, 255, 255, 0.6) 80%,
      rgba(255, 255, 255, 0.5) 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    filter: drop-shadow(var(--ui-shadow));
    background-size: 200% auto;
    animation: sheen 3.5s ease-in-out infinite;
  }

  @media (prefers-reduced-motion: reduce) {
    .phrase.shimmer {
      animation: none;
      background-size: auto;
      color: var(--ui-fg-dim);
    }
  }

  @keyframes sheen {
    0% {
      background-position: 200% 0;
    }

    100% {
      background-position: -200% 0;
    }
  }

  .is-hidden {
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.28s ease, visibility 0.28s step-end;
  }

  .is-visible {
    opacity: 1;
    visibility: visible;
    transition: opacity 0.28s ease;
  }

  .artwork-overlay {
    position: fixed;
    top: 20px;
    bottom: 20px;
    left: 20px;
    right: 20px;
    color: #f5f7ff;
    padding: clamp(20px, 4vw, 28px);
    border-radius: 18px;
    z-index: 10;
    pointer-events: none;
    opacity: 0;
    transform: translateY(20px);
    transition: opacity 0.35s ease, transform 0.35s ease;
    background: rgba(15, 17, 26, 0.48);
    box-shadow: 0 28px 60px rgba(0, 0, 0, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(16px) saturate(125%);
    -webkit-backdrop-filter: blur(16px) saturate(125%);
    overflow: hidden;
  }

  @supports not (backdrop-filter: blur(1px)) {
    .artwork-overlay {
      background: rgba(12, 14, 20, 0.72);
    }
  }

  .artwork-overlay.visible {
    opacity: 1;
    transform: translateY(0);
  }

  .artwork-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.16) 0%, rgba(255, 255, 255, 0) 40%);
    opacity: 0.9;
    pointer-events: none;
  }

  .artwork-overlay h2 {
    position: relative;
    font-size: clamp(1.5rem, 4vw, 2.2rem);
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.02em;
  }

  .artwork-overlay p {
    position: relative;
    margin: 6px 0;
    line-height: 1.65;
    color: rgba(240, 244, 255, 0.85);
  }

  .artwork-overlay p:last-child {
    margin-top: 16px;
    font-size: 0.95rem;
    color: rgba(224, 230, 255, 0.75);
  }
</style>
