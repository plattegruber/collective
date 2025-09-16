import './style.css';

// Configuration
const BASE_URL = import.meta.env.BASE_URL ?? '/';
const CONFIDENCE_THRESHOLD = 0.7;
const DISPLAY_CONFIDENCE = 0.78;
const HIDE_CONFIDENCE = 0.6;
const SMOOTHING_FACTOR = 0.55;
const DECAY_FACTOR = 0.6;
const SWITCH_MARGIN = 0.1;
const MIN_BUFFER_CONFIDENCE = 0.05;

// Mutable state
let artContent = {};
let labels = {};
let currentArtwork = null;
let session = null;
let isOnnxLoaded = false;
const detectionBuffer = new Map();

// DOM references
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const loading = document.getElementById('loading');
const artworkOverlay = document.getElementById('artwork-info');
const overlayEl = document.getElementById('overlay');
const phraseEl = document.getElementById('phrase');

// Phrase bank for dynamic overlay
const phrases = [
  'Point me toward the art.'
];

function pickNewPhrase() {
  const current = phraseEl.textContent;
  const options = current ? phrases.filter((phrase) => phrase !== current) : phrases;
  const nextPhrase = options[Math.floor(Math.random() * options.length)];
  phraseEl.classList.remove('shimmer');
  phraseEl.style.opacity = 0;
  window.setTimeout(() => {
    phraseEl.textContent = nextPhrase;
    phraseEl.style.opacity = 1;
    phraseEl.classList.add('shimmer');
  }, 140);
}

function hideOverlay() {
  overlayEl.classList.remove('is-visible');
  overlayEl.classList.add('is-hidden');
}

function showOverlay() {
  overlayEl.classList.remove('is-hidden');
  overlayEl.classList.add('is-visible');
}

function initOverlay() {
  pickNewPhrase();
  phraseEl.addEventListener('click', pickNewPhrase);
  phraseEl.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      pickNewPhrase();
    }
  });
  overlayEl.addEventListener('click', () => hideOverlay());
}

async function loadArtworkContent() {
  try {
    const response = await fetch(`${BASE_URL}data/art-content.v1.json`);
    artContent = await response.json();
    console.log('Loaded artwork content:', Object.keys(artContent).length, 'pieces');
  } catch (error) {
    console.error('Failed to load artwork content:', error);
  }
}

async function loadModel() {
  try {
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
    loading.textContent = 'Starting camera...';
  } catch (error) {
    console.error('Failed to load model:', error);
    loading.innerHTML = '<div class="error">Failed to load detector model. Please refresh.</div>';
    throw error;
  }
}

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

async function processFrame() {
  if (!session || video.readyState < 2) return [];

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const size = 320;

  const scale = Math.min(size / vh, size / vw);
  const newWidth = Math.floor(vw * scale);
  const newHeight = Math.floor(vh * scale);
  const top = Math.floor((size - newHeight) / 2);
  const left = Math.floor((size - newWidth) / 2);

  const { context } = getFrameCanvas(size);
  context.fillStyle = 'black';
  context.fillRect(0, 0, size, size);
  context.drawImage(video, 0, 0, vw, vh, left, top, newWidth, newHeight);

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

function drawBoxes(detections) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

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

function displayArtwork(artworkId) {
  const artwork = artContent[artworkId];
  if (!artwork) return;

  document.getElementById('artwork-title').textContent = artwork.title;
  document.getElementById('artwork-artist').textContent = `${artwork.artist}, ${artwork.year}`;
  document.getElementById('artwork-materials').textContent = artwork.materials;
  document.getElementById('artwork-description').textContent = artwork.description;

  artworkOverlay.classList.add('visible');
  currentArtwork = artworkId;
  hideOverlay();
}

function clearArtwork() {
  if (!currentArtwork) return;
  artworkOverlay.classList.remove('visible');
  currentArtwork = null;
  showOverlay();
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

  if (bestConfidence >= DISPLAY_CONFIDENCE && bestConfidence >= (currentConfidence + SWITCH_MARGIN)) {
    displayArtwork(candidateId);
  }
}

async function detectLoop() {
  try {
    const boxes = await processFrame();
    drawBoxes(boxes);
    updateArtworkOverlay(boxes);
  } catch (error) {
    console.error('Detection error:', error);
  }
  requestAnimationFrame(detectLoop);
}

async function init() {
  try {
    await Promise.all([
      loadArtworkContent(),
      loadModel(),
    ]);

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment' },
    });
    video.srcObject = stream;

    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
    });

    loading.style.display = 'none';
    initOverlay();
    detectLoop();
  } catch (error) {
    console.error('Initialization failed:', error);
    loading.innerHTML = '<div class="error">Failed to initialize. Please check camera permissions and refresh.</div>';
  }
}

init();
