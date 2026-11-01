import { computeLetterbox } from './letterbox.js';
import { decodeDetections } from './decode.js';
import { rgbaToChwFloat } from './preprocess.js';
import type { Detection, DetectorManifest, Letterbox } from './types.js';
import type * as Ort from 'onnxruntime-web';

/**
 * The runtime is served from static/ort (see scripts/copy-ort-assets.mjs) and imported at
 * runtime so Vite never bundles its multi-megabyte WebAssembly into the Worker's assets.
 */
const ORT_BASE = '/ort/';
let ortModule: Promise<typeof Ort> | undefined;

function loadOrt(): Promise<typeof Ort> {
	ortModule ??= import(/* @vite-ignore */ `${ORT_BASE}ort.wasm.min.mjs`).then(
		(module: typeof Ort) => {
			module.env.wasm.wasmPaths = ORT_BASE;
			return module;
		}
	);
	return ortModule;
}

export interface Detector {
	readonly manifest: DetectorManifest;
	/** Runs one inference on the current video frame. Returns detections in video pixels. */
	detect(video: HTMLVideoElement, threshold?: number): Promise<Detection[]>;
	dispose(): Promise<void>;
}

export const DEFAULT_THRESHOLD = 0.7;

/**
 * Loads onnxruntime-web lazily (it is ~1 MB of JS plus WebAssembly) and wraps a session
 * with the frame plumbing: letterbox → CHW tensor → run → decode.
 */
export async function loadDetector(
	manifest: DetectorManifest,
	onProgress?: (stage: string) => void
): Promise<Detector> {
	onProgress?.('Loading runtime');
	const ort = await loadOrt();

	onProgress?.('Downloading model');
	const response = await fetch(manifest.url, { cache: 'force-cache' });
	if (!response.ok) throw new Error(`Could not download the model (${response.status}).`);
	const bytes = new Uint8Array(await response.arrayBuffer());

	onProgress?.('Preparing model');
	const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
	const inputName = session.inputNames[0] ?? 'images';

	const size = manifest.inputSize;
	const canvas = document.createElement('canvas');
	canvas.width = size;
	canvas.height = size;
	const context = canvas.getContext('2d', { willReadFrequently: true });
	if (!context) throw new Error('2D canvas is not available.');
	const chw = new Float32Array(3 * size * size);
	let letterbox: Letterbox | null = null;
	let frameWidth = 0;
	let frameHeight = 0;

	return {
		manifest,
		async detect(video, threshold = DEFAULT_THRESHOLD) {
			if (video.readyState < 2 || video.videoWidth === 0) return [];
			if (!letterbox || frameWidth !== video.videoWidth || frameHeight !== video.videoHeight) {
				frameWidth = video.videoWidth;
				frameHeight = video.videoHeight;
				letterbox = computeLetterbox(frameWidth, frameHeight, size);
			}
			context.fillStyle = '#000';
			context.fillRect(0, 0, size, size);
			context.drawImage(
				video,
				0,
				0,
				frameWidth,
				frameHeight,
				letterbox.left,
				letterbox.top,
				letterbox.width,
				letterbox.height
			);
			const { data } = context.getImageData(0, 0, size, size);
			rgbaToChwFloat(data, size, size, chw);

			const input = new ort.Tensor('float32', chw, [1, 3, size, size]);
			const output = await session.run({ [inputName]: input });
			const boxes = output['boxes'];
			const scores = output['scores'];
			const labels = output['labels'];
			if (!boxes || !scores || !labels)
				throw new Error('Model outputs must be named boxes, scores and labels.');
			return decodeDetections(
				{
					boxes: { data: boxes.data as Float32Array, dims: boxes.dims },
					scores: { data: scores.data as Float32Array, dims: scores.dims },
					labels: { data: labels.data as BigInt64Array | Int32Array, dims: labels.dims }
				},
				{ threshold, letterbox, frameWidth, frameHeight, labels: manifest.labels }
			);
		},
		async dispose() {
			await session.release();
		}
	};
}
