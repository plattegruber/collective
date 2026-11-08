import { decodeRegions } from './decode.js';
import { FrameBuffer, isReady, sourceSize } from './frames.js';
import { createSession, type OrtRuntime } from './ort.js';
import { rgbaToChwAutoLevels } from './preprocess.js';
import type { DetectedRegion, PixelSource } from './types.js';

export interface Detector {
	readonly inputSize: number;
	/** Finds artworks in the source. Regions are in source pixels, best first. */
	detect(source: PixelSource, threshold: number): Promise<DetectedRegion[]>;
	dispose(): Promise<void>;
}

/**
 * Wraps the class-agnostic artwork detector (SSDLite, input `images [1,3,S,S]` RGB 0..1 on a
 * black letterbox; outputs `boxes`, `scores`, `labels`). The drawn pixels are white-balanced
 * and contrast-stretched first (`rgbaToChwAutoLevels`): the model was trained on daylight
 * composites and its confidence collapses in dim, warm rooms without it.
 */
export async function createDetector(
	runtime: OrtRuntime,
	bytes: Uint8Array,
	inputSize: number
): Promise<Detector> {
	const session = await createSession(runtime, bytes);
	const inputName = session.inputNames[0] ?? 'images';
	const frame = new FrameBuffer(inputSize, '#000');
	const shared = new Float32Array(3 * inputSize * inputSize);
	const { ort } = runtime;

	return {
		inputSize,
		async detect(source, threshold) {
			if (!isReady(source)) return [];
			const { width, height } = sourceSize(source);
			const { letterbox, rgba } = frame.draw(source);
			// A proxied session takes ownership of the input buffer, so it cannot be reused.
			const chw = runtime.transfersInputs ? new Float32Array(shared.length) : shared;
			rgbaToChwAutoLevels(rgba, inputSize, inputSize, letterbox, chw);
			const input = new ort.Tensor('float32', chw, [1, 3, inputSize, inputSize]);
			const output = await session.run({ [inputName]: input });
			const boxes = output['boxes'];
			const scores = output['scores'];
			if (!boxes || !scores) throw new Error('Detector outputs must be named boxes and scores.');
			return decodeRegions(
				{
					boxes: { data: boxes.data as Float32Array, dims: boxes.dims },
					scores: { data: scores.data as Float32Array, dims: scores.dims },
					labels: output['labels']
						? { data: output['labels'].data as BigInt64Array, dims: output['labels'].dims }
						: undefined
				},
				{ threshold, letterbox, frameWidth: width, frameHeight: height }
			);
		},
		async dispose() {
			await session.release();
		}
	};
}
