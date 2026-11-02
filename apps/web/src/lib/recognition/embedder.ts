import type { RecognizerConfig } from '#lib/domain/recognition.js';
import { expandBox } from './boxes.js';
import { FrameBuffer, isReady, sourceSize } from './frames.js';
import { createSession, type OrtRuntime } from './ort.js';
import { rgbaToChwNormalized } from './preprocess.js';
import type { Box, PixelSource } from './types.js';
import { concat, l2Normalize } from './vectors.js';

export interface Embedder {
	readonly inputSize: number;
	readonly dim: number;
	/** Embeds the whole source, or `region` (grown by the configured margin), as a unit vector. */
	embed(source: PixelSource, region?: Box): Promise<Float32Array>;
	dispose(): Promise<void>;
}

function meanColour(mean: readonly [number, number, number]): string {
	const [r, g, b] = mean.map((v) => Math.round(v * 255));
	return `rgb(${r} ${g} ${b})`;
}

function pool(
	cls: Float32Array,
	patches: Float32Array,
	channels: number,
	pooling: RecognizerConfig['pooling']
): Float32Array {
	const clsVector = cls.slice(0, channels);
	if (pooling === 'cls') return l2Normalize(clsVector);
	const tokens = patches.length / channels;
	const mean = new Float32Array(channels);
	for (let t = 0; t < tokens; t += 1) {
		const offset = t * channels;
		for (let c = 0; c < channels; c += 1) mean[c] = mean[c]! + patches[offset + c]!;
	}
	for (let c = 0; c < channels; c += 1) mean[c] = mean[c]! / Math.max(1, tokens);
	if (pooling === 'mean') return l2Normalize(mean);
	return l2Normalize(concat(clsVector, mean));
}

/**
 * Wraps the DINO embedder (input `image [1,3,S,S]` standardised RGB; outputs `cls [1,C]` and
 * `patches [1,N,C]`). Letterbox padding uses the dataset mean so it standardises to zero.
 */
export async function createEmbedder(
	runtime: OrtRuntime,
	bytes: Uint8Array,
	inputSize: number,
	dim: number,
	config: RecognizerConfig
): Promise<Embedder> {
	const session = await createSession(runtime, bytes);
	const inputName = session.inputNames[0] ?? 'image';
	const frame = new FrameBuffer(inputSize, meanColour(config.mean));
	const chw = new Float32Array(3 * inputSize * inputSize);
	const { ort } = runtime;

	return {
		inputSize,
		dim,
		async embed(source, region) {
			if (!isReady(source)) throw new Error('Source has no pixels yet.');
			const { width, height } = sourceSize(source);
			const box = region ? expandBox(region, config.cropMargin, width, height) : undefined;
			const { rgba } = frame.draw(source, box);
			rgbaToChwNormalized(rgba, inputSize, inputSize, config.mean, config.std, chw);
			const input = new ort.Tensor('float32', chw, [1, 3, inputSize, inputSize]);
			const output = await session.run({ [inputName]: input });
			const cls = output['cls'];
			const patches = output['patches'];
			if (!cls || !patches) throw new Error('Embedder outputs must be named cls and patches.');
			const channels = cls.dims.at(-1) ?? 0;
			const vector = pool(
				cls.data as Float32Array,
				patches.data as Float32Array,
				channels,
				config.pooling
			);
			if (vector.length !== dim)
				throw new Error(`Embedder produced ${vector.length} values, expected ${dim}.`);
			return vector;
		},
		async dispose() {
			await session.release();
		}
	};
}
