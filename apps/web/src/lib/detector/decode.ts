import { unletterboxBox } from './letterbox.js';
import type { Detection, Letterbox } from './types.js';

/** The subset of an ORT tensor we read; keeps this module free of the runtime import. */
export interface TensorLike {
	data: ArrayLike<number> | ArrayLike<bigint>;
	dims: readonly number[];
}

export interface DecodeOptions {
	threshold: number;
	letterbox: Letterbox;
	frameWidth: number;
	frameHeight: number;
	labels: Record<string, string>;
}

/**
 * Turns raw `boxes`/`scores`/`labels` outputs into artwork detections in frame coordinates.
 * Unknown classes and degenerate boxes are dropped.
 */
export function decodeDetections(
	output: { boxes: TensorLike; scores: TensorLike; labels: TensorLike },
	options: DecodeOptions
): Detection[] {
	const count = output.scores.dims.at(-1) ?? output.scores.data.length;
	const detections: Detection[] = [];
	for (let i = 0; i < count; i += 1) {
		const score = Number(output.scores.data[i]);
		if (!(score >= options.threshold)) continue;
		const classIndex = Number(output.labels.data[i]);
		if (classIndex <= 0) continue;
		const artworkId = options.labels[String(classIndex)];
		if (!artworkId) continue;
		const offset = i * 4;
		const box = unletterboxBox(
			{
				x1: Number(output.boxes.data[offset]),
				y1: Number(output.boxes.data[offset + 1]),
				x2: Number(output.boxes.data[offset + 2]),
				y2: Number(output.boxes.data[offset + 3])
			},
			options.letterbox,
			options.frameWidth,
			options.frameHeight
		);
		if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;
		detections.push({ artworkId, score, box });
	}
	return detections.sort((a, b) => b.score - a.score);
}
