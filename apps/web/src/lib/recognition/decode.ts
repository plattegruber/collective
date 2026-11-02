import { unletterboxBox } from './letterbox.js';
import type { DetectedRegion, Letterbox } from './types.js';

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
	/** Drop boxes whose shorter side is below this many source pixels. */
	minEdge?: number;
}

/**
 * Turns raw `boxes`/`scores` outputs into regions in frame coordinates, best first.
 * The detector is class-agnostic, so `labels` is ignored beyond skipping background (0).
 */
export function decodeRegions(
	output: { boxes: TensorLike; scores: TensorLike; labels?: TensorLike },
	options: DecodeOptions
): DetectedRegion[] {
	const count = output.scores.dims.at(-1) ?? output.scores.data.length;
	const minEdge = options.minEdge ?? 8;
	const regions: DetectedRegion[] = [];
	for (let i = 0; i < count; i += 1) {
		const score = Number(output.scores.data[i]);
		if (!(score >= options.threshold)) continue;
		if (output.labels && Number(output.labels.data[i]) <= 0) continue;
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
		if (box.x2 - box.x1 < minEdge || box.y2 - box.y1 < minEdge) continue;
		regions.push({ score, box });
	}
	return regions.sort((a, b) => b.score - a.score);
}
