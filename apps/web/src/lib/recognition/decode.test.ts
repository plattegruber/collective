import { describe, expect, it } from 'vitest';
import { decodeRegions } from './decode.js';
import { computeLetterbox } from './letterbox.js';

describe('decodeRegions', () => {
	const letterbox = computeLetterbox(640, 480, 320);
	const options = { threshold: 0.5, letterbox, frameWidth: 640, frameHeight: 480 };

	it('keeps confident regions, maps them to frame pixels, and sorts by score', () => {
		const regions = decodeRegions(
			{
				boxes: { data: [10, 50, 100, 150, 0, 40, 320, 280, 5, 5, 6, 6], dims: [1, 3, 4] },
				scores: { data: [0.6, 0.9, 0.95], dims: [1, 3] },
				labels: { data: [1n, 1n, 1n], dims: [1, 3] }
			},
			options
		);
		expect(regions.map((r) => r.score)).toEqual([0.9, 0.6]);
		expect(regions[0]?.box).toEqual({ x1: 0, y1: 0, x2: 640, y2: 480 });
		expect(regions[1]?.box.x1).toBeCloseTo(20);
	});

	it('drops background labels and scores under the threshold', () => {
		const regions = decodeRegions(
			{
				boxes: { data: [10, 50, 100, 150, 10, 50, 100, 150], dims: [1, 2, 4] },
				scores: { data: [0.9, 0.2], dims: [1, 2] },
				labels: { data: [0n, 1n], dims: [1, 2] }
			},
			options
		);
		expect(regions).toEqual([]);
	});
});
