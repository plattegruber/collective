import { describe, expect, it } from 'vitest';
import { decodeDetections } from './decode.js';
import { computeLetterbox } from './letterbox.js';

const letterbox = computeLetterbox(640, 480, 320);
const labels = { '1': 'art_a', '2': 'art_b' };

function tensors(rows: Array<{ box: number[]; score: number; label: number }>) {
	return {
		boxes: { data: rows.flatMap((row) => row.box), dims: [1, rows.length, 4] },
		scores: { data: rows.map((row) => row.score), dims: [1, rows.length] },
		labels: { data: rows.map((row) => BigInt(row.label)), dims: [1, rows.length] }
	};
}

describe('decodeDetections', () => {
	it('maps class indices to artwork ids and sorts by score', () => {
		const output = tensors([
			{ box: [10, 50, 110, 150], score: 0.6, label: 2 },
			{ box: [10, 50, 110, 150], score: 0.9, label: 1 }
		]);
		const result = decodeDetections(output, {
			threshold: 0.5,
			letterbox,
			frameWidth: 640,
			frameHeight: 480,
			labels
		});
		expect(result.map((d) => d.artworkId)).toEqual(['art_a', 'art_b']);
		expect(result[0]?.box).toEqual({ x1: 20, y1: 20, x2: 220, y2: 220 });
	});

	it('drops low scores, background, unknown classes and empty boxes', () => {
		const output = tensors([
			{ box: [10, 50, 110, 150], score: 0.2, label: 1 },
			{ box: [10, 50, 110, 150], score: 0.9, label: 0 },
			{ box: [10, 50, 110, 150], score: 0.9, label: 7 },
			{ box: [10, 50, 10, 150], score: 0.9, label: 1 }
		]);
		const result = decodeDetections(output, {
			threshold: 0.5,
			letterbox,
			frameWidth: 640,
			frameHeight: 480,
			labels
		});
		expect(result).toEqual([]);
	});

	it('handles NaN scores without throwing', () => {
		const output = tensors([{ box: [0, 0, 10, 10], score: Number.NaN, label: 1 }]);
		expect(
			decodeDetections(output, {
				threshold: 0.5,
				letterbox,
				frameWidth: 640,
				frameHeight: 480,
				labels
			})
		).toEqual([]);
	});
});
