import { describe, expect, it } from 'vitest';
import { computeLetterbox, unletterboxBox } from './letterbox.js';

describe('computeLetterbox', () => {
	it('centres a landscape frame vertically', () => {
		const box = computeLetterbox(640, 480, 320);
		expect(box.scale).toBeCloseTo(0.5);
		expect(box.width).toBe(320);
		expect(box.height).toBe(240);
		expect(box.left).toBe(0);
		expect(box.top).toBe(40);
	});

	it('centres a portrait frame horizontally', () => {
		const box = computeLetterbox(480, 640, 320);
		expect(box.width).toBe(240);
		expect(box.height).toBe(320);
		expect(box.left).toBe(40);
		expect(box.top).toBe(0);
	});

	it('never upscales past the input size', () => {
		const box = computeLetterbox(100, 50, 320);
		expect(box.width).toBe(320);
		expect(box.height).toBe(160);
	});

	it('rejects non-positive dimensions', () => {
		expect(() => computeLetterbox(0, 10, 320)).toThrow(RangeError);
	});
});

describe('unletterboxBox', () => {
	it('round-trips a box through the letterbox transform', () => {
		const letterbox = computeLetterbox(640, 480, 320);
		const inInput = { x1: 0 + 10, y1: 40 + 5, x2: 0 + 110, y2: 40 + 55 };
		const result = unletterboxBox(inInput, letterbox, 640, 480);
		expect(result).toEqual({ x1: 20, y1: 10, x2: 220, y2: 110 });
	});

	it('clamps boxes that spill into the padding', () => {
		const letterbox = computeLetterbox(640, 480, 320);
		const result = unletterboxBox({ x1: -50, y1: 0, x2: 400, y2: 320 }, letterbox, 640, 480);
		expect(result).toEqual({ x1: 0, y1: 0, x2: 640, y2: 480 });
	});
});
