import { describe, expect, it } from 'vitest';
import { analyseColour, applyColourCorrection, isIdentity } from './enhance.js';
import { createRaster } from './raster.js';

describe('analyseColour', () => {
	it('neutralises a warm cast using the bright mat and lifts a muddy black point', () => {
		const r = createRaster(64, 64);
		for (let y = 0; y < 64; y += 1) {
			for (let x = 0; x < 64; x += 1) {
				const mat = x < 8 || y < 8 || x >= 56 || y >= 56;
				// Warm light: red up, blue down. The art is mid grey with a shadow that only reaches 25.
				const base = mat ? 230 : x < 30 && y < 30 ? 25 : 110;
				r.data.set([Math.min(255, base * 1.1), base, base * 0.9, 255], (y * 64 + x) * 4);
			}
		}
		const correction = analyseColour(r);
		expect(correction.gains[0]).toBeLessThan(1);
		expect(correction.gains[2]).toBeGreaterThan(1);
		expect(correction.black).toBeGreaterThan(0);
		const fixed = applyColourCorrection(r, correction);
		const p = (20 * 64 + 20) * 4;
		expect(Math.abs(fixed.data[p]! - fixed.data[p + 2]!)).toBeLessThan(
			Math.abs(r.data[p]! - r.data[p + 2]!)
		);
	});

	it('leaves a colourful picture without neutral surfaces alone', () => {
		const r = createRaster(32, 32);
		for (let i = 0; i < 32 * 32; i += 1) r.data.set([200, 40, 30, 255], i * 4);
		const correction = analyseColour(r);
		expect(correction.gains).toEqual([1, 1, 1]);
		expect(isIdentity({ gains: [1, 1, 1], black: 0, scale: 1 })).toBe(true);
	});
});
