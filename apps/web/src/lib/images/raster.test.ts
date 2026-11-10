import { describe, expect, it } from 'vitest';
import { boxBlur, createRaster, downscale, luminance, percentile, sobel } from './raster.js';

describe('raster helpers', () => {
	it('downscales with a box filter', () => {
		const r = createRaster(4, 2);
		for (let x = 0; x < 4; x += 1) {
			r.data.set([x < 2 ? 0 : 200, 0, 0, 255], x * 4);
			r.data.set([x < 2 ? 0 : 200, 0, 0, 255], (4 + x) * 4);
		}
		const small = downscale(r, 2);
		expect([small.width, small.height]).toEqual([2, 1]);
		expect(small.data[0]).toBe(0);
		expect(small.data[4]).toBe(200);
	});

	it('computes luminance, gradients and percentiles', () => {
		const r = createRaster(3, 3);
		r.data.fill(255);
		for (let y = 0; y < 3; y += 1) r.data.set([0, 0, 0, 255], (y * 3 + 2) * 4); // dark right column
		const luma = luminance(r);
		expect(luma[0]).toBeCloseTo(255, 3);
		expect(luma[2]).toBe(0);
		const g = sobel(luma, 3, 3);
		expect(g.gx[4]).toBeLessThan(0);
		expect(g.gy[4]).toBe(0);
		expect(percentile(luma, 0.5)).toBeCloseTo(255, 0);
		expect(percentile(luma, 0.2)).toBe(0);
	});

	it('blurs with clamped edges and preserves a flat plane', () => {
		const flat = new Float32Array(16).fill(7);
		expect(Array.from(boxBlur(flat, 4, 4, 1))).toEqual(Array.from(flat));
		const spike = new Float32Array(9);
		spike[4] = 9;
		const blurred = boxBlur(spike, 3, 3, 1);
		expect(blurred[4]).toBeCloseTo(1, 5);
	});
});
