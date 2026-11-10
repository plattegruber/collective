import { describe, expect, it } from 'vitest';
import { boxToQuad, type Quad } from './geometry.js';
import { createRaster } from './raster.js';
import { warpPerspective } from './warp.js';

describe('warpPerspective', () => {
	it('copies an axis-aligned quad exactly', () => {
		const src = createRaster(8, 8);
		for (let i = 0; i < 64; i += 1) src.data.set([i * 4, 255 - i * 4, 7, 255], i * 4);
		const out = warpPerspective(src, boxToQuad(2, 2, 6, 6), { width: 4, height: 4 });
		for (let y = 0; y < 4; y += 1) {
			for (let x = 0; x < 4; x += 1) {
				const i = (y + 2) * 8 + (x + 2);
				const o = (y * 4 + x) * 4;
				expect(out.data[o]).toBe(i * 4);
				expect(out.data[o + 1]).toBe(255 - i * 4);
			}
		}
	});

	it('straightens a skewed quad so its corners land on the output corners', () => {
		// A bright skewed quadrilateral on black; after warping, the output should be bright
		// everywhere and the outside (never sampled) is irrelevant.
		const src = createRaster(100, 100);
		const quad: Quad = [
			{ x: 20, y: 10 },
			{ x: 80, y: 20 },
			{ x: 70, y: 90 },
			{ x: 10, y: 80 }
		];
		const inside = (x: number, y: number) => {
			let sign = 0;
			for (let i = 0; i < 4; i += 1) {
				const a = quad[i]!;
				const b = quad[(i + 1) % 4]!;
				const cross = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
				const s = Math.sign(cross);
				if (sign === 0) sign = s;
				else if (s !== 0 && s !== sign) return false;
			}
			return true;
		};
		for (let y = 0; y < 100; y += 1)
			for (let x = 0; x < 100; x += 1)
				if (inside(x + 0.5, y + 0.5)) src.data.set([255, 255, 255, 255], (y * 100 + x) * 4);
		const out = warpPerspective(src, quad, { width: 50, height: 60 });
		let bright = 0;
		for (let i = 0; i < 50 * 60; i += 1) if (out.data[i * 4]! > 200) bright += 1;
		expect(bright / (50 * 60)).toBeGreaterThan(0.95);
	});
});
