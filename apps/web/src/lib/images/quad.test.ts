import { describe, expect, it } from 'vitest';
import { boxToQuad, type Quad } from './geometry.js';
import { findQuad } from './quad.js';
import { createRaster, type Raster } from './raster.js';

/** A light wall with a dark-framed, perspective-skewed picture drawn into it. */
function scene(quad: Quad, width = 400, height = 400): Raster {
	const wall = createRaster(width, height);
	wall.data.fill(225);
	const inside = (x: number, y: number) => {
		let sign = 0;
		for (let i = 0; i < 4; i += 1) {
			const a = quad[i]!;
			const b = quad[(i + 1) % 4]!;
			const s = Math.sign((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x));
			if (sign === 0) sign = s;
			else if (s !== 0 && s !== sign) return false;
		}
		return true;
	};
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			if (!inside(x + 0.5, y + 0.5)) continue;
			// Distance to the nearest quad edge decides frame vs art, a good enough stand-in.
			let minDistance = Infinity;
			for (let i = 0; i < 4; i += 1) {
				const a = quad[i]!;
				const b = quad[(i + 1) % 4]!;
				const len = Math.hypot(b.x - a.x, b.y - a.y);
				const d = Math.abs((b.x - a.x) * (a.y - y) - (a.x - x) * (b.y - a.y)) / len;
				minDistance = Math.min(minDistance, d);
			}
			const v = minDistance < 10 ? 30 : 120 + ((x * 3 + y * 5) % 60);
			wall.data.set([v, v, v, 255], (y * width + x) * 4);
		}
	}
	return wall;
}

describe('findQuad', () => {
	it('recovers the corners of a skewed frame from a loose box', () => {
		const quad: Quad = [
			{ x: 90, y: 70 },
			{ x: 320, y: 90 },
			{ x: 300, y: 330 },
			{ x: 110, y: 300 }
		];
		const found = findQuad(
			scene(quad),
			{ x1: 70, y1: 50, x2: 340, y2: 350 },
			{ analysisEdge: 400 }
		);
		expect(found.confidence).toBeGreaterThan(0.5);
		found.quad.forEach((corner, i) => {
			expect(Math.abs(corner.x - quad[i]!.x)).toBeLessThanOrEqual(3);
			expect(Math.abs(corner.y - quad[i]!.y)).toBeLessThanOrEqual(3);
		});
	});

	it('falls back to the box on a featureless image', () => {
		const flat = createRaster(300, 300);
		flat.data.fill(200);
		const box = { x1: 40, y1: 40, x2: 260, y2: 260 };
		const found = findQuad(flat, box, { analysisEdge: 300 });
		expect(found.confidence).toBe(0);
		expect(found.quad).toEqual(boxToQuad(40, 40, 260, 260));
	});
});
