import { describe, expect, it } from 'vitest';
import { alignShots, fuseShots } from './fuse.js';
import { createRaster, type Raster } from './raster.js';

/** A textured "artwork" with an additive bright blob (the reflection) at `blob`. */
function shot(blob: { x: number; y: number } | null, shift = { x: 0, y: 0 }, size = 160): Raster {
	const r = createRaster(size, size);
	for (let y = 0; y < size; y += 1) {
		for (let x = 0; x < size; x += 1) {
			const sx = x - shift.x;
			const sy = y - shift.y;
			let v = 60 + ((sx * 7 + sy * 13) % 90) + (Math.floor(sx / 20) % 2 === 0 ? 30 : 0);
			if (blob) {
				const d = Math.hypot(x - blob.x, y - blob.y);
				v += Math.max(0, 120 - d * 4);
			}
			r.data.set([v, v * 0.9, v * 0.8, 255], (y * size + x) * 4);
		}
	}
	return r;
}

describe('fuseShots', () => {
	it('removes a reflection present in only one shot and reports the change', () => {
		const clean = shot(null);
		const a = shot({ x: 60, y: 60 });
		const b = shot({ x: 110, y: 100 });
		const fused = fuseShots([a, b]);
		let error = 0;
		let worst = 0;
		for (let p = 0; p < clean.data.length; p += 4) {
			const d = Math.abs(fused.raster.data[p]! - clean.data[p]!);
			error += d;
			worst = Math.max(worst, d);
		}
		error /= clean.data.length / 4;
		expect(error).toBeLessThan(6);
		expect(worst).toBeLessThan(60);
		expect(fused.changed).toBeGreaterThan(0.02);
	});

	it('aligns a shot that is translated against the reference', () => {
		const a = shot(null);
		const b = shot(null, { x: 5, y: -3 });
		const shifts = alignShots([a, b], 0, 12);
		expect(shifts[0]).toEqual({ x: 0, y: 0 });
		expect(Math.abs(shifts[1]!.x - 5)).toBeLessThanOrEqual(1);
		expect(Math.abs(shifts[1]!.y + 3)).toBeLessThanOrEqual(1);
	});

	it('returns a single shot untouched', () => {
		const a = shot(null);
		expect(fuseShots([a]).raster).toBe(a);
	});
});
