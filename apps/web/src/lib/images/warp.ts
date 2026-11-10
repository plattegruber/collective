import { boxToQuad, solveHomography, type Quad, type Size } from './geometry.js';
import { createRaster, type Raster } from './raster.js';

/**
 * Resamples the quad of `source` into an upright `size` rectangle with bilinear
 * interpolation. Every output pixel is mapped back into the source (inverse warping), so
 * there are no holes; the projective terms are advanced incrementally along each row and
 * only the final division is per pixel.
 */
export function warpPerspective(source: Raster, quad: Quad, size: Size): Raster {
	const out = createRaster(size.width, size.height);
	const h = solveHomography(boxToQuad(0, 0, size.width, size.height), quad);
	const { data: src, width: sw, height: sh } = source;
	const dst = out.data;
	const maxX = sw - 1;
	const maxY = sh - 1;
	for (let y = 0; y < size.height; y += 1) {
		// Sample at pixel centres so a quad that is already axis-aligned copies exactly.
		const cy = y + 0.5;
		let nx = h[1]! * cy + h[2]! + h[0]! * 0.5;
		let ny = h[4]! * cy + h[5]! + h[3]! * 0.5;
		let nw = h[7]! * cy + h[8]! + h[6]! * 0.5;
		let o = y * size.width * 4;
		for (let x = 0; x < size.width; x += 1, o += 4) {
			const inv = 1 / nw;
			const sx = Math.min(maxX, Math.max(0, nx * inv - 0.5));
			const sy = Math.min(maxY, Math.max(0, ny * inv - 0.5));
			const x0 = Math.floor(sx);
			const y0 = Math.floor(sy);
			const x1 = Math.min(maxX, x0 + 1);
			const y1 = Math.min(maxY, y0 + 1);
			const fx = sx - x0;
			const fy = sy - y0;
			const w00 = (1 - fx) * (1 - fy);
			const w10 = fx * (1 - fy);
			const w01 = (1 - fx) * fy;
			const w11 = fx * fy;
			const p00 = (y0 * sw + x0) * 4;
			const p10 = (y0 * sw + x1) * 4;
			const p01 = (y1 * sw + x0) * 4;
			const p11 = (y1 * sw + x1) * 4;
			for (let c = 0; c < 3; c += 1) {
				dst[o + c] =
					src[p00 + c]! * w00 + src[p10 + c]! * w10 + src[p01 + c]! * w01 + src[p11 + c]! * w11;
			}
			dst[o + 3] = 255;
			nx += h[0]!;
			ny += h[3]!;
			nw += h[6]!;
		}
	}
	return out;
}
