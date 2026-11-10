/**
 * Small pixel-buffer helpers shared by the photo tools. A `Raster` is the shape of
 * `ImageData`, so canvases feed in and out without copying.
 */

export interface Raster {
	data: Uint8ClampedArray<ArrayBuffer>;
	width: number;
	height: number;
}

export function createRaster(width: number, height: number): Raster {
	return { data: new Uint8ClampedArray(width * height * 4), width, height };
}

/** Rec. 601 luma in 0..255 as floats, one value per pixel. */
export function luminance(raster: Raster): Float32Array {
	const { data, width, height } = raster;
	const out = new Float32Array(width * height);
	for (let i = 0, p = 0; i < out.length; i += 1, p += 4) {
		out[i] = 0.299 * data[p]! + 0.587 * data[p + 1]! + 0.114 * data[p + 2]!;
	}
	return out;
}

/**
 * Box-filter downscale by an integer factor. Fast and alias-free enough for analysis
 * (edges, alignment, glare maps); display scaling stays with the canvas.
 */
export function downscale(raster: Raster, factor: number): Raster {
	if (factor <= 1) return raster;
	const width = Math.max(1, Math.floor(raster.width / factor));
	const height = Math.max(1, Math.floor(raster.height / factor));
	const out = createRaster(width, height);
	const norm = 1 / (factor * factor);
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			let r = 0;
			let g = 0;
			let b = 0;
			for (let dy = 0; dy < factor; dy += 1) {
				let p = ((y * factor + dy) * raster.width + x * factor) * 4;
				for (let dx = 0; dx < factor; dx += 1, p += 4) {
					r += raster.data[p]!;
					g += raster.data[p + 1]!;
					b += raster.data[p + 2]!;
				}
			}
			const o = (y * width + x) * 4;
			out.data[o] = r * norm;
			out.data[o + 1] = g * norm;
			out.data[o + 2] = b * norm;
			out.data[o + 3] = 255;
		}
	}
	return out;
}

/** The integer factor that brings the long edge to at most `maxEdge`. */
export function downscaleFactor(width: number, height: number, maxEdge: number): number {
	return Math.max(1, Math.ceil(Math.max(width, height) / maxEdge));
}

export interface Gradient {
	/** Horizontal derivative (positive when brighter to the right). */
	gx: Float32Array;
	/** Vertical derivative (positive when brighter below). */
	gy: Float32Array;
	width: number;
	height: number;
}

/** Sobel derivatives of a luminance plane; the one-pixel border is zero. */
export function sobel(luma: Float32Array, width: number, height: number): Gradient {
	const gx = new Float32Array(width * height);
	const gy = new Float32Array(width * height);
	for (let y = 1; y < height - 1; y += 1) {
		for (let x = 1; x < width - 1; x += 1) {
			const i = y * width + x;
			const tl = luma[i - width - 1]!;
			const t = luma[i - width]!;
			const tr = luma[i - width + 1]!;
			const l = luma[i - 1]!;
			const r = luma[i + 1]!;
			const bl = luma[i + width - 1]!;
			const b = luma[i + width]!;
			const br = luma[i + width + 1]!;
			gx[i] = tr + 2 * r + br - tl - 2 * l - bl;
			gy[i] = bl + 2 * b + br - tl - 2 * t - tr;
		}
	}
	return { gx, gy, width, height };
}

/** Separable box blur of a float plane (radius in pixels), edges clamped. */
export function boxBlur(
	plane: Float32Array,
	width: number,
	height: number,
	radius: number
): Float32Array {
	if (radius <= 0) return plane;
	const tmp = new Float32Array(plane.length);
	const out = new Float32Array(plane.length);
	const span = radius * 2 + 1;
	for (let y = 0; y < height; y += 1) {
		const row = y * width;
		let sum = 0;
		for (let k = -radius; k <= radius; k += 1)
			sum += plane[row + Math.min(width - 1, Math.max(0, k))]!;
		for (let x = 0; x < width; x += 1) {
			tmp[row + x] = sum / span;
			const add = Math.min(width - 1, x + radius + 1);
			const drop = Math.max(0, x - radius);
			sum += plane[row + add]! - plane[row + drop]!;
		}
	}
	for (let x = 0; x < width; x += 1) {
		let sum = 0;
		for (let k = -radius; k <= radius; k += 1)
			sum += tmp[Math.min(height - 1, Math.max(0, k)) * width + x]!;
		for (let y = 0; y < height; y += 1) {
			out[y * width + x] = sum / span;
			const add = Math.min(height - 1, y + radius + 1);
			const drop = Math.max(0, y - radius);
			sum += tmp[add * width + x]! - tmp[drop * width + x]!;
		}
	}
	return out;
}

/** Value below which `fraction` of the plane lies (0..1), via a 1024-bin histogram over 0..255. */
export function percentile(plane: ArrayLike<number>, fraction: number): number {
	const bins = new Uint32Array(1024);
	for (let i = 0; i < plane.length; i += 1) {
		const v = Math.min(1023, Math.max(0, Math.round((plane[i]! / 255) * 1023)));
		bins[v] = bins[v]! + 1;
	}
	const target = fraction * plane.length;
	let seen = 0;
	for (let b = 0; b < 1024; b += 1) {
		seen += bins[b]!;
		if (seen >= target) return (b / 1023) * 255;
	}
	return 255;
}
