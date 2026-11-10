import { downscale, downscaleFactor, luminance, percentile, type Raster } from './raster.js';

/**
 * A conservative colour correction for photos of art: neutralise the light's cast using the
 * bright, unsaturated surfaces in the picture (mat, wall, paper) and lift a muddy black
 * point. Gains and stretch are clamped so the artwork's own palette is never "improved".
 */
export interface ColourCorrection {
	gains: [number, number, number];
	/** Input value mapped to black. */
	black: number;
	/** Multiplier applied after subtracting `black`. */
	scale: number;
}

export const IDENTITY_CORRECTION: ColourCorrection = { gains: [1, 1, 1], black: 0, scale: 1 };

export interface EnhanceOptions {
	/** Pixels at or above this luminance percentile are white-balance candidates. */
	brightPercentile: number;
	/** Candidates must be this unsaturated ((max - min) / max). */
	maxSaturation: number;
	/** Minimum share of pixels that must qualify before gains are trusted. */
	minCandidateShare: number;
	gainRange: [number, number];
	/** Black points above this are left alone (the picture is meant to be low-key). */
	maxBlackLift: number;
	maxScale: number;
}

export const DEFAULT_ENHANCE_OPTIONS: EnhanceOptions = {
	brightPercentile: 0.85,
	maxSaturation: 0.25,
	minCandidateShare: 0.01,
	gainRange: [0.85, 1.2],
	maxBlackLift: 40,
	maxScale: 1.25
};

export function analyseColour(
	raster: Raster,
	options: Partial<EnhanceOptions> = {}
): ColourCorrection {
	const opts = { ...DEFAULT_ENHANCE_OPTIONS, ...options };
	const small = downscale(raster, downscaleFactor(raster.width, raster.height, 320));
	const luma = luminance(small);
	// Half a histogram bin below the percentile, so the pixels that set it still qualify.
	const bright = percentile(luma, opts.brightPercentile) - 0.5;
	let r = 0;
	let g = 0;
	let b = 0;
	let count = 0;
	for (let i = 0, p = 0; i < luma.length; i += 1, p += 4) {
		if (luma[i]! < bright) continue;
		const cr = small.data[p]!;
		const cg = small.data[p + 1]!;
		const cb = small.data[p + 2]!;
		const max = Math.max(cr, cg, cb);
		if (max === 0 || (max - Math.min(cr, cg, cb)) / max > opts.maxSaturation) continue;
		r += cr;
		g += cg;
		b += cb;
		count += 1;
	}
	let gains: [number, number, number] = [1, 1, 1];
	if (count >= luma.length * opts.minCandidateShare && r > 0 && g > 0 && b > 0) {
		const mean = (r + g + b) / 3;
		const clamp = (v: number) => Math.min(opts.gainRange[1], Math.max(opts.gainRange[0], v));
		gains = [clamp(mean / r), clamp(mean / g), clamp(mean / b)];
	}
	const black = percentile(luma, 0.002);
	const white = percentile(luma, 0.998);
	const lift = black <= opts.maxBlackLift ? black : 0;
	const scale = Math.min(opts.maxScale, 255 / Math.max(1, Math.max(white, 200) - lift));
	return { gains, black: lift, scale };
}

export function isIdentity(correction: ColourCorrection): boolean {
	return (
		correction.black === 0 && correction.scale === 1 && correction.gains.every((gain) => gain === 1)
	);
}

/** Applies a correction through per-channel lookup tables; returns a new raster. */
export function applyColourCorrection(raster: Raster, correction: ColourCorrection): Raster {
	const luts = correction.gains.map((gain) => {
		const lut = new Uint8ClampedArray(256);
		for (let v = 0; v < 256; v += 1) lut[v] = (v * gain - correction.black) * correction.scale;
		return lut;
	}) as [Uint8ClampedArray, Uint8ClampedArray, Uint8ClampedArray];
	const out = new Uint8ClampedArray(raster.data.length);
	const { data } = raster;
	for (let p = 0; p < data.length; p += 4) {
		out[p] = luts[0][data[p]!]!;
		out[p + 1] = luts[1][data[p + 1]!]!;
		out[p + 2] = luts[2][data[p + 2]!]!;
		out[p + 3] = data[p + 3]!;
	}
	return { data: out, width: raster.width, height: raster.height };
}
