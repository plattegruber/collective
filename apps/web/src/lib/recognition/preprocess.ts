/**
 * Converts interleaved RGBA pixels into planar CHW float32 in [0, 1], the layout the
 * detector expects. Writes into `out` when provided to avoid per-frame allocation.
 */
export function rgbaToChwFloat(
	rgba: Uint8ClampedArray,
	width: number,
	height: number,
	out?: Float32Array
): Float32Array {
	const plane = width * height;
	const target = out ?? new Float32Array(3 * plane);
	if (target.length !== 3 * plane) throw new RangeError('Output buffer has the wrong size.');
	for (let i = 0, j = 0; i < plane; i += 1, j += 4) {
		target[i] = rgba[j]! / 255;
		target[plane + i] = rgba[j + 1]! / 255;
		target[2 * plane + i] = rgba[j + 2]! / 255;
	}
	return target;
}

/**
 * Same layout as `rgbaToChwFloat` but standardised with per-channel mean/std, the input
 * contract of ImageNet-style backbones such as DINO.
 */
export function rgbaToChwNormalized(
	rgba: Uint8ClampedArray,
	width: number,
	height: number,
	mean: readonly [number, number, number],
	std: readonly [number, number, number],
	out?: Float32Array
): Float32Array {
	const plane = width * height;
	const target = out ?? new Float32Array(3 * plane);
	if (target.length !== 3 * plane) throw new RangeError('Output buffer has the wrong size.');
	for (let i = 0, j = 0; i < plane; i += 1, j += 4) {
		target[i] = (rgba[j]! / 255 - mean[0]) / std[0];
		target[plane + i] = (rgba[j + 1]! / 255 - mean[1]) / std[1];
		target[2 * plane + i] = (rgba[j + 2]! / 255 - mean[2]) / std[2];
	}
	return target;
}

export interface AutoLevelsOptions {
	/** Percentile of the drawn pixels that becomes black. */
	lowPercentile: number;
	/** Percentile of the drawn pixels that becomes white. */
	highPercentile: number;
	/** Per-channel gain limits for the gray-world white balance. */
	minGain: number;
	maxGain: number;
}

export const DEFAULT_AUTO_LEVELS: AutoLevelsOptions = {
	lowPercentile: 1,
	highPercentile: 99,
	minGain: 0.6,
	maxGain: 1.8
};

/** The part of a letterboxed frame that holds image pixels; the rest is padding. */
export interface PixelRegion {
	left: number;
	top: number;
	width: number;
	height: number;
}

const HISTOGRAM_BINS = 1024;

/**
 * Like `rgbaToChwFloat`, but first balances colour (gray world, per-channel gain) and
 * stretches contrast between two percentiles of the drawn region. Padding outside `region`
 * stays black. Dim, tungsten-lit rooms otherwise push the detector's confidence towards
 * zero; levelling the input restores it for about a millisecond per 320px frame.
 */
export function rgbaToChwAutoLevels(
	rgba: Uint8ClampedArray,
	width: number,
	height: number,
	region: PixelRegion,
	out?: Float32Array,
	options: AutoLevelsOptions = DEFAULT_AUTO_LEVELS
): Float32Array {
	const plane = width * height;
	const target = out ?? new Float32Array(3 * plane);
	if (target.length !== 3 * plane) throw new RangeError('Output buffer has the wrong size.');
	const left = Math.max(0, Math.floor(region.left));
	const top = Math.max(0, Math.floor(region.top));
	const right = Math.min(width, left + Math.floor(region.width));
	const bottom = Math.min(height, top + Math.floor(region.height));
	if (right <= left || bottom <= top) return rgbaToChwFloat(rgba, width, height, target);

	const sums: [number, number, number] = [0, 0, 0];
	for (let y = top; y < bottom; y += 1) {
		for (let x = left, j = (y * width + left) * 4; x < right; x += 1, j += 4) {
			sums[0] += rgba[j]!;
			sums[1] += rgba[j + 1]!;
			sums[2] += rgba[j + 2]!;
		}
	}
	const count = (right - left) * (bottom - top);
	const means = sums.map((sum) => sum / count) as [number, number, number];
	const gray = (means[0] + means[1] + means[2]) / 3;
	const gains = means.map((mean) =>
		Math.min(options.maxGain, Math.max(options.minGain, gray / Math.max(mean, 1e-3)))
	) as [number, number, number];

	const histogram = new Uint32Array(HISTOGRAM_BINS);
	for (let y = top; y < bottom; y += 1) {
		for (let x = left, j = (y * width + left) * 4; x < right; x += 1, j += 4) {
			for (let c = 0; c < 3; c += 1) {
				const bin = Math.min(HISTOGRAM_BINS - 1, Math.round(rgba[j + c]! * gains[c]!));
				histogram[bin] = (histogram[bin] ?? 0) + 1;
			}
		}
	}
	const total = count * 3;
	const percentile = (fraction: number): number => {
		const wanted = fraction * total;
		let seen = 0;
		for (let bin = 0; bin < HISTOGRAM_BINS; bin += 1) {
			seen += histogram[bin]!;
			if (seen >= wanted) return bin;
		}
		return HISTOGRAM_BINS - 1;
	};
	const low = percentile(options.lowPercentile / 100);
	const high = percentile(options.highPercentile / 100);
	const range = Math.max(1, high - low);

	target.fill(0);
	for (let y = top; y < bottom; y += 1) {
		for (let x = left, j = (y * width + left) * 4; x < right; x += 1, j += 4) {
			const i = y * width + x;
			for (let c = 0; c < 3; c += 1) {
				const value = (rgba[j + c]! * gains[c]! - low) / range;
				target[c * plane + i] = value < 0 ? 0 : value > 1 ? 1 : value;
			}
		}
	}
	return target;
}
