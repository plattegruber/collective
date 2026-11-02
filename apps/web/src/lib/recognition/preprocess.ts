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
