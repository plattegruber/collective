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
