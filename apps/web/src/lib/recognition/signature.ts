/**
 * Tiny luminance thumbnails that tell, between embeddings, whether a tracked region still
 * shows the same thing. Comparing two 8×8 signatures costs nothing next to an embedding,
 * so a region whose content changed (the camera panned to another piece) can be re-embedded
 * at once instead of waiting out the refresh interval.
 */

export const SIGNATURE_SIZE = 8;

/** Rec. 601 luminance of each pixel, scaled to 0..1. */
export function luminanceSignature(rgba: Uint8ClampedArray): Float32Array {
	const pixels = rgba.length / 4;
	const out = new Float32Array(pixels);
	for (let i = 0, j = 0; i < pixels; i += 1, j += 4) {
		out[i] = (0.299 * rgba[j]! + 0.587 * rgba[j + 1]! + 0.114 * rgba[j + 2]!) / 255;
	}
	return out;
}

/** Mean absolute difference between two signatures of equal length, 0..1. */
export function signatureDistance(a: Float32Array, b: Float32Array): number {
	if (a.length !== b.length) throw new RangeError('Signatures differ in length.');
	if (a.length === 0) return 0;
	let sum = 0;
	for (let i = 0; i < a.length; i += 1) sum += Math.abs(a[i]! - b[i]!);
	return sum / a.length;
}
