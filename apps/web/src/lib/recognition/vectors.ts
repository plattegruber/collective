/** Small vector helpers shared by the Worker (prototype checks) and the browser (matching). */

export function l2Normalize(vector: Float32Array): Float32Array {
	let sum = 0;
	for (let i = 0; i < vector.length; i += 1) sum += vector[i]! * vector[i]!;
	const norm = Math.sqrt(sum);
	if (norm === 0) return vector;
	const out = new Float32Array(vector.length);
	for (let i = 0; i < vector.length; i += 1) out[i] = vector[i]! / norm;
	return out;
}

/** Cosine similarity; both inputs are expected to be unit vectors but it is robust either way. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
	if (a.length !== b.length) throw new RangeError('Vectors differ in length.');
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i += 1) {
		dot += a[i]! * b[i]!;
		na += a[i]! * a[i]!;
		nb += b[i]! * b[i]!;
	}
	if (na === 0 || nb === 0) return 0;
	return dot / Math.sqrt(na * nb);
}

export function concat(a: Float32Array, b: Float32Array): Float32Array {
	const out = new Float32Array(a.length + b.length);
	out.set(a, 0);
	out.set(b, a.length);
	return out;
}
