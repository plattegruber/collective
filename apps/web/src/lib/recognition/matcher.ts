import { base64UrlToBytes } from '#lib/utils/encoding.js';
import type { PrototypeIndexPayload } from '#lib/domain/recognition.js';
import { cosineSimilarity } from './vectors.js';
import type { Match } from './types.js';

function decodeVector(encoded: string, dim: number): Float32Array {
	const bytes = base64UrlToBytes(encoded);
	if (bytes.byteLength !== dim * 4) throw new RangeError('Prototype has the wrong length.');
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const out = new Float32Array(dim);
	for (let i = 0; i < dim; i += 1) out[i] = view.getFloat32(i * 4, true);
	return out;
}

/**
 * The collector's pieces as vectors. Matching is nearest-prototype with cosine similarity;
 * an artwork's score is the best of its prototypes, so extra photos and scan memories only
 * ever help.
 */
export class PrototypeIndex {
	readonly #vectors = new Map<string, Float32Array[]>();
	readonly releaseId: string;
	readonly dim: number;

	constructor(releaseId: string, dim: number) {
		this.releaseId = releaseId;
		this.dim = dim;
	}

	static fromPayload(payload: PrototypeIndexPayload): PrototypeIndex {
		const index = new PrototypeIndex(payload.releaseId, payload.dim);
		for (const [artworkId, encoded] of Object.entries(payload.artworks)) {
			for (const vector of encoded) index.add(artworkId, decodeVector(vector, payload.dim));
		}
		return index;
	}

	get size(): number {
		let total = 0;
		for (const list of this.#vectors.values()) total += list.length;
		return total;
	}

	get artworkIds(): string[] {
		return [...this.#vectors.keys()];
	}

	has(artworkId: string): boolean {
		return this.#vectors.has(artworkId);
	}

	add(artworkId: string, vector: Float32Array): void {
		if (vector.length !== this.dim) throw new RangeError('Vector has the wrong length.');
		const list = this.#vectors.get(artworkId) ?? [];
		list.push(vector);
		this.#vectors.set(artworkId, list);
	}

	/** Highest similarity to any prototype of `artworkId`, or -1 when it has none. */
	similarityTo(artworkId: string, vector: Float32Array): number {
		let best = -1;
		for (const prototype of this.#vectors.get(artworkId) ?? []) {
			best = Math.max(best, cosineSimilarity(prototype, vector));
		}
		return best;
	}

	match(vector: Float32Array): Match | null {
		let best: { artworkId: string; similarity: number } | null = null;
		let second: { artworkId: string; similarity: number } | null = null;
		for (const artworkId of this.#vectors.keys()) {
			const similarity = this.similarityTo(artworkId, vector);
			if (!best || similarity > best.similarity) {
				second = best;
				best = { artworkId, similarity };
			} else if (!second || similarity > second.similarity) {
				second = { artworkId, similarity };
			}
		}
		return best ? { ...best, runnerUp: second } : null;
	}
}
