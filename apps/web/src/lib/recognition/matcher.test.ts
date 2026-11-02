import { describe, expect, it } from 'vitest';
import { PrototypeIndex } from './matcher.js';
import { bytesToBase64Url } from '#lib/utils/encoding.js';

function encode(values: number[]): string {
	const bytes = new Uint8Array(values.length * 4);
	const view = new DataView(bytes.buffer);
	values.forEach((v, i) => view.setFloat32(i * 4, v, true));
	return bytesToBase64Url(bytes);
}

describe('PrototypeIndex', () => {
	const index = PrototypeIndex.fromPayload({
		releaseId: 'r1',
		dim: 3,
		artworks: {
			a: [encode([1, 0, 0]), encode([0.9, 0.1, 0])],
			b: [encode([0, 1, 0])]
		}
	});

	it('decodes payloads and counts prototypes', () => {
		expect(index.size).toBe(3);
		expect(index.artworkIds.sort()).toEqual(['a', 'b']);
		expect(index.has('a')).toBe(true);
		expect(index.has('zzz')).toBe(false);
	});

	it('matches the nearest artwork using its best prototype', () => {
		const match = index.match(new Float32Array([0.95, 0.05, 0]));
		expect(match?.artworkId).toBe('a');
		expect(match?.similarity).toBeGreaterThan(0.99);
		expect(match?.runnerUp?.artworkId).toBe('b');
	});

	it('reports similarity to a specific artwork', () => {
		expect(index.similarityTo('b', new Float32Array([0, 1, 0]))).toBeCloseTo(1);
		expect(index.similarityTo('missing', new Float32Array([0, 1, 0]))).toBe(-1);
	});

	it('rejects vectors and prototypes of the wrong length', () => {
		expect(() => index.add('a', new Float32Array([1, 2]))).toThrow(RangeError);
		expect(() =>
			PrototypeIndex.fromPayload({ releaseId: 'r', dim: 2, artworks: { a: [encode([1, 2, 3])] } })
		).toThrow(RangeError);
	});

	it('returns null on an empty index', () => {
		expect(new PrototypeIndex('r', 2).match(new Float32Array([1, 0]))).toBeNull();
	});
});
