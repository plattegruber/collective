import { describe, expect, it } from 'vitest';
import { PrototypeIndex, acceptsMatch, effectiveThresholds } from './matcher.js';
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

	it('accepts only clear winners', () => {
		const thresholds = { match: 0.6, margin: 0.1 };
		const runnerUp = { artworkId: 'b', similarity: 0.62 };
		expect(acceptsMatch(null, thresholds)).toBe(false);
		expect(acceptsMatch({ artworkId: 'a', similarity: 0.55, runnerUp: null }, thresholds)).toBe(
			false
		);
		expect(acceptsMatch({ artworkId: 'a', similarity: 0.65, runnerUp: null }, thresholds)).toBe(
			true
		);
		expect(acceptsMatch({ artworkId: 'a', similarity: 0.7, runnerUp }, thresholds)).toBe(false);
		expect(acceptsMatch({ artworkId: 'a', similarity: 0.73, runnerUp }, thresholds)).toBe(true);
	});

	it('raises the bar with collection size but never lowers it', () => {
		const base = { match: 0.56, strong: 0.59, margin: 0.08, matchReference: 7, matchSlope: 0.034 };
		expect(effectiveThresholds(base, 7)).toEqual(base);
		expect(effectiveThresholds(base, 1).match).toBe(0.56);
		expect(effectiveThresholds(base, 100).match).toBeCloseTo(0.65, 2);
		expect(effectiveThresholds(base, 100).strong).toBeCloseTo(0.68, 2);
		expect(effectiveThresholds(base, 100).margin).toBe(0.08);
	});

	it('returns null on an empty index', () => {
		expect(new PrototypeIndex('r', 2).match(new Float32Array([1, 0]))).toBeNull();
	});
});
