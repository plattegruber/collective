import { describe, expect, it } from 'vitest';
import { concat, cosineSimilarity, l2Normalize } from './vectors.js';

describe('vectors', () => {
	it('normalises to unit length and leaves zero vectors alone', () => {
		const unit = l2Normalize(new Float32Array([3, 4]));
		expect(unit[0]).toBeCloseTo(0.6);
		expect(unit[1]).toBeCloseTo(0.8);
		expect(Array.from(l2Normalize(new Float32Array([0, 0])))).toEqual([0, 0]);
	});
	it('computes cosine similarity', () => {
		expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([1, 0]))).toBeCloseTo(1);
		expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBeCloseTo(0);
		expect(() => cosineSimilarity(new Float32Array([1]), new Float32Array([1, 2]))).toThrow(
			RangeError
		);
	});
	it('concatenates', () => {
		expect(Array.from(concat(new Float32Array([1]), new Float32Array([2, 3])))).toEqual([1, 2, 3]);
	});
});
