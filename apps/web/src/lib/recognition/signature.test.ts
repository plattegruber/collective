import { describe, expect, it } from 'vitest';
import { luminanceSignature, signatureDistance } from './signature.js';

describe('luminanceSignature', () => {
	it('weights green most and ignores alpha', () => {
		const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 128]);
		const signature = luminanceSignature(rgba);
		expect(signature[1]).toBeGreaterThan(signature[0]!);
		expect(signature[0]).toBeGreaterThan(signature[2]!);
		expect(signature[1]).toBeCloseTo(0.587, 3);
	});
});

describe('signatureDistance', () => {
	it('is zero for identical content and grows with the change', () => {
		const a = new Float32Array([0.1, 0.5, 0.9]);
		expect(signatureDistance(a, a)).toBe(0);
		expect(signatureDistance(a, new Float32Array([0.2, 0.5, 0.9]))).toBeCloseTo(1 / 30);
		expect(signatureDistance(a, new Float32Array([0.9, 0.5, 0.1]))).toBeCloseTo(1.6 / 3);
	});

	it('rejects mismatched lengths', () => {
		expect(() => signatureDistance(new Float32Array(2), new Float32Array(3))).toThrow(RangeError);
	});
});
