import { describe, expect, it } from 'vitest';
import { rgbaToChwFloat } from './preprocess.js';

describe('rgbaToChwFloat', () => {
	it('splits interleaved RGBA into planar RGB scaled to [0,1]', () => {
		const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]);
		const out = rgbaToChwFloat(rgba, 2, 1);
		expect(Array.from(out)).toEqual([1, 0, 0, 1, 0, 0]);
	});

	it('reuses a provided buffer and validates its size', () => {
		const rgba = new Uint8ClampedArray(4);
		const buffer = new Float32Array(3);
		expect(rgbaToChwFloat(rgba, 1, 1, buffer)).toBe(buffer);
		expect(() => rgbaToChwFloat(rgba, 1, 1, new Float32Array(2))).toThrow(RangeError);
	});
});
