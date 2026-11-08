import { describe, expect, it } from 'vitest';
import { rgbaToChwAutoLevels, rgbaToChwFloat } from './preprocess.js';

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

describe('rgbaToChwAutoLevels', () => {
	const grid = (pixels: number[][], width: number) => {
		const rgba = new Uint8ClampedArray(pixels.length * 4);
		pixels.forEach(([r, g, b], i) => rgba.set([r!, g!, b!, 255], i * 4));
		return { rgba, width, height: pixels.length / width };
	};

	it('stretches a dim frame so its darkest pixel is black and brightest is white', () => {
		const { rgba, width, height } = grid(
			[
				[20, 20, 20],
				[40, 40, 40],
				[60, 60, 60],
				[80, 80, 80]
			],
			2
		);
		const out = rgbaToChwAutoLevels(rgba, width, height, { left: 0, top: 0, width, height });
		expect(out[0]).toBe(0);
		expect(out[3]).toBe(1);
		expect(out[1]).toBeCloseTo(1 / 3, 2);
	});

	it('neutralises a colour cast with per-channel gains', () => {
		const { rgba, width, height } = grid(
			[
				[120, 100, 80],
				[60, 50, 40]
			],
			2
		);
		const out = rgbaToChwAutoLevels(rgba, width, height, { left: 0, top: 0, width, height });
		const plane = width * height;
		// After white balance both pixels are gray, so the three channels agree.
		expect(out[0]).toBeCloseTo(out[plane]!, 5);
		expect(out[0]).toBeCloseTo(out[2 * plane]!, 5);
	});

	it('ignores letterbox padding and leaves it black', () => {
		const { rgba, width, height } = grid(
			[
				[0, 0, 0],
				[0, 0, 0],
				[90, 90, 90],
				[110, 110, 110]
			],
			1
		);
		const out = rgbaToChwAutoLevels(rgba, width, height, { left: 0, top: 2, width: 1, height: 2 });
		expect(out[0]).toBe(0);
		expect(out[1]).toBe(0);
		expect(out[2]).toBe(0);
		expect(out[3]).toBe(1);
	});

	it('falls back to a plain conversion when the region is empty', () => {
		const { rgba, width, height } = grid([[255, 0, 0]], 1);
		const out = rgbaToChwAutoLevels(rgba, width, height, { left: 0, top: 0, width: 0, height: 0 });
		expect(Array.from(out)).toEqual([1, 0, 0]);
	});
});
