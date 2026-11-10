import { describe, expect, it } from 'vitest';
import {
	applyHomography,
	boundsQuad,
	boxToQuad,
	intersectLines,
	orderQuad,
	quadArea,
	quadIsConvex,
	rectifiedSize,
	solveHomography,
	transferQuad,
	type Quad
} from './geometry.js';

const skewed: Quad = [
	{ x: 10, y: 20 },
	{ x: 310, y: 40 },
	{ x: 290, y: 420 },
	{ x: 30, y: 400 }
];

describe('orderQuad', () => {
	it('puts the corners clockwise from the top-left regardless of input order', () => {
		const shuffled = [skewed[2], skewed[0], skewed[3], skewed[1]];
		expect(orderQuad(shuffled)).toEqual(skewed);
	});
});

describe('solveHomography', () => {
	it('maps every source corner onto its target', () => {
		const target = boxToQuad(0, 0, 300, 400);
		const h = solveHomography(skewed, target);
		skewed.forEach((corner, i) => {
			const mapped = applyHomography(h, corner);
			expect(mapped.x).toBeCloseTo(target[i]!.x, 6);
			expect(mapped.y).toBeCloseTo(target[i]!.y, 6);
		});
	});

	it('rejects a degenerate quad', () => {
		const flat: Quad = [
			{ x: 0, y: 0 },
			{ x: 1, y: 0 },
			{ x: 2, y: 0 },
			{ x: 3, y: 0 }
		];
		expect(() => solveHomography(flat, boxToQuad(0, 0, 1, 1))).toThrow(RangeError);
	});
});

describe('rectifiedSize', () => {
	it('averages the longer sides and caps the long edge', () => {
		const size = rectifiedSize(boxToQuad(0, 0, 4000, 2000), 3200);
		expect(size).toEqual({ width: 3200, height: 1600 });
	});

	it('honours a known aspect ratio without shrinking the long side', () => {
		const size = rectifiedSize(boxToQuad(0, 0, 1000, 1000), 3200, 2);
		expect(size).toEqual({ width: 2000, height: 1000 });
	});
});

describe('boundsQuad and transferQuad', () => {
	it('bounds a skewed quad', () => {
		expect(boundsQuad(skewed)).toEqual(boxToQuad(10, 20, 310, 420));
	});

	it('keeps corners at the same relative position inside a new box', () => {
		const from = boxToQuad(0, 0, 100, 200);
		const to = boxToQuad(50, 50, 250, 450);
		const moved = transferQuad(boxToQuad(10, 20, 90, 180), from, to);
		expect(moved).toEqual(boxToQuad(70, 90, 230, 410));
	});
});

describe('quad helpers', () => {
	it('measures area and convexity', () => {
		expect(quadArea(boxToQuad(0, 0, 10, 5))).toBe(50);
		expect(quadIsConvex(skewed)).toBe(true);
		const bowtie: Quad = [
			{ x: 0, y: 0 },
			{ x: 10, y: 10 },
			{ x: 10, y: 0 },
			{ x: 0, y: 10 }
		];
		expect(quadIsConvex(bowtie)).toBe(false);
	});

	it('intersects lines and reports parallel ones', () => {
		const p = intersectLines({ x: 0, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }, { x: 10, y: 0 });
		expect(p).toEqual({ x: 5, y: 5 });
		expect(
			intersectLines({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 })
		).toBeNull();
	});
});
