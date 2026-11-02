import { describe, expect, it } from 'vitest';
import {
	boxArea,
	boxIoU,
	chooseArtworkRegion,
	containment,
	expandBox,
	lerpBox,
	normalizeBox,
	unionBox
} from './boxes.js';

describe('boxes', () => {
	const a = { x1: 0, y1: 0, x2: 100, y2: 100 };
	it('measures containment', () => {
		expect(containment({ x1: 10, y1: 10, x2: 20, y2: 20 }, a)).toBe(1);
		expect(containment({ x1: 90, y1: 0, x2: 110, y2: 100 }, a)).toBeCloseTo(0.5);
	});
	it('computes area and IoU', () => {
		expect(boxArea(a)).toBe(10_000);
		expect(boxIoU(a, { x1: 50, y1: 0, x2: 150, y2: 100 })).toBeCloseTo(1 / 3);
		expect(boxIoU(a, { x1: 200, y1: 200, x2: 300, y2: 300 })).toBe(0);
	});
	it('expands with a margin and clamps to the frame', () => {
		expect(expandBox(a, 0.1, 105, 1000)).toEqual({ x1: 0, y1: 0, x2: 105, y2: 110 });
	});
	it('interpolates and normalises', () => {
		expect(lerpBox(a, { x1: 100, y1: 100, x2: 200, y2: 200 }, 0.5)).toEqual({
			x1: 50,
			y1: 50,
			x2: 150,
			y2: 150
		});
		expect(normalizeBox(a, 200, 400)).toEqual({ x1: 0, y1: 0, x2: 0.5, y2: 0.25 });
	});
});

describe('chooseArtworkRegion', () => {
	const box = (x1: number, y1: number, x2: number, y2: number) => ({ box: { x1, y1, x2, y2 } });
	it('returns null when nothing sizeable was detected', () => {
		expect(chooseArtworkRegion([], 1000, 1000)).toBeNull();
		expect(chooseArtworkRegion([box(0, 0, 100, 100)], 1000, 1000)).toBeNull();
	});
	it('unions multi-panel detections that together cover the photo', () => {
		const panels = [box(0, 0, 300, 300), box(700, 0, 1000, 300), box(0, 700, 300, 1000)];
		expect(chooseArtworkRegion(panels, 1000, 1000)).toEqual({ x1: 0, y1: 0, x2: 1000, y2: 1000 });
	});
	it('ignores tiny detections but keeps a dominant one', () => {
		expect(chooseArtworkRegion([box(100, 100, 900, 900), box(0, 0, 50, 50)], 1000, 1000)).toEqual({
			x1: 100,
			y1: 100,
			x2: 900,
			y2: 900
		});
	});
	it('unionBox refuses an empty list', () => {
		expect(() => unionBox([])).toThrow(RangeError);
	});
});
