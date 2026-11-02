import { describe, expect, it } from 'vitest';
import { RegionTracker } from './regionTracker.js';

const box = (x1: number, y1: number, x2: number, y2: number) => ({ x1, y1, x2, y2 });

describe('RegionTracker', () => {
	it('keeps the same id for an overlapping detection and smooths its box', () => {
		const tracker = new RegionTracker({ boxSmoothing: 0.5 });
		const [first] = tracker.update([{ score: 0.9, box: box(0, 0, 100, 100) }]);
		const [second] = tracker.update([{ score: 0.8, box: box(20, 0, 120, 100) }]);
		expect(second?.id).toBe(first?.id);
		expect(second?.box.x1).toBeCloseTo(10);
		expect(second?.age).toBe(1);
	});

	it('starts a new track for a non-overlapping detection and expires missing ones', () => {
		const tracker = new RegionTracker({ maxMisses: 1 });
		tracker.update([{ score: 0.9, box: box(0, 0, 100, 100) }]);
		const tracks = tracker.update([{ score: 0.9, box: box(500, 500, 600, 600) }]);
		expect(tracks).toHaveLength(2);
		expect(tracks[0]?.misses).toBe(1);
		const later = tracker.update([{ score: 0.9, box: box(500, 500, 600, 600) }]);
		expect(later).toHaveLength(1);
		expect(later[0]?.box.x1).toBe(500);
	});

	it('assigns the best overlap first', () => {
		const tracker = new RegionTracker({ boxSmoothing: 1 });
		tracker.update([
			{ score: 0.9, box: box(0, 0, 100, 100) },
			{ score: 0.9, box: box(90, 0, 200, 100) }
		]);
		const tracks = tracker.update([
			{ score: 0.9, box: box(95, 0, 205, 100) },
			{ score: 0.9, box: box(5, 0, 105, 100) }
		]);
		expect(tracks.map((t) => t.box.x1)).toEqual([5, 95]);
	});

	it('flags drift once the box moved away from where it was embedded', () => {
		const tracker = new RegionTracker({ boxSmoothing: 1 });
		const [track] = tracker.update([{ score: 0.9, box: box(0, 0, 100, 100) }]);
		expect(RegionTracker.drifted(track!)).toBe(true);
		track!.embeddedBox = { ...track!.box };
		expect(RegionTracker.drifted(track!)).toBe(false);
		tracker.update([{ score: 0.9, box: box(40, 0, 140, 100) }]);
		expect(track!.box.x1).toBe(40);
		expect(RegionTracker.drifted(track!)).toBe(true);
	});

	it('finds the largest track', () => {
		const tracker = new RegionTracker();
		const tracks = tracker.update([
			{ score: 0.9, box: box(0, 0, 10, 10) },
			{ score: 0.9, box: box(100, 100, 300, 300) }
		]);
		expect(RegionTracker.largest(tracks)?.box.x1).toBe(100);
		expect(RegionTracker.largest([])).toBeNull();
	});
});
