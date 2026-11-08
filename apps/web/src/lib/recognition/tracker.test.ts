import { describe, expect, it } from 'vitest';
import { DetectionTracker } from './tracker.js';

/** Feeds `frames` observations of one artwork, `stepMs` apart, starting at `start`. */
const see = (
	tracker: DetectionTracker,
	artworkId: string,
	score: number,
	frames: number,
	start = 0,
	stepMs = 100
) => {
	let last = tracker.update([], start);
	for (let i = 1; i <= frames; i += 1)
		last = tracker.update([{ artworkId, score }], start + i * stepMs);
	return { last, at: start + frames * stepMs };
};

describe('DetectionTracker', () => {
	it('needs a second frame of evidence before showing an artwork', () => {
		const tracker = new DetectionTracker();
		expect(tracker.update([{ artworkId: 'a', score: 0.95 }], 0).current).toBeNull();
		const { last } = see(tracker, 'a', 0.95, 4, 0);
		expect(last.current).toBe('a');
		expect(last.confidence).toBeGreaterThan(0.78);
	});

	it('reports a change exactly once when an artwork appears', () => {
		const tracker = new DetectionTracker();
		const changes = Array.from({ length: 6 }, (_, i) =>
			tracker.update([{ artworkId: 'a', score: 0.95 }], i * 100)
		).filter((u) => u.changed);
		expect(changes).toHaveLength(1);
	});

	it('keeps showing through a brief dropout, then hides', () => {
		const tracker = new DetectionTracker();
		const { at } = see(tracker, 'a', 0.95, 6);
		expect(tracker.update([], at + 100).current).toBe('a');
		let last = tracker.update([], at + 200);
		for (let i = 3; i < 12 && last.current; i += 1) last = tracker.update([], at + i * 100);
		expect(last.current).toBeNull();
	});

	it.each([
		['30 fps', 33],
		['3 fps', 333]
	])('hides within 400 ms of the piece leaving the frame at %s', (_, stepMs) => {
		const tracker = new DetectionTracker();
		const { at } = see(tracker, 'a', 0.95, 10, 0, stepMs);
		let now = at;
		let current: string | null = 'a';
		while (current && now - at < 2000) {
			now += stepMs;
			current = tracker.update([], now).current;
		}
		expect(current).toBeNull();
		expect(now - at).toBeLessThanOrEqual(400);
	});

	it('shows a new piece within 300 ms of its first evidence at any frame rate', () => {
		for (const stepMs of [33, 333]) {
			const tracker = new DetectionTracker();
			tracker.update([], 0);
			let now = 0;
			let current: string | null = null;
			while (!current && now < 2000) {
				now += stepMs;
				current = tracker.update([{ artworkId: 'b', score: 0.97 }], now).current;
			}
			expect(current).toBe('b');
			expect(now).toBeLessThanOrEqual(stepMs > 300 ? 2 * stepMs : 300);
		}
	});

	it('does not switch to a challenger that is only marginally stronger', () => {
		const tracker = new DetectionTracker();
		const { at } = see(tracker, 'a', 0.9, 6);
		const result = tracker.update(
			[
				{ artworkId: 'a', score: 0.85 },
				{ artworkId: 'b', score: 0.86 }
			],
			at + 100
		);
		expect(result.current).toBe('a');
	});

	it('switches when a challenger clearly dominates', () => {
		const tracker = new DetectionTracker();
		const { at } = see(tracker, 'a', 0.9, 6);
		let result = tracker.update([], at + 100);
		for (let i = 2; i < 10; i += 1)
			result = tracker.update([{ artworkId: 'b', score: 0.98 }], at + i * 100);
		expect(result.current).toBe('b');
	});

	it('forgets labels whose confidence decays away', () => {
		const tracker = new DetectionTracker();
		tracker.update([{ artworkId: 'a', score: 0.5 }], 0);
		for (let i = 1; i <= 25; i += 1) tracker.update([], i * 100);
		expect(tracker.confidenceOf('a')).toBe(0);
	});

	it('uses the strongest score when a label appears several times in a frame', () => {
		const tracker = new DetectionTracker({ riseMs: 1 });
		tracker.update(
			[
				{ artworkId: 'a', score: 0.2 },
				{ artworkId: 'a', score: 0.9 }
			],
			0
		);
		expect(tracker.confidenceOf('a')).toBeCloseTo(0.9 * (1 - Math.exp(-1)), 5);
	});

	it('treats a long pause as a single step rather than instant forgetting', () => {
		const tracker = new DetectionTracker();
		const { at } = see(tracker, 'a', 0.95, 6);
		const before = tracker.confidenceOf('a');
		tracker.update([], at + 60_000);
		expect(tracker.confidenceOf('a')).toBeGreaterThan(before * Math.pow(0.5, 1000 / 300) - 1e-9);
	});
});
