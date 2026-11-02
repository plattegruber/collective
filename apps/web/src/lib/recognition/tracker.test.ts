import { describe, expect, it } from 'vitest';
import { DetectionTracker } from './tracker.js';

const see = (tracker: DetectionTracker, artworkId: string, score: number, frames = 1) => {
	let last = tracker.update([]);
	for (let i = 0; i < frames; i += 1) last = tracker.update([{ artworkId, score }]);
	return last;
};

describe('DetectionTracker', () => {
	it('needs sustained evidence before showing an artwork', () => {
		const tracker = new DetectionTracker();
		expect(tracker.update([{ artworkId: 'a', score: 0.95 }]).current).toBeNull();
		const result = see(tracker, 'a', 0.95, 4);
		expect(result.current).toBe('a');
		expect(result.confidence).toBeGreaterThan(0.78);
	});

	it('reports a change exactly once when an artwork appears', () => {
		const tracker = new DetectionTracker();
		const changes = Array.from({ length: 6 }, () =>
			tracker.update([{ artworkId: 'a', score: 0.95 }])
		).filter((u) => u.changed);
		expect(changes).toHaveLength(1);
	});

	it('keeps showing through brief dropouts, then hides', () => {
		const tracker = new DetectionTracker();
		see(tracker, 'a', 0.95, 6);
		expect(tracker.update([]).current).toBe('a');
		let last = tracker.update([]);
		for (let i = 0; i < 12 && last.current; i += 1) last = tracker.update([]);
		expect(last.current).toBeNull();
	});

	it('does not switch to a challenger that is only marginally stronger', () => {
		const tracker = new DetectionTracker();
		see(tracker, 'a', 0.9, 6);
		const result = tracker.update([
			{ artworkId: 'a', score: 0.85 },
			{ artworkId: 'b', score: 0.86 }
		]);
		expect(result.current).toBe('a');
	});

	it('switches when a challenger clearly dominates', () => {
		const tracker = new DetectionTracker();
		see(tracker, 'a', 0.9, 6);
		let result = tracker.update([]);
		for (let i = 0; i < 8; i += 1) result = tracker.update([{ artworkId: 'b', score: 0.98 }]);
		expect(result.current).toBe('b');
	});

	it('forgets labels whose confidence decays away', () => {
		const tracker = new DetectionTracker();
		tracker.update([{ artworkId: 'a', score: 0.5 }]);
		for (let i = 0; i < 25; i += 1) tracker.update([]);
		expect(tracker.confidenceOf('a')).toBe(0);
	});

	it('uses the strongest score when a label appears several times in a frame', () => {
		const tracker = new DetectionTracker({ smoothing: 1 });
		tracker.update([
			{ artworkId: 'a', score: 0.2 },
			{ artworkId: 'a', score: 0.9 }
		]);
		expect(tracker.confidenceOf('a')).toBeCloseTo(0.9);
	});
});
