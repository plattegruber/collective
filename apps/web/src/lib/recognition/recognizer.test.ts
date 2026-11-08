import { describe, expect, it } from 'vitest';
import {
	acceptsTrackMatch,
	centreRegion,
	focusScore,
	prioritise,
	requiredSimilarity,
	similarityToConfidence
} from './recognizer.js';
import type { TrackedRegion } from './types.js';

const thresholds = {
	detect: 0.4,
	match: 0.6,
	strong: 0.72,
	margin: 0.08,
	matchReference: 7,
	matchSlope: 0.034,
	memoryNovelty: 0.9
};

const track = (overrides: Partial<TrackedRegion>): TrackedRegion => ({
	id: 1,
	box: { x1: 0, y1: 0, x2: 100, y2: 100 },
	score: 0.9,
	age: 0,
	misses: 0,
	lastSeenAt: 0,
	strong: true,
	synthetic: false,
	artworkId: null,
	similarity: 0,
	embedding: null,
	lastEmbeddedAt: -Infinity,
	embeddedBox: null,
	signature: null,
	nested: false,
	...overrides
});

describe('similarityToConfidence', () => {
	it('is 0.5 at the match threshold and saturates above strong', () => {
		expect(similarityToConfidence(0.6, thresholds)).toBeCloseTo(0.5);
		expect(similarityToConfidence(0.72, thresholds)).toBeGreaterThan(0.85);
		expect(similarityToConfidence(0.3, thresholds)).toBeLessThan(0.01);
	});
});

describe('centreRegion', () => {
	it('is centred and covers the requested fraction of each edge', () => {
		const box = centreRegion(1000, 500, 0.72);
		expect(box.x1).toBeCloseTo(140);
		expect(box.x2).toBeCloseTo(860);
		expect(box.y1).toBeCloseTo(70);
		expect(box.y2).toBeCloseTo(430);
	});
});

describe('focusScore', () => {
	it('prefers large, central regions', () => {
		const central = focusScore({ x1: 300, y1: 200, x2: 700, y2: 400 }, 1000, 600);
		const corner = focusScore({ x1: 0, y1: 0, x2: 400, y2: 200 }, 1000, 600);
		const small = focusScore({ x1: 450, y1: 250, x2: 550, y2: 350 }, 1000, 600);
		expect(central).toBeGreaterThan(corner);
		expect(central).toBeGreaterThan(small);
	});
});

describe('prioritise', () => {
	it('embeds new regions first, then strong ones, then what the camera points at', () => {
		const vector = new Float32Array(4);
		const stale = track({ id: 1, embedding: vector, lastEmbeddedAt: 0 });
		const fresh = track({ id: 2, box: { x1: 0, y1: 0, x2: 20, y2: 20 } });
		const weak = track({ id: 3, strong: false, box: { x1: 400, y1: 300, x2: 600, y2: 500 } });
		const centralStrong = track({ id: 4, box: { x1: 350, y1: 250, x2: 650, y2: 550 } });
		const order = prioritise([stale, weak, centralStrong, fresh], 1000, 800).map((t) => t.id);
		expect(order).toEqual([4, 2, 3, 1]);
	});
});

describe('acceptsTrackMatch', () => {
	const match = { artworkId: 'a', similarity: 0.66, runnerUp: null };

	it('accepts a match above the match threshold for a confident detection', () => {
		expect(requiredSimilarity(track({}), thresholds)).toBe(0.6);
		expect(acceptsTrackMatch(track({}), match, thresholds)).toBe(true);
	});

	it('demands a strong match for weak detections and centre guesses', () => {
		expect(requiredSimilarity(track({ strong: false }), thresholds)).toBe(0.72);
		expect(acceptsTrackMatch(track({ strong: false }), match, thresholds)).toBe(false);
		expect(acceptsTrackMatch(track({ synthetic: true }), match, thresholds)).toBe(false);
		expect(
			acceptsTrackMatch(track({ strong: false }), { ...match, similarity: 0.8 }, thresholds)
		).toBe(true);
	});

	it('still applies the runner-up margin', () => {
		const close = { ...match, similarity: 0.8, runnerUp: { artworkId: 'b', similarity: 0.75 } };
		expect(acceptsTrackMatch(track({}), close, thresholds)).toBe(false);
	});
});
