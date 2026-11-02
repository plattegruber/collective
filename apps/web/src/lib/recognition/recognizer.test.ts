import { describe, expect, it } from 'vitest';
import { similarityToConfidence } from './recognizer.js';

describe('similarityToConfidence', () => {
	const thresholds = { detect: 0.4, match: 0.6, strong: 0.72, memoryNovelty: 0.9 };
	it('is 0.5 at the match threshold and saturates above strong', () => {
		expect(similarityToConfidence(0.6, thresholds)).toBeCloseTo(0.5);
		expect(similarityToConfidence(0.72, thresholds)).toBeGreaterThan(0.85);
		expect(similarityToConfidence(0.3, thresholds)).toBeLessThan(0.01);
	});
});
