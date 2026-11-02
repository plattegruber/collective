import type { RecognizerConfig } from '#lib/domain/recognition.js';
import { containment } from './boxes.js';
import type { RecognizerModels } from './loader.js';
import type { PrototypeIndex } from './matcher.js';
import { RegionTracker } from './regionTracker.js';
import { DetectionTracker } from './tracker.js';
import type { PixelSource, TrackedRegion } from './types.js';

export interface RecognizerOptions {
	/** Re-embed a stable region at most this often. */
	embedIntervalMs: number;
	/** Exponential smoothing applied to a region's similarity (weight of the new value). */
	similaritySmoothing: number;
	/** Regions whose shorter side is below this fraction of the frame are ignored. */
	minRegionFraction: number;
}

export const DEFAULT_RECOGNIZER_OPTIONS: RecognizerOptions = {
	embedIntervalMs: 900,
	similaritySmoothing: 0.6,
	minRegionFraction: 0.08
};

export interface RecognitionFrame {
	regions: readonly TrackedRegion[];
	/** What the overlay should show, after hysteresis. */
	current: { artworkId: string; region: TrackedRegion | null; confidence: number } | null;
	changed: boolean;
	/** The largest region that is not in the collection, for the "add this piece" prompt. */
	unknown: TrackedRegion | null;
	timings: { detectMs: number; embedMs: number };
}

/** Maps cosine similarity onto 0..1 confidence centred on the match threshold. */
export function similarityToConfidence(
	similarity: number,
	thresholds: RecognizerConfig['thresholds']
): number {
	const width = Math.max(0.02, (thresholds.strong - thresholds.match) / 2);
	return 1 / (1 + Math.exp(-(similarity - thresholds.match) / width));
}

/**
 * Runs the two-stage pipeline on live frames: find artworks, follow them, embed each one
 * every so often, match against the collector's prototypes, and stabilise the answer.
 */
export class Recognizer {
	readonly #models: RecognizerModels;
	readonly #index: PrototypeIndex;
	readonly #options: RecognizerOptions;
	readonly #regions = new RegionTracker();
	readonly #decision: DetectionTracker;
	#busy = false;

	constructor(
		models: RecognizerModels,
		index: PrototypeIndex,
		options: Partial<RecognizerOptions> = {}
	) {
		this.#models = models;
		this.#index = index;
		this.#options = { ...DEFAULT_RECOGNIZER_OPTIONS, ...options };
		this.#decision = new DetectionTracker({ showAt: 0.7, hideAt: 0.45, switchMargin: 0.12 });
	}

	get index(): PrototypeIndex {
		return this.#index;
	}

	get models(): RecognizerModels {
		return this.#models;
	}

	reset(): void {
		this.#regions.reset();
		this.#decision.reset();
	}

	/** Processes one frame. Calls overlap-safe: a frame arriving while another runs is skipped. */
	async step(source: PixelSource, now = performance.now()): Promise<RecognitionFrame | null> {
		if (this.#busy) return null;
		this.#busy = true;
		try {
			return await this.#process(source, now);
		} finally {
			this.#busy = false;
		}
	}

	async #process(source: PixelSource, now: number): Promise<RecognitionFrame> {
		const { config } = this.#models.manifest;
		const t0 = performance.now();
		const detections = await this.#models.detector.detect(source, config.thresholds.detect);
		const detectMs = performance.now() - t0;
		const frameShort = Math.min(
			'videoWidth' in source ? source.videoWidth : source.width,
			'videoHeight' in source ? source.videoHeight : source.height
		);
		const minEdge = frameShort * this.#options.minRegionFraction;
		const tracks = this.#regions.update(
			detections.filter((d) => d.box.x2 - d.box.x1 >= minEdge && d.box.y2 - d.box.y1 >= minEdge)
		);

		let embedMs = 0;
		const due = tracks.filter(
			(track) =>
				track.misses === 0 &&
				(track.embedding === null ||
					now - track.lastEmbeddedAt >= this.#options.embedIntervalMs ||
					RegionTracker.drifted(track))
		);
		// One embedding per frame keeps the loop responsive; the rest catch up on later frames.
		const next = due.sort((a, b) => a.lastEmbeddedAt - b.lastEmbeddedAt)[0];
		if (next) {
			const t1 = performance.now();
			const vector = await this.#models.embedder.embed(source, next.box);
			embedMs = performance.now() - t1;
			next.embedding = vector;
			next.lastEmbeddedAt = now;
			next.embeddedBox = { ...next.box };
			const match = this.#index.match(vector);
			const similarity = match?.similarity ?? 0;
			const smoothing = next.artworkId ? this.#options.similaritySmoothing : 1;
			if (match && (next.artworkId === null || next.artworkId === match.artworkId)) {
				next.similarity = next.similarity + (similarity - next.similarity) * smoothing;
			} else {
				next.similarity = similarity;
			}
			next.artworkId = match && next.similarity >= config.thresholds.match ? match.artworkId : null;
		}

		for (const track of tracks) {
			track.nested = tracks.some(
				(other) =>
					other !== track &&
					other.artworkId !== null &&
					other.misses === 0 &&
					containment(track.box, other.box) >= 0.8
			);
		}
		const evidence = tracks
			.filter((track) => track.artworkId && track.misses === 0)
			.map((track) => ({
				artworkId: track.artworkId!,
				score: similarityToConfidence(track.similarity, config.thresholds)
			}));
		const update = this.#decision.update(evidence);
		const currentRegion = update.current
			? (tracks.find((t) => t.artworkId === update.current && t.misses === 0) ?? null)
			: null;
		const unknown =
			RegionTracker.largest(
				tracks.filter(
					(track) =>
						track.embedding !== null &&
						track.artworkId === null &&
						track.misses === 0 &&
						!track.nested
				)
			) ?? null;
		return {
			regions: tracks,
			current: update.current
				? { artworkId: update.current, region: currentRegion, confidence: update.confidence }
				: null,
			changed: update.changed,
			unknown,
			timings: { detectMs, embedMs }
		};
	}
}
