import type { RecognizerConfig } from '#lib/domain/recognition.js';
import { boxArea, containment } from './boxes.js';
import { FrameBuffer } from './frames.js';
import type { RecognizerModels } from './loader.js';
import { acceptsMatch, effectiveThresholds, type PrototypeIndex } from './matcher.js';
import { RegionTracker } from './regionTracker.js';
import { luminanceSignature, SIGNATURE_SIZE, signatureDistance } from './signature.js';
import { DetectionTracker } from './tracker.js';
import type { Box, DetectedRegion, Match, PixelSource, TrackedRegion } from './types.js';

export interface RecognizerOptions {
	/** Re-embed a matched, stable region at most this often. */
	embedIntervalMs: number;
	/** Re-check an unmatched region this often; lighting and blur change from frame to frame. */
	retryIntervalMs: number;
	/** Regions whose shorter side is below this fraction of the frame are ignored. */
	minRegionFraction: number;
	/** Detections scoring at least `detect × ratio` are kept as weak candidates. */
	weakThresholdRatio: number;
	/** With nothing detected for this long, the frame centre is embedded as a guess. */
	fallbackAfterMs: number;
	/** Width and height of the centre guess as a fraction of the frame. */
	fallbackFraction: number;
	/** Mean luminance change (0–1) of a region's thumbnail that counts as new content. */
	changeThreshold: number;
}

export const DEFAULT_RECOGNIZER_OPTIONS: RecognizerOptions = {
	embedIntervalMs: 900,
	retryIntervalMs: 350,
	minRegionFraction: 0.08,
	weakThresholdRatio: 0.5,
	fallbackAfterMs: 600,
	fallbackFraction: 0.72,
	changeThreshold: 0.1
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

/** The region to try when the detector finds nothing: the collector is pointing at it. */
export function centreRegion(width: number, height: number, fraction: number): Box {
	const dx = (width * (1 - fraction)) / 2;
	const dy = (height * (1 - fraction)) / 2;
	return { x1: dx, y1: dy, x2: width - dx, y2: height - dy };
}

/** How much a region looks like the thing the camera is aimed at: big and central scores high. */
export function focusScore(box: Box, width: number, height: number): number {
	const area = boxArea(box) / Math.max(1, width * height);
	const cx = (box.x1 + box.x2) / 2 / Math.max(1, width) - 0.5;
	const cy = (box.y1 + box.y2) / 2 / Math.max(1, height) - 0.5;
	const offCentre = Math.min(1, Math.hypot(cx, cy) / Math.SQRT1_2);
	return area * (1 - 0.5 * offCentre);
}

/**
 * Orders regions worth embedding this frame: never-embedded first, then confident
 * detections before weak ones, then whatever the camera is aimed at, then the stalest.
 */
export function prioritise(
	due: readonly TrackedRegion[],
	width: number,
	height: number
): TrackedRegion[] {
	return [...due].sort((a, b) => {
		const fresh = Number(a.embedding !== null) - Number(b.embedding !== null);
		if (fresh !== 0) return fresh;
		const strength = Number(b.strong) - Number(a.strong);
		if (strength !== 0) return strength;
		const focus = focusScore(b.box, width, height) - focusScore(a.box, width, height);
		if (Math.abs(focus) > 1e-6) return focus;
		return a.lastEmbeddedAt - b.lastEmbeddedAt;
	});
}

/**
 * A weak or guessed region is accepted only on a strong match: it is cheap insurance for
 * dim rooms, not a licence to label every poster-shaped thing in view.
 */
export function requiredSimilarity(
	track: Pick<TrackedRegion, 'strong' | 'synthetic'>,
	thresholds: { match: number; strong: number }
): number {
	return track.strong && !track.synthetic ? thresholds.match : thresholds.strong;
}

export function acceptsTrackMatch(
	track: Pick<TrackedRegion, 'strong' | 'synthetic'>,
	match: Match | null,
	thresholds: { match: number; strong: number; margin: number }
): match is Match {
	return (
		acceptsMatch(match, thresholds) && match.similarity >= requiredSimilarity(track, thresholds)
	);
}

function frameSize(source: PixelSource): { width: number; height: number } {
	return {
		width: 'videoWidth' in source ? source.videoWidth : source.width,
		height: 'videoHeight' in source ? source.videoHeight : source.height
	};
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
	readonly #decision = new DetectionTracker();
	#thumbs: FrameBuffer | undefined;
	#busy = false;
	#lastDetectionAt: number | undefined;

	constructor(
		models: RecognizerModels,
		index: PrototypeIndex,
		options: Partial<RecognizerOptions> = {}
	) {
		this.#models = models;
		this.#index = index;
		this.#options = { ...DEFAULT_RECOGNIZER_OPTIONS, ...options };
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
		this.#lastDetectionAt = undefined;
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

	#signature(source: PixelSource, box: Box): Float32Array {
		this.#thumbs ??= new FrameBuffer(SIGNATURE_SIZE);
		return luminanceSignature(this.#thumbs.draw(source, box).rgba);
	}

	/** True when the region's content no longer resembles what was last embedded. */
	#changed(source: PixelSource, track: TrackedRegion): boolean {
		if (!track.signature) return false;
		const distance = signatureDistance(track.signature, this.#signature(source, track.box));
		return distance >= this.#options.changeThreshold;
	}

	#candidates(detections: DetectedRegion[], width: number, height: number, now: number) {
		const minEdge = Math.min(width, height) * this.#options.minRegionFraction;
		const kept = detections.filter(
			(d) => d.box.x2 - d.box.x1 >= minEdge && d.box.y2 - d.box.y1 >= minEdge
		);
		if (kept.length > 0 || this.#lastDetectionAt === undefined) this.#lastDetectionAt = now;
		else if (now - this.#lastDetectionAt >= this.#options.fallbackAfterMs) {
			kept.push({
				score: 0,
				box: centreRegion(width, height, this.#options.fallbackFraction),
				synthetic: true
			});
		}
		return kept;
	}

	async #process(source: PixelSource, now: number): Promise<RecognitionFrame> {
		const { config } = this.#models.manifest;
		const thresholds = effectiveThresholds(config.thresholds, this.#index.artworkIds.length);
		const { width, height } = frameSize(source);
		const t0 = performance.now();
		const detections = await this.#models.detector.detect(
			source,
			config.thresholds.detect * this.#options.weakThresholdRatio
		);
		const detectMs = performance.now() - t0;
		const tracks = this.#regions.update(this.#candidates(detections, width, height, now), now);
		for (const track of tracks) {
			if (track.misses === 0)
				track.strong = !track.synthetic && track.score >= config.thresholds.detect;
		}

		let embedMs = 0;
		const due = tracks.filter((track) => {
			if (track.misses !== 0) return false;
			if (track.embedding === null || RegionTracker.drifted(track)) return true;
			if (this.#changed(source, track)) return true;
			const interval = track.artworkId
				? this.#options.embedIntervalMs
				: this.#options.retryIntervalMs;
			return now - track.lastEmbeddedAt >= interval;
		});
		// One embedding per frame keeps the loop responsive; the rest catch up on later frames.
		const next = prioritise(due, width, height)[0];
		const contradicted: { artworkId: string; score: number }[] = [];
		if (next) {
			const t1 = performance.now();
			const vector = await this.#models.embedder.embed(source, next.box);
			embedMs = performance.now() - t1;
			next.embedding = vector;
			next.lastEmbeddedAt = now;
			next.embeddedBox = { ...next.box };
			next.signature = this.#signature(source, next.box);
			const match = this.#index.match(vector);
			const similarity = match?.similarity ?? 0;
			const previous = next.artworkId;
			if (match && previous === match.artworkId) {
				next.similarity = next.similarity + (similarity - next.similarity) * 0.6;
			} else {
				next.similarity = similarity;
			}
			next.artworkId =
				acceptsTrackMatch(next, match, thresholds) &&
				next.similarity >= requiredSimilarity(next, thresholds)
					? match.artworkId
					: null;
			// The region that carried the previous answer now shows something else: that is
			// direct evidence against it, worth far more than letting its confidence decay.
			if (previous && previous !== next.artworkId) {
				contradicted.push({
					artworkId: previous,
					score: similarityToConfidence(this.#index.similarityTo(previous, vector), thresholds)
				});
			}
		}

		for (const track of tracks) {
			track.nested = tracks.some(
				(other) =>
					other !== track &&
					other.artworkId !== null &&
					!other.synthetic &&
					other.misses === 0 &&
					containment(track.box, other.box) >= 0.8
			);
		}
		const evidence = tracks
			.filter((track) => track.artworkId && track.misses === 0)
			.map((track) => ({
				artworkId: track.artworkId!,
				score: similarityToConfidence(track.similarity, thresholds)
			}));
		for (const item of contradicted) {
			if (!evidence.some((e) => e.artworkId === item.artworkId)) evidence.push(item);
		}
		const update = this.#decision.update(evidence, now);
		const currentRegion = update.current
			? (tracks.find((t) => t.artworkId === update.current && t.misses === 0) ?? null)
			: null;
		const unknown =
			RegionTracker.largest(
				tracks.filter(
					(track) =>
						track.strong &&
						!track.synthetic &&
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
