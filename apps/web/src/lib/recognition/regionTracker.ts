import { boxIoU, lerpBox } from './boxes.js';
import type { Box, DetectedRegion, TrackedRegion } from './types.js';

export interface RegionTrackerOptions {
	/** Minimum IoU for a detection to continue an existing track. */
	matchIoU: number;
	/** How long a track survives without a detection, in ms. */
	maxMissMs: number;
	/** 0–1 weight of the new box when smoothing (1 = no smoothing). */
	boxSmoothing: number;
}

export const DEFAULT_REGION_TRACKER_OPTIONS: RegionTrackerOptions = {
	matchIoU: 0.3,
	maxMissMs: 450,
	boxSmoothing: 0.5
};

/**
 * Keeps identities for detected regions across frames so a region is embedded once in a
 * while rather than every frame, and so overlays do not jitter. Tracks expire by time, not
 * frame count, so a slow phone forgets a vanished piece as quickly as a fast laptop.
 */
export class RegionTracker {
	readonly #options: RegionTrackerOptions;
	#tracks: TrackedRegion[] = [];
	#nextId = 1;

	constructor(options: Partial<RegionTrackerOptions> = {}) {
		this.#options = { ...DEFAULT_REGION_TRACKER_OPTIONS, ...options };
	}

	get tracks(): readonly TrackedRegion[] {
		return this.#tracks;
	}

	reset(): void {
		this.#tracks = [];
	}

	/** Associates this frame's detections with tracks (greedy, best IoU first). */
	update(detections: readonly DetectedRegion[], now = performance.now()): readonly TrackedRegion[] {
		const { matchIoU, maxMissMs, boxSmoothing } = this.#options;
		const pairs: { track: TrackedRegion; detection: DetectedRegion; iou: number }[] = [];
		for (const track of this.#tracks) {
			for (const detection of detections) {
				const iou = boxIoU(track.box, detection.box);
				if (iou >= matchIoU) pairs.push({ track, detection, iou });
			}
		}
		pairs.sort((a, b) => b.iou - a.iou);

		const usedTracks = new Set<TrackedRegion>();
		const usedDetections = new Set<DetectedRegion>();
		for (const { track, detection } of pairs) {
			if (usedTracks.has(track) || usedDetections.has(detection)) continue;
			usedTracks.add(track);
			usedDetections.add(detection);
			track.box = lerpBox(track.box, detection.box, boxSmoothing);
			track.score = detection.score;
			track.synthetic = detection.synthetic ?? false;
			track.misses = 0;
			track.lastSeenAt = now;
			track.age += 1;
		}
		for (const track of this.#tracks) {
			if (!usedTracks.has(track)) {
				track.misses += 1;
				track.age += 1;
			}
		}
		for (const detection of detections) {
			if (usedDetections.has(detection)) continue;
			this.#tracks.push({
				id: this.#nextId++,
				box: { ...detection.box },
				score: detection.score,
				age: 0,
				misses: 0,
				lastSeenAt: now,
				strong: false,
				synthetic: detection.synthetic ?? false,
				artworkId: null,
				similarity: 0,
				embedding: null,
				lastEmbeddedAt: -Infinity,
				embeddedBox: null,
				signature: null,
				nested: false
			});
		}
		this.#tracks = this.#tracks.filter((track) => now - track.lastSeenAt <= maxMissMs);
		return this.#tracks;
	}

	/** True when the track moved or resized enough since its last embedding to be worth another look. */
	static drifted(track: TrackedRegion, threshold = 0.6): boolean {
		return !track.embeddedBox || boxIoU(track.embeddedBox, track.box) < threshold;
	}

	static largest(tracks: readonly TrackedRegion[]): TrackedRegion | null {
		let best: TrackedRegion | null = null;
		let bestArea = 0;
		for (const track of tracks) {
			const area = (track.box.x2 - track.box.x1) * (track.box.y2 - track.box.y1);
			if (area > bestArea) {
				best = track;
				bestArea = area;
			}
		}
		return best;
	}

	static boxOf(track: TrackedRegion): Box {
		return track.box;
	}
}
