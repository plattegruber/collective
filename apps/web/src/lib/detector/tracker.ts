export interface TrackerOptions {
	/** Exponential smoothing weight for new evidence (0–1). Higher reacts faster. */
	smoothing: number;
	/** Per-frame multiplier applied to labels that were not seen this frame. */
	decay: number;
	/** Confidence required to show an artwork when none is showing. */
	showAt: number;
	/** Confidence below which the current artwork is hidden. */
	hideAt: number;
	/** A challenger must beat the current artwork by this margin to replace it. */
	switchMargin: number;
	/** Labels whose smoothed confidence falls below this are forgotten. */
	forgetBelow: number;
}

export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
	smoothing: 0.55,
	decay: 0.85,
	showAt: 0.78,
	hideAt: 0.6,
	switchMargin: 0.1,
	forgetBelow: 0.05
};

export interface TrackerUpdate {
	/** The artwork that should be displayed after this frame, if any. */
	current: string | null;
	/** True when `current` differs from the previous frame. */
	changed: boolean;
	/** Smoothed confidence of `current`. */
	confidence: number;
}

/**
 * Turns noisy per-frame detections into a stable "what is in front of the camera" answer.
 *
 * Each label keeps an exponentially smoothed confidence; labels missing from a frame decay.
 * Showing, hiding, and switching use hysteresis so the overlay does not flicker when the
 * detector hovers around a threshold.
 */
export class DetectionTracker {
	readonly #options: TrackerOptions;
	readonly #confidence = new Map<string, number>();
	#current: string | null = null;

	constructor(options: Partial<TrackerOptions> = {}) {
		this.#options = { ...DEFAULT_TRACKER_OPTIONS, ...options };
	}

	get current(): string | null {
		return this.#current;
	}

	confidenceOf(label: string): number {
		return this.#confidence.get(label) ?? 0;
	}

	reset(): void {
		this.#confidence.clear();
		this.#current = null;
	}

	update(detections: ReadonlyArray<{ artworkId: string; score: number }>): TrackerUpdate {
		const { smoothing, decay, showAt, hideAt, switchMargin, forgetBelow } = this.#options;

		const best = new Map<string, number>();
		for (const { artworkId, score } of detections) {
			if (score > (best.get(artworkId) ?? 0)) best.set(artworkId, score);
		}

		for (const [label, previous] of this.#confidence) {
			const observed = best.get(label);
			const next =
				observed === undefined ? previous * decay : previous + (observed - previous) * smoothing;
			if (next < forgetBelow) this.#confidence.delete(label);
			else this.#confidence.set(label, next);
		}
		for (const [label, score] of best) {
			if (!this.#confidence.has(label)) this.#confidence.set(label, score * smoothing);
		}

		const previous = this.#current;
		let leader: string | null = null;
		let leaderConfidence = 0;
		for (const [label, confidence] of this.#confidence) {
			if (confidence > leaderConfidence) {
				leader = label;
				leaderConfidence = confidence;
			}
		}

		const currentConfidence = previous ? this.confidenceOf(previous) : 0;
		if (previous && currentConfidence < hideAt) this.#current = null;

		if (leader) {
			if (!this.#current) {
				if (leaderConfidence >= showAt) this.#current = leader;
			} else if (
				leader !== this.#current &&
				leaderConfidence >= showAt &&
				leaderConfidence >= currentConfidence + switchMargin
			) {
				this.#current = leader;
			}
		}

		return {
			current: this.#current,
			changed: this.#current !== previous,
			confidence: this.#current ? this.confidenceOf(this.#current) : 0
		};
	}
}
