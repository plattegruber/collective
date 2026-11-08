export interface TrackerOptions {
	/** Time constant in ms for taking on new evidence; shorter reacts faster. */
	riseMs: number;
	/** Half-life in ms of a label's confidence while it is not seen. */
	halfLifeMs: number;
	/** Confidence required to show an artwork when none is showing. */
	showAt: number;
	/** Confidence below which the current artwork is hidden. */
	hideAt: number;
	/** A challenger must beat the current artwork by this margin to replace it. */
	switchMargin: number;
	/** Labels whose smoothed confidence falls below this are forgotten. */
	forgetBelow: number;
}

/**
 * Tuned so an answer appears within two or three frames of the first positive embedding
 * and disappears about 300 ms after the piece leaves the frame, whatever the frame rate.
 */
export const DEFAULT_TRACKER_OPTIONS: TrackerOptions = {
	riseMs: 120,
	halfLifeMs: 300,
	showAt: 0.7,
	hideAt: 0.5,
	switchMargin: 0.12,
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

/** Longest gap between frames that still counts as continuous observation. */
const MAX_STEP_MS = 1000;

/**
 * Turns noisy per-frame detections into a stable "what is in front of the camera" answer.
 *
 * Each label keeps an exponentially smoothed confidence; labels missing from a frame decay.
 * Smoothing and decay are defined in wall-clock time, so a phone running the pipeline at
 * three frames per second behaves like a desktop at thirty. Showing, hiding and switching
 * use hysteresis so the overlay does not flicker when a match hovers around a threshold.
 */
export class DetectionTracker {
	readonly #options: TrackerOptions;
	readonly #confidence = new Map<string, number>();
	#current: string | null = null;
	#lastUpdate: number | undefined;

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
		this.#lastUpdate = undefined;
	}

	update(
		detections: ReadonlyArray<{ artworkId: string; score: number }>,
		now = performance.now()
	): TrackerUpdate {
		const { riseMs, halfLifeMs, showAt, hideAt, switchMargin, forgetBelow } = this.#options;
		const elapsed =
			this.#lastUpdate === undefined
				? riseMs
				: Math.min(MAX_STEP_MS, Math.max(1, now - this.#lastUpdate));
		this.#lastUpdate = now;
		const gain = 1 - Math.exp(-elapsed / riseMs);
		const decay = Math.pow(0.5, elapsed / halfLifeMs);

		const best = new Map<string, number>();
		for (const { artworkId, score } of detections) {
			if (score > (best.get(artworkId) ?? 0)) best.set(artworkId, score);
		}

		for (const [label, previous] of this.#confidence) {
			const observed = best.get(label);
			const next =
				observed === undefined ? previous * decay : previous + (observed - previous) * gain;
			if (next < forgetBelow) this.#confidence.delete(label);
			else this.#confidence.set(label, next);
		}
		for (const [label, score] of best) {
			if (!this.#confidence.has(label)) this.#confidence.set(label, score * gain);
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
