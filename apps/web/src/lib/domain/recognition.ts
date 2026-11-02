/**
 * Browser-safe recognition types shared by the Worker, the scanner and the ML tooling.
 * Keep this free of server-only imports.
 */

export const PROTOTYPE_SOURCES = ['photo', 'scan'] as const;
export type PrototypeSource = (typeof PROTOTYPE_SOURCES)[number];

/** A box in normalised image coordinates (0..1, origin top-left). */
export interface NormalizedBox {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
}

/** How the embedder output turns into one comparable vector. */
export type Pooling = 'cls' | 'mean' | 'cls+mean';

/** Everything the browser runtime needs beyond the model files themselves. */
export interface RecognizerConfig {
	/** Per-channel RGB mean/std applied to 0..1 pixels before the embedder. */
	mean: [number, number, number];
	std: [number, number, number];
	pooling: Pooling;
	/** Margin added around a detected box before embedding, as a fraction of the box size. */
	cropMargin: number;
	thresholds: {
		/** Minimum detector score for a region to be considered. */
		detect: number;
		/** Cosine similarity at which a region is called a match. */
		match: number;
		/** Similarity above which the match is shown immediately (no second confirmation). */
		strong: number;
		/** A scan memory is only kept when it is at least this far from existing prototypes. */
		memoryNovelty: number;
	};
	/** Maximum scan memories kept per artwork. */
	maxMemories: number;
	/** Attribution the UI shows for the embedder, e.g. "Built with DINOv3". */
	attribution: string;
}

export const DEFAULT_RECOGNIZER_CONFIG: RecognizerConfig = {
	mean: [0.485, 0.456, 0.406],
	std: [0.229, 0.224, 0.225],
	pooling: 'cls+mean',
	cropMargin: 0.06,
	thresholds: { detect: 0.45, match: 0.6, strong: 0.72, memoryNovelty: 0.92 },
	maxMemories: 8,
	attribution: 'Built with DINOv3'
};

/** One downloadable model file. */
export interface ModelAsset {
	url: string;
	sha256: string;
	bytes: number;
	inputSize: number;
}

/** What a scanner (or the upload page) needs to run recognition. */
export interface RecognizerManifest {
	id: string;
	version: string;
	detector: ModelAsset;
	embedder: ModelAsset & { dim: number };
	config: RecognizerConfig;
}

/** The prototypes a scanner compares against, as the server sends them. */
export interface PrototypeIndexPayload {
	releaseId: string;
	dim: number;
	/** artworkId → base64url-encoded little-endian float32 vectors. */
	artworks: Record<string, string[]>;
}
