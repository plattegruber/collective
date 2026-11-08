export interface Box {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
}

/** Something the recognizer can read pixels from. */
export type PixelSource = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap;

/** A class-agnostic "there is a piece of art here" from the detector, in source pixels. */
export interface DetectedRegion {
	score: number;
	box: Box;
	/** True for a guessed region (the frame centre when the detector finds nothing). */
	synthetic?: boolean;
}

/** A region followed across frames and, once embedded, matched against the index. */
export interface TrackedRegion {
	id: number;
	box: Box;
	/** Latest detector score. */
	score: number;
	/** Frames since first seen. */
	age: number;
	/** Consecutive frames without a detection. */
	misses: number;
	/** Time of the last detection that continued this track. */
	lastSeenAt: number;
	/** The latest detection cleared the detector threshold (weak candidates need a strong match). */
	strong: boolean;
	/** The region was guessed rather than detected; drawn only once it matches. */
	synthetic: boolean;
	/** Best matching artwork, if the similarity cleared the match threshold. */
	artworkId: string | null;
	/** Smoothed cosine similarity of the current match (0 when unknown). */
	similarity: number;
	/** Last embedding of this region, used for memories and the "add this piece" flow. */
	embedding: Float32Array | null;
	lastEmbeddedAt: number;
	/** Box at the time of the last embedding, to decide when a re-embed is worth it. */
	embeddedBox: Box | null;
	/** Luminance thumbnail at the time of the last embedding; a change triggers a re-embed. */
	signature: Float32Array | null;
	/** True when the region sits inside a matched piece (a picture within the picture). */
	nested: boolean;
}

export interface Letterbox {
	/** Multiplier applied to the source region to fit inside the model input. */
	scale: number;
	/** Offsets of the scaled region inside the square input. */
	left: number;
	top: number;
	/** Size of the scaled region. */
	width: number;
	height: number;
	/** Model input edge length. */
	size: number;
}

export interface Match {
	artworkId: string;
	similarity: number;
	/** Second-best artwork, for margin checks and debugging. */
	runnerUp: { artworkId: string; similarity: number } | null;
}
