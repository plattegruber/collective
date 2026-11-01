/** What a scanner needs to load a collector's model and map classes back to artworks. */
export interface DetectorManifest {
	id: string;
	version: string;
	format: 'onnx';
	url: string;
	sha256: string;
	inputSize: number;
	/** Class index (as a string, starting at "1") → artwork id. */
	labels: Record<string, string>;
}

export interface Detection {
	/** Artwork id resolved through the manifest labels. */
	artworkId: string;
	score: number;
	/** Box in source-video pixel coordinates. */
	box: { x1: number; y1: number; x2: number; y2: number };
}

export interface Letterbox {
	/** Multiplier applied to the source frame to fit inside the model input. */
	scale: number;
	/** Offsets of the scaled frame inside the square input. */
	left: number;
	top: number;
	/** Size of the scaled frame. */
	width: number;
	height: number;
	/** Model input edge length. */
	size: number;
}
