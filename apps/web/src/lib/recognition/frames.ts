import { computeLetterbox } from './letterbox.js';
import type { Box, Letterbox, PixelSource } from './types.js';

export function sourceSize(source: PixelSource): { width: number; height: number } {
	if (source instanceof HTMLVideoElement)
		return { width: source.videoWidth, height: source.videoHeight };
	if (source instanceof HTMLImageElement)
		return { width: source.naturalWidth, height: source.naturalHeight };
	return { width: source.width, height: source.height };
}

export function isReady(source: PixelSource): boolean {
	if (source instanceof HTMLVideoElement) return source.readyState >= 2 && source.videoWidth > 0;
	if (source instanceof HTMLImageElement) return source.complete && source.naturalWidth > 0;
	return true;
}

/**
 * A reusable square canvas that letterboxes a source (or a region of it) into the model's
 * input size and exposes the pixels. One per model, so the scanner never allocates per frame.
 */
export class FrameBuffer {
	readonly size: number;
	readonly #canvas: HTMLCanvasElement;
	readonly #context: CanvasRenderingContext2D;
	readonly #fill: string;

	constructor(size: number, fill = '#000') {
		this.size = size;
		this.#fill = fill;
		this.#canvas = document.createElement('canvas');
		this.#canvas.width = size;
		this.#canvas.height = size;
		const context = this.#canvas.getContext('2d', { willReadFrequently: true });
		if (!context) throw new Error('2D canvas is not available.');
		this.#context = context;
	}

	/** Draws `region` of the source (whole source by default) letterboxed; returns the mapping and pixels. */
	draw(source: PixelSource, region?: Box): { letterbox: Letterbox; rgba: Uint8ClampedArray } {
		const full = sourceSize(source);
		const box = region ?? { x1: 0, y1: 0, x2: full.width, y2: full.height };
		const width = Math.max(1, box.x2 - box.x1);
		const height = Math.max(1, box.y2 - box.y1);
		const letterbox = computeLetterbox(width, height, this.size);
		const context = this.#context;
		context.fillStyle = this.#fill;
		context.fillRect(0, 0, this.size, this.size);
		context.imageSmoothingQuality = 'medium';
		context.drawImage(
			source,
			box.x1,
			box.y1,
			width,
			height,
			letterbox.left,
			letterbox.top,
			letterbox.width,
			letterbox.height
		);
		return { letterbox, rgba: context.getImageData(0, 0, this.size, this.size).data };
	}
}

/** Encodes a region of the source as a JPEG, e.g. to turn a scanned unknown piece into its first photo. */
export async function cropToBlob(
	source: PixelSource,
	region: Box,
	maxEdge = 1600,
	quality = 0.9
): Promise<Blob> {
	const width = Math.max(1, Math.round(region.x2 - region.x1));
	const height = Math.max(1, Math.round(region.y2 - region.y1));
	const scale = Math.min(1, maxEdge / Math.max(width, height));
	const canvas = document.createElement('canvas');
	canvas.width = Math.max(1, Math.round(width * scale));
	canvas.height = Math.max(1, Math.round(height * scale));
	const context = canvas.getContext('2d');
	if (!context) throw new Error('2D canvas is not available.');
	context.imageSmoothingQuality = 'high';
	context.drawImage(source, region.x1, region.y1, width, height, 0, 0, canvas.width, canvas.height);
	const blob = await new Promise<Blob | null>((resolve) =>
		canvas.toBlob(resolve, 'image/jpeg', quality)
	);
	if (!blob) throw new Error('Could not encode the crop.');
	return blob;
}
