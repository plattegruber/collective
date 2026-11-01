import type { Letterbox } from './types.js';

/** Fits a `width × height` frame inside a `size × size` square, centred, preserving aspect ratio. */
export function computeLetterbox(width: number, height: number, size: number): Letterbox {
	if (width <= 0 || height <= 0 || size <= 0) throw new RangeError('Dimensions must be positive.');
	const scale = Math.min(size / width, size / height);
	const scaledWidth = Math.floor(width * scale);
	const scaledHeight = Math.floor(height * scale);
	return {
		scale,
		left: Math.floor((size - scaledWidth) / 2),
		top: Math.floor((size - scaledHeight) / 2),
		width: scaledWidth,
		height: scaledHeight,
		size
	};
}

/** Maps a box in model-input pixels back to source-frame pixels, clamped to the frame. */
export function unletterboxBox(
	box: { x1: number; y1: number; x2: number; y2: number },
	letterbox: Letterbox,
	frameWidth: number,
	frameHeight: number
): { x1: number; y1: number; x2: number; y2: number } {
	const clampX = (value: number) => Math.min(frameWidth, Math.max(0, value));
	const clampY = (value: number) => Math.min(frameHeight, Math.max(0, value));
	return {
		x1: clampX((box.x1 - letterbox.left) / letterbox.scale),
		y1: clampY((box.y1 - letterbox.top) / letterbox.scale),
		x2: clampX((box.x2 - letterbox.left) / letterbox.scale),
		y2: clampY((box.y2 - letterbox.top) / letterbox.scale)
	};
}
