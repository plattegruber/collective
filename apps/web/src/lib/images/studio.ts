/**
 * The photo studio: everything between "the collector picked a photo" and "a clean,
 * upright picture of the piece is ready to upload". Browser-only (canvas, the recognizer's
 * detector); the pure steps live in the sibling modules and are unit-tested there.
 */
import type { RecognizerManifest } from '#lib/domain/recognition.js';
import { chooseArtworkRegion } from '#lib/recognition/boxes.js';
import { loadRecognizer } from '#lib/recognition/loader.js';
import type { Box } from '#lib/recognition/types.js';
import { analyseColour, applyColourCorrection, type ColourCorrection } from './enhance.js';
import { fuseShots, type FuseResult } from './fuse.js';
import { boundsQuad, boxToQuad, rectifiedSize, transferQuad, type Quad } from './geometry.js';
import { FULL_MAX_EDGE, prepareBitmap, type PreparedImage } from './prepare.js';
import { findQuad, type QuadDetection } from './quad.js';
import type { Raster } from './raster.js';
import { warpPerspective } from './warp.js';

/** Long edge the studio works at; also the stored full-size edge. */
export const WORKING_EDGE = FULL_MAX_EDGE;
/** Long edge of interactive previews, so re-rendering after a drag stays instant. */
export const PREVIEW_EDGE = 1100;

export interface StudioShot {
	id: number;
	file: File;
	raster: Raster;
	/** Where the detector (or, failing that, the whole photo) says the piece is. */
	box: Box;
	detected: boolean;
	quad: Quad;
	/** Confidence of the automatic corners; 0 after the collector moved them. */
	confidence: number;
}

export interface StudioSettings {
	straighten: boolean;
	colour: boolean;
	/** Keep the frame (outer edges) or crop to the art (inner edges). */
	prefer: 'outer' | 'inner';
	/** Known width / height of the piece, used as the output aspect ratio. */
	aspect: number | null;
}

export const DEFAULT_STUDIO_SETTINGS: StudioSettings = {
	straighten: true,
	colour: true,
	prefer: 'outer',
	aspect: null
};

export interface StudioRender {
	raster: Raster;
	correction: ColourCorrection | null;
	fused: FuseResult | null;
}

/** Decodes a file to RGBA at most `maxEdge` on the long side, honouring EXIF orientation. */
export async function decodeToRaster(file: Blob, maxEdge = WORKING_EDGE): Promise<Raster> {
	const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
	try {
		const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
		const width = Math.max(1, Math.round(bitmap.width * scale));
		const height = Math.max(1, Math.round(bitmap.height * scale));
		const canvas = document.createElement('canvas');
		canvas.width = width;
		canvas.height = height;
		const context = canvas.getContext('2d', { willReadFrequently: true });
		if (!context) throw new Error('Canvas is not available.');
		context.imageSmoothingQuality = 'high';
		context.drawImage(bitmap, 0, 0, width, height);
		const image = context.getImageData(0, 0, width, height);
		return { data: image.data, width, height };
	} finally {
		bitmap.close();
	}
}

export function rasterToCanvas(raster: Raster): HTMLCanvasElement {
	const canvas = document.createElement('canvas');
	canvas.width = raster.width;
	canvas.height = raster.height;
	const context = canvas.getContext('2d');
	if (!context) throw new Error('Canvas is not available.');
	context.putImageData(new ImageData(raster.data, raster.width, raster.height), 0, 0);
	return canvas;
}

/**
 * Where the piece is in the photo. With a recognizer release available the detector looks
 * for it (the same way indexing does); otherwise, or when nothing is found, the photo is
 * assumed to be the piece.
 */
export async function locatePiece(
	raster: Raster,
	manifest: RecognizerManifest | null
): Promise<{ box: Box; detected: boolean }> {
	const whole = { x1: 0, y1: 0, x2: raster.width, y2: raster.height };
	if (!manifest) return { box: whole, detected: false };
	try {
		const models = await loadRecognizer(manifest);
		const canvas = rasterToCanvas(raster);
		const regions = await models.detector.detect(canvas, manifest.config.thresholds.detect);
		const box = chooseArtworkRegion(regions, raster.width, raster.height);
		return box ? { box, detected: true } : { box: whole, detected: false };
	} catch (cause) {
		console.warn('Detector unavailable in the studio; using the whole photo.', cause);
		return { box: whole, detected: false };
	}
}

export function detectCorners(
	raster: Raster,
	box: Box,
	detected: boolean,
	prefer: StudioSettings['prefer']
): QuadDetection {
	// A photo the collector already cropped only has edges near its border.
	return findQuad(raster, box, detected ? { prefer } : { prefer, band: 0.08 });
}

export async function openShot(
	id: number,
	file: File,
	manifest: RecognizerManifest | null,
	prefer: StudioSettings['prefer'],
	reference: StudioShot | null = null
): Promise<StudioShot> {
	const raster = await decodeToRaster(file);
	const { box, detected } = await locatePiece(raster, manifest);
	if (reference) {
		// Extra shots must land on the same edges as the lead, or fusing them smears the frame:
		// start from the lead's corners and only snap within a narrow band.
		const predicted = transferQuad(
			reference.quad,
			boxToQuad(reference.box.x1, reference.box.y1, reference.box.x2, reference.box.y2),
			boxToQuad(box.x1, box.y1, box.x2, box.y2)
		);
		const bounds = boundsQuad(predicted);
		const snapped = findQuad(
			raster,
			{ x1: bounds[0].x, y1: bounds[0].y, x2: bounds[2].x, y2: bounds[2].y },
			{ prefer, band: 0.05 }
		);
		return {
			id,
			file,
			raster,
			box,
			detected,
			quad: snapped.confidence > 0 ? snapped.quad : predicted,
			confidence: snapped.confidence
		};
	}
	const corners = detectCorners(raster, box, detected, prefer);
	return { id, file, raster, box, detected, quad: corners.quad, confidence: corners.confidence };
}

/** Lets the event loop breathe between heavy synchronous steps so the UI can repaint. */
const breathe = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/**
 * Produces the picture: every shot straightened to one rectangle (or simply cropped to its
 * quad's bounds when straightening is off), the shots fused when there are several, then
 * colour-corrected. `maxEdge` bounds the output; previews pass something small.
 */
export async function renderStudio(
	shots: StudioShot[],
	settings: StudioSettings,
	maxEdge = WORKING_EDGE
): Promise<StudioRender> {
	if (shots.length === 0) throw new RangeError('No shots to render.');
	const lead = shots[0]!;
	const size = rectifiedSize(lead.quad, maxEdge, settings.straighten ? settings.aspect : null);
	const rectified: Raster[] = [];
	for (const shot of shots) {
		const quad = settings.straighten ? shot.quad : boundsQuad(shot.quad);
		rectified.push(warpPerspective(shot.raster, quad, size));
		await breathe();
	}
	const fused = rectified.length > 1 ? fuseShots(rectified) : null;
	let raster = fused ? fused.raster : rectified[0]!;
	let correction: ColourCorrection | null = null;
	if (settings.colour) {
		correction = analyseColour(raster);
		raster = applyColourCorrection(raster, correction);
	}
	return { raster, correction, fused };
}

/** Encodes a rendered picture for upload, keeping the lead shot's file as the original. */
export async function prepareRender(
	render: StudioRender,
	original: File | null
): Promise<PreparedImage> {
	return prepareBitmap(rasterToCanvas(render.raster), original);
}
