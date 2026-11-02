import type { RecognizerManifest } from '#lib/domain/recognition.js';
import { bytesToBase64Url } from '#lib/utils/encoding.js';
import { chooseArtworkRegion, normalizeBox } from './boxes.js';
import { loadRecognizer, type LoadStage, type RecognizerModels } from './loader.js';
import type { Box, PixelSource } from './types.js';

export interface TaughtPrototype {
	/** Where the artwork was found, normalised; null when the whole photo was used. */
	box: Box | null;
	embedding: Float32Array;
}

/** Where the artwork sits in an uploaded photo; null means the whole photo. */
export async function locateArtwork(
	models: RecognizerModels,
	source: PixelSource
): Promise<Box | null> {
	const { config } = models.manifest;
	const regions = await models.detector.detect(source, config.thresholds.detect);
	const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
	const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
	return chooseArtworkRegion(regions, width, height);
}

export async function embedPhoto(
	models: RecognizerModels,
	source: PixelSource
): Promise<TaughtPrototype> {
	const box = await locateArtwork(models, source);
	const embedding = await models.embedder.embed(source, box ?? undefined);
	const width = 'naturalWidth' in source ? source.naturalWidth : source.width;
	const height = 'naturalHeight' in source ? source.naturalHeight : source.height;
	return { box: box ? normalizeBox(box, width, height) : null, embedding };
}

export function encodeForUpload(vector: Float32Array): string {
	const bytes = new Uint8Array(vector.length * 4);
	const view = new DataView(bytes.buffer);
	for (let i = 0; i < vector.length; i += 1) view.setFloat32(i * 4, vector[i]!, true);
	return bytesToBase64Url(bytes);
}

export async function loadImage(url: string): Promise<HTMLImageElement> {
	const image = new Image();
	image.decoding = 'async';
	image.src = url;
	await image.decode();
	return image;
}

async function postPrototype(
	artworkId: string,
	body: Record<string, unknown>
): Promise<{ saved: boolean }> {
	const response = await fetch(`/app/artworks/${encodeURIComponent(artworkId)}/prototypes`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!response.ok) {
		const payload = (await response.json().catch(() => null)) as { message?: string } | null;
		throw new Error(payload?.message ?? `Could not save the prototype (${response.status}).`);
	}
	return (await response.json()) as { saved: boolean };
}

/** Embeds one uploaded photo and stores the prototype. Returns the box it used. */
export async function teachPhoto(
	models: RecognizerModels,
	artworkId: string,
	imageId: string,
	source: PixelSource
): Promise<TaughtPrototype> {
	const taught = await embedPhoto(models, source);
	await postPrototype(artworkId, {
		releaseId: models.manifest.id,
		source: 'photo',
		imageId,
		box: taught.box,
		embedding: encodeForUpload(taught.embedding)
	});
	return taught;
}

/** Stores a view captured while scanning. The server decides whether it is novel enough to keep. */
export async function rememberScan(
	models: RecognizerModels,
	artworkId: string,
	embedding: Float32Array
): Promise<boolean> {
	const result = await postPrototype(artworkId, {
		releaseId: models.manifest.id,
		source: 'scan',
		embedding: encodeForUpload(embedding)
	});
	return result.saved;
}

export interface RebuildProgress {
	done: number;
	total: number;
	failed: number;
	stage?: LoadStage;
}

interface RecognizerStatusPayload {
	manifest: RecognizerManifest | null;
	status: {
		images: number;
		indexed: number;
		memories: number;
		pending: { artworkId: string; imageId: string; url: string }[];
	} | null;
}

/**
 * Embeds every photo that has no prototype for the active release. Runs after a release
 * changes, when the scanner notices gaps, or from the settings page.
 */
export async function rebuildIndex(
	onProgress?: (progress: RebuildProgress) => void,
	signal?: AbortSignal,
	onTaught?: (artworkId: string, embedding: Float32Array) => void
): Promise<RebuildProgress> {
	const response = await fetch('/app/recognizer');
	if (!response.ok) throw new Error(`Could not read the recognizer status (${response.status}).`);
	const { manifest, status } = (await response.json()) as RecognizerStatusPayload;
	const progress: RebuildProgress = { done: 0, total: status?.pending.length ?? 0, failed: 0 };
	if (!manifest || !status || status.pending.length === 0) return progress;
	const models = await loadRecognizer(manifest, (stage) => onProgress?.({ ...progress, stage }));
	for (const item of status.pending) {
		if (signal?.aborted) break;
		try {
			const image = await loadImage(item.url);
			const taught = await teachPhoto(models, item.artworkId, item.imageId, image);
			onTaught?.(item.artworkId, taught.embedding);
			progress.done += 1;
		} catch (cause) {
			console.warn('Could not index photo', item, cause);
			progress.failed += 1;
		}
		onProgress?.({ ...progress });
	}
	return progress;
}
