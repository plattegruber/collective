import type { ModelAsset } from '#lib/domain/recognition.js';
import { sha256Hex } from '#lib/utils/encoding.js';

const CACHE_NAME = 'collective-models-v1';

export interface DownloadProgress {
	/** Bytes received so far. */
	loaded: number;
	/** Expected total, from the manifest. */
	total: number;
	/** True when served from the local cache. */
	cached: boolean;
}

async function openCache(): Promise<Cache | null> {
	if (typeof caches === 'undefined') return null;
	try {
		return await caches.open(CACHE_NAME);
	} catch {
		return null;
	}
}

async function readWithProgress(
	response: Response,
	total: number,
	onProgress?: (progress: DownloadProgress) => void
): Promise<Uint8Array> {
	if (!response.body) return new Uint8Array(await response.arrayBuffer());
	const reader = response.body.getReader();
	const chunks: Uint8Array[] = [];
	let loaded = 0;
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		chunks.push(value);
		loaded += value.byteLength;
		onProgress?.({ loaded, total, cached: false });
	}
	const out = new Uint8Array(loaded);
	let offset = 0;
	for (const chunk of chunks) {
		out.set(chunk, offset);
		offset += chunk.byteLength;
	}
	return out;
}

/**
 * Downloads a model once and keeps it in the Cache API so later visits (and the scanner,
 * the upload page and the settings rebuild) share one copy. The SHA-256 from the manifest
 * is checked so a truncated or swapped download never reaches the runtime.
 */
export async function fetchModel(
	asset: ModelAsset,
	onProgress?: (progress: DownloadProgress) => void
): Promise<Uint8Array> {
	const cache = await openCache();
	const cached = await cache?.match(asset.url);
	if (cached) {
		const bytes = new Uint8Array(await cached.arrayBuffer());
		if ((await sha256Hex(bytes)) === asset.sha256) {
			onProgress?.({ loaded: bytes.byteLength, total: asset.bytes, cached: true });
			return bytes;
		}
		await cache?.delete(asset.url);
	}
	const response = await fetch(asset.url);
	if (!response.ok) throw new Error(`Could not download the model (${response.status}).`);
	const bytes = await readWithProgress(response, asset.bytes, onProgress);
	if ((await sha256Hex(bytes)) !== asset.sha256) {
		throw new Error('The downloaded model did not match its checksum. Try again.');
	}
	await cache
		?.put(
			asset.url,
			new Response(bytes.slice().buffer, {
				headers: { 'content-type': 'application/octet-stream' }
			})
		)
		.catch(() => undefined);
	return bytes;
}

/** Drops every cached model (used after a release change or from a debug control). */
export async function clearModelCache(): Promise<void> {
	if (typeof caches === 'undefined') return;
	await caches.delete(CACHE_NAME);
}
