/**
 * Client-side image preparation. Photos from phones are often 10+ MB; we downscale before
 * upload so the Worker never has to decode pixels (it cannot) and R2 stays lean. The camera
 * original travels alongside untouched, so nothing is lost to the resize and later tools
 * (re-processing, 3D) can start from the real pixels.
 */

export const FULL_MAX_EDGE = 3200;
export const THUMB_MAX_EDGE = 480;
export const FULL_QUALITY = 0.9;
export const THUMB_QUALITY = 0.82;
/** Originals above this are not kept; the Worker refuses them anyway. */
export const ORIGINAL_MAX_BYTES = 30 * 1024 * 1024;

export interface PreparedImage {
	full: Blob;
	thumb: Blob;
	/** The upload as shot, when it is worth keeping (a real file of a known type). */
	original: Blob | null;
	width: number;
	height: number;
}

export function fit(
	width: number,
	height: number,
	maxEdge: number
): { width: number; height: number } {
	const scale = Math.min(1, maxEdge / Math.max(width, height));
	return {
		width: Math.max(1, Math.round(width * scale)),
		height: Math.max(1, Math.round(height * scale))
	};
}

export async function encodeCanvasSource(
	source: CanvasImageSource,
	size: { width: number; height: number },
	type: string,
	quality: number
): Promise<Blob> {
	const canvas = document.createElement('canvas');
	canvas.width = size.width;
	canvas.height = size.height;
	const context = canvas.getContext('2d', { alpha: type === 'image/png' });
	if (!context) throw new Error('Canvas is not available.');
	if (type !== 'image/png') {
		context.fillStyle = '#ffffff';
		context.fillRect(0, 0, size.width, size.height);
	}
	context.imageSmoothingQuality = 'high';
	context.drawImage(source, 0, 0, size.width, size.height);
	const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, type, quality));
	if (!blob) throw new Error('Could not encode image.');
	return blob;
}

const KEEPABLE_ORIGINALS = new Set([
	'image/jpeg',
	'image/png',
	'image/webp',
	'image/heic',
	'image/heif'
]);

/** Whether an upload is worth storing as shot: a real file of a type the server accepts. */
export function keepOriginal(file: Blob | null | undefined): file is Blob {
	return Boolean(
		file && file.size > 0 && file.size <= ORIGINAL_MAX_BYTES && KEEPABLE_ORIGINALS.has(file.type)
	);
}

/** Full-size and thumbnail derivatives of a bitmap, plus the original when one is given. */
export async function prepareBitmap(
	bitmap: ImageBitmap | HTMLCanvasElement,
	original: Blob | null,
	keepPng = false
): Promise<PreparedImage> {
	const fullSize = fit(bitmap.width, bitmap.height, FULL_MAX_EDGE);
	const thumbSize = fit(bitmap.width, bitmap.height, THUMB_MAX_EDGE);
	const [full, thumb] = await Promise.all([
		encodeCanvasSource(bitmap, fullSize, keepPng ? 'image/png' : 'image/jpeg', FULL_QUALITY),
		encodeCanvasSource(bitmap, thumbSize, 'image/jpeg', THUMB_QUALITY)
	]);
	return {
		full,
		thumb,
		original: keepOriginal(original) ? original : null,
		width: fullSize.width,
		height: fullSize.height
	};
}

export async function prepareImage(file: File): Promise<PreparedImage> {
	const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
	try {
		return await prepareBitmap(bitmap, file, file.type === 'image/png');
	} finally {
		bitmap.close();
	}
}

export async function uploadArtworkImage(
	artworkId: string,
	prepared: PreparedImage,
	onProgress?: (fraction: number) => void
): Promise<{ id: string; thumbUrl: string; url: string }> {
	const body = new FormData();
	body.set('full', prepared.full, 'image');
	body.set('thumb', prepared.thumb, 'thumb');
	if (prepared.original) body.set('original', prepared.original, 'original');
	body.set('width', String(prepared.width));
	body.set('height', String(prepared.height));

	return new Promise((resolve, reject) => {
		const xhr = new XMLHttpRequest();
		xhr.open('POST', `/app/artworks/${artworkId}/images`);
		xhr.upload.onprogress = (event) => {
			if (event.lengthComputable && onProgress) onProgress(event.loaded / event.total);
		};
		xhr.onload = () => {
			try {
				const payload = JSON.parse(xhr.responseText) as {
					message?: string;
					id?: string;
					url?: string;
					thumbUrl?: string;
				};
				if (
					xhr.status >= 200 &&
					xhr.status < 300 &&
					payload.id &&
					payload.url &&
					payload.thumbUrl
				) {
					resolve({ id: payload.id, url: payload.url, thumbUrl: payload.thumbUrl });
				} else {
					reject(new Error(payload.message ?? `Upload failed (${xhr.status}).`));
				}
			} catch {
				reject(new Error(`Upload failed (${xhr.status}).`));
			}
		};
		xhr.onerror = () => reject(new Error('Network error during upload.'));
		xhr.send(body);
	});
}
