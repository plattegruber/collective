/**
 * Client-side image preparation. Photos from phones are often 10+ MB; we downscale
 * before upload so the Worker never has to decode pixels (it cannot) and R2 stays lean.
 */

export const FULL_MAX_EDGE = 2400;
export const THUMB_MAX_EDGE = 480;

export interface PreparedImage {
	full: Blob;
	thumb: Blob;
	width: number;
	height: number;
}

function fit(width: number, height: number, maxEdge: number): { width: number; height: number } {
	const scale = Math.min(1, maxEdge / Math.max(width, height));
	return {
		width: Math.max(1, Math.round(width * scale)),
		height: Math.max(1, Math.round(height * scale))
	};
}

async function encode(
	source: ImageBitmap,
	size: { width: number; height: number },
	type: string,
	quality: number
) {
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

export async function prepareImage(file: File): Promise<PreparedImage> {
	const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
	try {
		const keepPng = file.type === 'image/png';
		const fullSize = fit(bitmap.width, bitmap.height, FULL_MAX_EDGE);
		const thumbSize = fit(bitmap.width, bitmap.height, THUMB_MAX_EDGE);
		const [full, thumb] = await Promise.all([
			encode(bitmap, fullSize, keepPng ? 'image/png' : 'image/jpeg', 0.88),
			encode(bitmap, thumbSize, 'image/jpeg', 0.82)
		]);
		return { full, thumb, width: fullSize.width, height: fullSize.height };
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
