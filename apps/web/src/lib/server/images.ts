import { env } from 'cloudflare:workers';
import { and, asc, eq, sql } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { newId } from '#lib/utils/id.js';

export const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
/** Camera originals are kept as shot (30 MB covers a 48 MP phone JPEG), never decoded here. */
export const MAX_ORIGINAL_BYTES = 30 * 1024 * 1024;
export const ALLOWED_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
export const ALLOWED_ORIGINAL_TYPES = new Set([...ALLOWED_IMAGE_TYPES, 'image/heic', 'image/heif']);
export const MAX_IMAGES_PER_ARTWORK = 12;

function extensionFor(contentType: string): string {
	return (
		{
			'image/jpeg': 'jpg',
			'image/png': 'png',
			'image/webp': 'webp',
			'image/heic': 'heic',
			'image/heif': 'heif'
		}[contentType] ?? 'bin'
	);
}

export interface UploadedImage {
	full: { bytes: ArrayBuffer; contentType: string; width: number; height: number };
	thumb: { bytes: ArrayBuffer; contentType: string };
	/** The upload as the camera produced it, kept for re-processing and downloads. */
	original?: { bytes: ArrayBuffer; contentType: string };
}

/** Stores a full-size image, its thumbnail and optionally the original in R2 and records them on the artwork. */
export async function addArtworkImage(
	userId: string,
	artworkId: string,
	upload: UploadedImage
): Promise<schema.ArtworkImage> {
	const artwork = await db().query.artworks.findFirst({
		where: and(eq(schema.artworks.id, artworkId), eq(schema.artworks.userId, userId)),
		columns: { id: true, primaryImageId: true },
		with: { images: { columns: { id: true } } }
	});
	if (!artwork) throw new ImageError('not_found', 'Artwork not found.');
	if (artwork.images.length >= MAX_IMAGES_PER_ARTWORK) {
		throw new ImageError('limit', `Up to ${MAX_IMAGES_PER_ARTWORK} images per artwork.`);
	}

	const id = newId();
	const ext = extensionFor(upload.full.contentType);
	const base = `users/${userId}/artworks/${artworkId}/${id}`;
	const key = `${base}.${ext}`;
	const thumbKey = `${base}.thumb.${extensionFor(upload.thumb.contentType)}`;
	const originalKey = upload.original
		? `${base}.original.${extensionFor(upload.original.contentType)}`
		: null;

	await Promise.all([
		env.MEDIA.put(key, upload.full.bytes, {
			httpMetadata: { contentType: upload.full.contentType }
		}),
		env.MEDIA.put(thumbKey, upload.thumb.bytes, {
			httpMetadata: { contentType: upload.thumb.contentType }
		}),
		originalKey && upload.original
			? env.MEDIA.put(originalKey, upload.original.bytes, {
					httpMetadata: { contentType: upload.original.contentType }
				})
			: Promise.resolve()
	]);

	// Computed in the INSERT itself so concurrent uploads never share a position.
	const position = sql<number>`(select coalesce(max(${schema.artworkImages.position}) + 1, 0) from ${schema.artworkImages} where ${schema.artworkImages.artworkId} = ${artworkId})`;
	const [image] = await db()
		.insert(schema.artworkImages)
		.values({
			id,
			artworkId,
			key,
			thumbKey,
			originalKey,
			contentType: upload.full.contentType,
			width: upload.full.width,
			height: upload.full.height,
			bytes: upload.full.bytes.byteLength,
			position
		})
		.returning();

	if (!artwork.primaryImageId) {
		await db()
			.update(schema.artworks)
			.set({ primaryImageId: id })
			.where(eq(schema.artworks.id, artworkId));
	}
	return image!;
}

export async function removeArtworkImage(
	userId: string,
	artworkId: string,
	imageId: string
): Promise<boolean> {
	const image = await db().query.artworkImages.findFirst({
		where: and(eq(schema.artworkImages.id, imageId), eq(schema.artworkImages.artworkId, artworkId)),
		with: { artwork: { columns: { userId: true, primaryImageId: true } } }
	});
	if (!image || image.artwork.userId !== userId) return false;

	await db().delete(schema.artworkImages).where(eq(schema.artworkImages.id, imageId));
	await env.MEDIA.delete(
		[image.key, image.thumbKey, image.originalKey].filter((k): k is string => k !== null)
	);

	if (image.artwork.primaryImageId === imageId) {
		const next = await db().query.artworkImages.findFirst({
			where: eq(schema.artworkImages.artworkId, artworkId),
			orderBy: asc(schema.artworkImages.position)
		});
		await db()
			.update(schema.artworks)
			.set({ primaryImageId: next?.id ?? null })
			.where(eq(schema.artworks.id, artworkId));
	}
	return true;
}

export async function setPrimaryImage(
	userId: string,
	artworkId: string,
	imageId: string
): Promise<boolean> {
	const image = await db().query.artworkImages.findFirst({
		where: and(eq(schema.artworkImages.id, imageId), eq(schema.artworkImages.artworkId, artworkId)),
		with: { artwork: { columns: { userId: true } } }
	});
	if (!image || image.artwork.userId !== userId) return false;
	await db()
		.update(schema.artworks)
		.set({ primaryImageId: imageId })
		.where(eq(schema.artworks.id, artworkId));
	return true;
}

/** Deletes every R2 object under an artwork before the row cascade removes the records. */
export async function deleteArtworkMedia(userId: string, artworkId: string): Promise<void> {
	const prefix = `users/${userId}/artworks/${artworkId}/`;
	let cursor: string | undefined;
	do {
		const listed = await env.MEDIA.list({ prefix, cursor });
		if (listed.objects.length > 0)
			await env.MEDIA.delete(listed.objects.map((object) => object.key));
		cursor = listed.truncated ? listed.cursor : undefined;
	} while (cursor);
}

/**
 * Resolves who may read an object key. Keys are unguessable, but we still gate on
 * ownership or the owner's sharing settings so revoking visibility actually revokes access.
 *
 * Layout: `users/{ownerId}/artworks/{artworkId}/…` for photos and
 * `users/{ownerId}/models/{modelId}/…` for detector models.
 */
export async function canReadMedia(key: string, viewerId: string | null): Promise<boolean> {
	const match = /^users\/([^/]+)\/(artworks|models)\/([^/]+)\//.exec(key);
	if (!match) return false;
	const [, ownerId, kind, resourceId] = match;
	if (viewerId && viewerId === ownerId) return true;

	if (kind === 'models') {
		const owner = await db().query.users.findFirst({
			where: eq(schema.users.id, ownerId!),
			columns: { galleryPublic: true }
		});
		return Boolean(owner?.galleryPublic);
	}

	const artwork = await db().query.artworks.findFirst({
		where: eq(schema.artworks.id, resourceId!),
		columns: { isPublic: true },
		with: { owner: { columns: { galleryPublic: true } } }
	});
	return Boolean(artwork?.isPublic && artwork.owner.galleryPublic);
}

export class ImageError extends Error {
	constructor(
		public readonly code: 'not_found' | 'limit' | 'invalid',
		message: string
	) {
		super(message);
	}
}

export function mediaUrl(key: string): string {
	return `/media/${key}`;
}
