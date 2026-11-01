import { error } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import { authenticateApiToken } from '#lib/server/apiTokens.js';
import { listArtworks } from '#lib/server/artworks.js';
import { db, schema } from '#lib/server/db/index.js';
import { asc, inArray } from 'drizzle-orm';
import type { RequestHandler } from './$types';

/**
 * Collection export for the training pipeline.
 * Returns every artwork with signed-in-by-token image URLs the pipeline can download.
 */
export const GET: RequestHandler = async ({ request, url }) => {
	const auth = await authenticateApiToken(request);
	if (!auth) error(401, 'Invalid or missing API token.');

	const artworks = await listArtworks(auth.userId, { sort: 'title' });
	const ids = artworks.map((artwork) => artwork.id);
	const images = ids.length
		? await db().query.artworkImages.findMany({
				where: inArray(schema.artworkImages.artworkId, ids),
				orderBy: asc(schema.artworkImages.position)
			})
		: [];
	const byArtwork = new Map<string, typeof images>();
	for (const image of images) {
		const list = byArtwork.get(image.artworkId) ?? [];
		list.push(image);
		byArtwork.set(image.artworkId, list);
	}

	return Response.json({
		userId: auth.userId,
		origin: url.origin,
		publicOrigin: env.PUBLIC_ORIGIN,
		artworks: artworks.map((artwork) => ({
			id: artwork.id,
			title: artwork.title,
			artist: artwork.artist,
			year: artwork.year,
			status: artwork.status,
			images: (byArtwork.get(artwork.id) ?? []).map((image) => ({
				id: image.id,
				url: `${url.origin}/api/ml/media/${image.key}`,
				width: image.width,
				height: image.height,
				contentType: image.contentType
			}))
		}))
	});
};
