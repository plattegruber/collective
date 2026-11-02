import { error } from '@sveltejs/kit';
import { getPublicArtwork, type ArtworkWithImages } from '#lib/server/artworks.js';
import type { PageServerLoad } from './$types';

/** Allow-list of fields a public page may show. Prices, notes, seller and location never leave the server. */
function publicView(artwork: ArtworkWithImages) {
	return {
		id: artwork.id,
		title: artwork.title,
		artist: artwork.artist,
		year: artwork.year,
		dateText: artwork.dateText,
		medium: artwork.medium,
		heightCm: artwork.heightCm,
		widthCm: artwork.widthCm,
		depthCm: artwork.depthCm,
		dimensionUnit: artwork.dimensionUnit,
		description: artwork.description,
		provenance: artwork.provenance,
		primaryImageId: artwork.primaryImageId,
		images: artwork.images,
		tags: artwork.tags.map((tag) => tag.name)
	};
}

export const load: PageServerLoad = async ({ parent, params, locals }) => {
	const { gallery } = await parent();
	const artwork = await getPublicArtwork(params.artworkId, locals.user?.id ?? null);
	if (!artwork || artwork.userId !== gallery.ownerId) error(404, 'Artwork not found');
	return { artwork: publicView(artwork) };
};
