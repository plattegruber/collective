import { listPublicArtworks } from '#lib/server/artworks.js';
import { activeModel } from '#lib/server/models.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { gallery } = await parent();
	const [artworks, model] = await Promise.all([
		listPublicArtworks(gallery.ownerId),
		activeModel(gallery.ownerId)
	]);
	return { artworks, canScan: Boolean(model) };
};
