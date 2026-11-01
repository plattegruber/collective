import { listArtworks } from '#lib/server/artworks.js';
import { activeModel, modelManifest } from '#lib/server/models.js';
import { toScannerContent } from '#lib/scanner/content.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { user } = await parent();
	const [model, artworks] = await Promise.all([activeModel(user.id), listArtworks(user.id)]);
	return {
		manifest: model ? modelManifest(model) : null,
		artworks: toScannerContent(artworks),
		photographed: artworks.filter((artwork) => artwork.primaryImage).length
	};
};
