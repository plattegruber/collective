import { error } from '@sveltejs/kit';
import { listPublicArtworks } from '#lib/server/artworks.js';
import { activeModel, modelManifest } from '#lib/server/models.js';
import { restrictLabels, toScannerContent } from '#lib/scanner/content.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { gallery } = await parent();
	const [model, artworks] = await Promise.all([
		activeModel(gallery.ownerId),
		listPublicArtworks(gallery.ownerId)
	]);
	if (!model) error(404, 'This gallery has no scanner yet.');
	const manifest = modelManifest(model);
	const visible = new Set(artworks.map((artwork) => artwork.id));
	return {
		manifest: { ...manifest, labels: restrictLabels(manifest.labels, visible) },
		artworks: toScannerContent(artworks)
	};
};
