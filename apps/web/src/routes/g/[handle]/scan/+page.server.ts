import { error } from '@sveltejs/kit';
import { listPublicArtworks } from '#lib/server/artworks.js';
import {
	activeRelease,
	listPrototypes,
	releaseManifest,
	toIndexPayload
} from '#lib/server/recognition.js';
import { toScannerContent } from '#lib/scanner/content.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { gallery } = await parent();
	const [release, artworks] = await Promise.all([
		activeRelease(),
		listPublicArtworks(gallery.ownerId)
	]);
	if (!release) error(404, 'The scanner is not available yet.');
	const prototypes = await listPrototypes(
		release.id,
		artworks.map((artwork) => artwork.id)
	);
	return {
		manifest: releaseManifest(release),
		index: toIndexPayload(release.id, release.embedderDim, prototypes),
		artworks: toScannerContent(artworks),
		canTeach: gallery.isOwner
	};
};
