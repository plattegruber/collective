import { listPublicArtworks } from '#lib/server/artworks.js';
import { activeRelease } from '#lib/server/recognition.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { gallery } = await parent();
	const [artworks, release] = await Promise.all([
		listPublicArtworks(gallery.ownerId),
		activeRelease()
	]);
	return {
		artworks,
		canScan: release !== null && artworks.some((artwork) => artwork.primaryImage)
	};
};
