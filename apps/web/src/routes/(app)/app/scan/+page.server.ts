import { listArtworks } from '#lib/server/artworks.js';
import {
	activeRelease,
	listPrototypes,
	releaseManifest,
	toIndexPayload
} from '#lib/server/recognition.js';
import { toScannerContent } from '#lib/scanner/content.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent }) => {
	const { user } = await parent();
	const [release, artworks] = await Promise.all([activeRelease(), listArtworks(user.id)]);
	if (!release) {
		return { manifest: null, index: null, artworks: toScannerContent(artworks), canTeach: true };
	}
	const prototypes = await listPrototypes(
		release.id,
		artworks.map((artwork) => artwork.id)
	);
	return {
		manifest: releaseManifest(release),
		index: toIndexPayload(release.id, release.embedderDim, prototypes),
		artworks: toScannerContent(artworks),
		canTeach: true
	};
};
