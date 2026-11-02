import { collectionStats, listArtworks } from '#lib/server/artworks.js';
import { activeRelease, indexStatus } from '#lib/server/recognition.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent, url }) => {
	const { user } = await parent();
	const [stats, recent, release] = await Promise.all([
		collectionStats(user.id),
		listArtworks(user.id, { sort: 'newest' }),
		activeRelease()
	]);
	const status = release ? await indexStatus(user.id, release.id) : null;
	return {
		stats,
		recent: recent.slice(0, 6),
		recognizer: status
			? { indexed: status.indexed, images: status.images, pending: status.pending.length }
			: null,
		welcome: url.searchParams.has('welcome')
	};
};
