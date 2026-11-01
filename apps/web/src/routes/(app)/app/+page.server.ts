import { collectionStats, listArtworks } from '#lib/server/artworks.js';
import { activeModel } from '#lib/server/models.js';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ parent, url }) => {
	const { user } = await parent();
	const [stats, recent, model] = await Promise.all([
		collectionStats(user.id),
		listArtworks(user.id, { sort: 'newest' }),
		activeModel(user.id)
	]);
	return {
		stats,
		recent: recent.slice(0, 6),
		model: model
			? { createdAt: model.createdAt, labelCount: Object.keys(model.labels).length }
			: null,
		welcome: url.searchParams.has('welcome')
	};
};
