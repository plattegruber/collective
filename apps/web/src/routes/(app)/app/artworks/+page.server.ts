import { ARTWORK_STATUSES } from '#lib/server/db/schema.js';
import { listArtworks, listTags, type ArtworkSort } from '#lib/server/artworks.js';
import type { PageServerLoad } from './$types';

const SORTS = new Set<ArtworkSort>(['newest', 'oldest', 'title', 'artist', 'year', 'value']);

export const load: PageServerLoad = async ({ parent, url }) => {
	const { user } = await parent();
	const query = url.searchParams.get('q')?.trim() ?? '';
	const statusParam = url.searchParams.get('status') ?? 'all';
	const status = (ARTWORK_STATUSES as readonly string[]).includes(statusParam)
		? (statusParam as (typeof ARTWORK_STATUSES)[number])
		: 'all';
	const sortParam = url.searchParams.get('sort') ?? 'newest';
	const sort = SORTS.has(sortParam as ArtworkSort) ? (sortParam as ArtworkSort) : 'newest';
	const tag = url.searchParams.get('tag') ?? undefined;

	const [artworks, tags] = await Promise.all([
		listArtworks(user.id, { query: query || undefined, status, sort, tag }),
		listTags(user.id)
	]);
	return { artworks, tags, filters: { query, status, sort, tag: tag ?? '' } };
};
