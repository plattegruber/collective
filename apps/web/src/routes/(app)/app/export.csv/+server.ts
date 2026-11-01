import { requireUser } from '#lib/server/auth/guard.js';
import { listArtworks } from '#lib/server/artworks.js';
import { formatLength } from '#lib/utils/dimensions.js';
import { fromMinor } from '#lib/utils/money.js';
import { toCsv } from '#lib/utils/csv.js';
import type { RequestHandler } from './$types';

const HEADER = [
	'id',
	'title',
	'artist',
	'year',
	'date_text',
	'medium',
	'height',
	'width',
	'depth',
	'unit',
	'status',
	'location',
	'acquired_on',
	'acquired_from',
	'purchase_price',
	'estimated_value',
	'currency',
	'tags',
	'public',
	'description',
	'provenance',
	'notes',
	'photos',
	'created_at',
	'updated_at'
] as const;

/** Full inventory export for the signed-in collector: everything they can see, including private fields. */
export const GET: RequestHandler = async (event) => {
	const user = requireUser(event);
	const artworks = await listArtworks(user.id, { sort: 'artist' });
	const rows = artworks.map((artwork) => {
		const length = (cm: number | null) =>
			cm === null ? null : formatLength(cm, artwork.dimensionUnit);
		const money = (minor: number | null) =>
			minor === null ? null : fromMinor(minor, artwork.currency);
		return [
			artwork.id,
			artwork.title,
			artwork.artist,
			artwork.year,
			artwork.dateText,
			artwork.medium,
			length(artwork.heightCm),
			length(artwork.widthCm),
			length(artwork.depthCm),
			artwork.dimensionUnit,
			artwork.status,
			artwork.location,
			artwork.acquiredOn,
			artwork.acquiredFrom,
			money(artwork.purchasePrice),
			money(artwork.estimatedValue),
			artwork.currency,
			artwork.tags.map((tag) => tag.name).join('; '),
			artwork.isPublic ? 'yes' : 'no',
			artwork.description,
			artwork.provenance,
			artwork.notes,
			artwork.primaryImage ? 'yes' : 'no',
			artwork.createdAt,
			artwork.updatedAt
		];
	});
	const stamp = new Date().toISOString().slice(0, 10);
	return new Response(toCsv(HEADER, rows), {
		headers: {
			'content-type': 'text/csv; charset=utf-8',
			'content-disposition': `attachment; filename="collective-${stamp}.csv"`,
			'cache-control': 'no-store'
		}
	});
};
