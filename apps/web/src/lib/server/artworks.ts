import { and, asc, count, desc, eq, inArray, like, or, sql } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { newId } from '#lib/utils/id.js';
import { parseMoneyInput } from '#lib/utils/money.js';
import { toCm } from '#lib/utils/dimensions.js';
import { parseTagList, type ArtworkFormData } from '#lib/schemas/artwork.js';
import type { ArtworkStatus } from './db/schema.js';

export type ArtworkWithImages = schema.Artwork & {
	images: schema.ArtworkImage[];
	tags: schema.Tag[];
};

export type ArtworkListItem = schema.Artwork & {
	primaryImage: schema.ArtworkImage | null;
	tags: schema.Tag[];
};

export type ArtworkSort = 'newest' | 'oldest' | 'title' | 'artist' | 'year' | 'value';

export interface ListOptions {
	query?: string;
	status?: ArtworkStatus | 'all';
	tag?: string;
	sort?: ArtworkSort;
}

function fromForm(data: ArtworkFormData) {
	const purchasePrice = parseMoneyInput(data.purchasePrice, data.currency);
	const estimatedValue = parseMoneyInput(data.estimatedValue, data.currency);
	return {
		title: data.title,
		artist: data.artist ?? null,
		year: data.year ?? null,
		dateText: data.dateText ?? null,
		medium: data.medium ?? null,
		heightCm: data.height === undefined ? null : toCm(data.height, data.dimensionUnit),
		widthCm: data.width === undefined ? null : toCm(data.width, data.dimensionUnit),
		depthCm: data.depth === undefined ? null : toCm(data.depth, data.dimensionUnit),
		dimensionUnit: data.dimensionUnit,
		description: data.description ?? null,
		status: data.status,
		location: data.location ?? null,
		acquiredOn: data.acquiredOn ?? null,
		acquiredFrom: data.acquiredFrom ?? null,
		purchasePrice: purchasePrice ?? null,
		estimatedValue: estimatedValue ?? null,
		currency: data.currency,
		provenance: data.provenance ?? null,
		notes: data.notes ?? null,
		isPublic: data.isPublic
	};
}

/** Returns field errors for values the schema cannot validate without context. */
export function validateMoney(data: ArtworkFormData): Record<string, string> {
	const errors: Record<string, string> = {};
	if (parseMoneyInput(data.purchasePrice, data.currency) === undefined)
		errors.purchasePrice = 'Enter an amount.';
	if (parseMoneyInput(data.estimatedValue, data.currency) === undefined)
		errors.estimatedValue = 'Enter an amount.';
	return errors;
}

export async function createArtwork(userId: string, data: ArtworkFormData): Promise<string> {
	const id = newId();
	const now = new Date();
	await db()
		.insert(schema.artworks)
		.values({ id, userId, ...fromForm(data), updatedAt: now });
	await syncTags(userId, id, parseTagList(data.tags));
	return id;
}

export async function updateArtwork(
	userId: string,
	id: string,
	data: ArtworkFormData
): Promise<boolean> {
	const result = await db()
		.update(schema.artworks)
		.set({ ...fromForm(data), updatedAt: new Date() })
		.where(and(eq(schema.artworks.id, id), eq(schema.artworks.userId, userId)))
		.returning({ id: schema.artworks.id });
	if (result.length === 0) return false;
	await syncTags(userId, id, parseTagList(data.tags));
	return true;
}

export async function deleteArtwork(userId: string, id: string): Promise<boolean> {
	const result = await db()
		.delete(schema.artworks)
		.where(and(eq(schema.artworks.id, id), eq(schema.artworks.userId, userId)))
		.returning({ id: schema.artworks.id });
	return result.length > 0;
}

export async function getArtwork(userId: string, id: string): Promise<ArtworkWithImages | null> {
	const row = await db().query.artworks.findFirst({
		where: and(eq(schema.artworks.id, id), eq(schema.artworks.userId, userId)),
		with: {
			images: { orderBy: asc(schema.artworkImages.position) },
			artworkTags: { with: { tag: true } }
		}
	});
	if (!row) return null;
	const { artworkTags, ...artwork } = row;
	return {
		...artwork,
		tags: artworkTags.map((at) => at.tag).sort((a, b) => a.name.localeCompare(b.name))
	};
}

/** Artwork visible on a public gallery; checks the owner's public flag too. */
export async function getPublicArtwork(
	id: string
): Promise<(ArtworkWithImages & { owner: schema.User }) | null> {
	const row = await db().query.artworks.findFirst({
		where: and(eq(schema.artworks.id, id), eq(schema.artworks.isPublic, true)),
		with: {
			owner: true,
			images: { orderBy: asc(schema.artworkImages.position) },
			artworkTags: { with: { tag: true } }
		}
	});
	if (!row || !row.owner.galleryPublic) return null;
	const { artworkTags, ...artwork } = row;
	return { ...artwork, tags: artworkTags.map((at) => at.tag) };
}

export async function listArtworks(
	userId: string,
	options: ListOptions = {}
): Promise<ArtworkListItem[]> {
	const conditions = [eq(schema.artworks.userId, userId)];
	if (options.status && options.status !== 'all')
		conditions.push(eq(schema.artworks.status, options.status));
	if (options.query) {
		const pattern = `%${options.query.replace(/[%_]/g, '')}%`;
		conditions.push(
			or(
				like(schema.artworks.title, pattern),
				like(schema.artworks.artist, pattern),
				like(schema.artworks.medium, pattern),
				like(schema.artworks.location, pattern)
			)!
		);
	}
	if (options.tag) {
		conditions.push(
			inArray(
				schema.artworks.id,
				db()
					.select({ id: schema.artworkTags.artworkId })
					.from(schema.artworkTags)
					.innerJoin(schema.tags, eq(schema.tags.id, schema.artworkTags.tagId))
					.where(and(eq(schema.tags.userId, userId), eq(schema.tags.name, options.tag)))
			)
		);
	}

	const orderBy = {
		newest: [desc(schema.artworks.createdAt)],
		oldest: [asc(schema.artworks.createdAt)],
		title: [asc(schema.artworks.title)],
		artist: [asc(schema.artworks.artist), asc(schema.artworks.title)],
		year: [desc(schema.artworks.year), asc(schema.artworks.title)],
		value: [desc(schema.artworks.estimatedValue), desc(schema.artworks.purchasePrice)]
	}[options.sort ?? 'newest'];

	const rows = await db().query.artworks.findMany({
		where: and(...conditions),
		orderBy,
		with: {
			images: { orderBy: asc(schema.artworkImages.position) },
			artworkTags: { with: { tag: true } }
		}
	});

	return rows.map(({ images, artworkTags, ...artwork }) => ({
		...artwork,
		primaryImage: images.find((image) => image.id === artwork.primaryImageId) ?? images[0] ?? null,
		tags: artworkTags.map((at) => at.tag).sort((a, b) => a.name.localeCompare(b.name))
	}));
}

export async function listPublicArtworks(userId: string): Promise<ArtworkListItem[]> {
	const rows = await db().query.artworks.findMany({
		where: and(eq(schema.artworks.userId, userId), eq(schema.artworks.isPublic, true)),
		orderBy: [asc(schema.artworks.artist), asc(schema.artworks.title)],
		with: {
			images: { orderBy: asc(schema.artworkImages.position) },
			artworkTags: { with: { tag: true } }
		}
	});
	return rows.map(({ images, artworkTags, ...artwork }) => ({
		...artwork,
		primaryImage: images.find((image) => image.id === artwork.primaryImageId) ?? images[0] ?? null,
		tags: artworkTags.map((at) => at.tag)
	}));
}

export interface CollectionStats {
	total: number;
	owned: number;
	withImages: number;
	publicCount: number;
	artists: number;
	/** Sum of estimated values, falling back to purchase price, grouped by currency. */
	valueByCurrency: Record<string, number>;
	costByCurrency: Record<string, number>;
}

export async function collectionStats(userId: string): Promise<CollectionStats> {
	const database = db();
	const [rows, imageRows] = await Promise.all([
		database
			.select({
				id: schema.artworks.id,
				status: schema.artworks.status,
				artist: schema.artworks.artist,
				isPublic: schema.artworks.isPublic,
				currency: schema.artworks.currency,
				purchasePrice: schema.artworks.purchasePrice,
				estimatedValue: schema.artworks.estimatedValue
			})
			.from(schema.artworks)
			.where(eq(schema.artworks.userId, userId)),
		database
			.select({ artworkId: schema.artworkImages.artworkId, count: count() })
			.from(schema.artworkImages)
			.innerJoin(schema.artworks, eq(schema.artworks.id, schema.artworkImages.artworkId))
			.where(eq(schema.artworks.userId, userId))
			.groupBy(schema.artworkImages.artworkId)
	]);
	const imageCounts = new Map(imageRows.map((row) => [row.artworkId, row.count]));

	const stats: CollectionStats = {
		total: rows.length,
		owned: 0,
		withImages: 0,
		publicCount: 0,
		artists: 0,
		valueByCurrency: {},
		costByCurrency: {}
	};
	const artists = new Set<string>();
	for (const row of rows) {
		if (row.status === 'owned' || row.status === 'on_loan') stats.owned += 1;
		if ((imageCounts.get(row.id) ?? 0) > 0) stats.withImages += 1;
		if (row.isPublic) stats.publicCount += 1;
		if (row.artist) artists.add(row.artist.trim().toLowerCase());
		const value = row.estimatedValue ?? row.purchasePrice;
		if (value !== null && row.status !== 'sold' && row.status !== 'wishlist') {
			stats.valueByCurrency[row.currency] = (stats.valueByCurrency[row.currency] ?? 0) + value;
		}
		if (row.purchasePrice !== null && row.status !== 'wishlist') {
			stats.costByCurrency[row.currency] =
				(stats.costByCurrency[row.currency] ?? 0) + row.purchasePrice;
		}
	}
	stats.artists = artists.size;
	return stats;
}

export async function listTags(userId: string): Promise<(schema.Tag & { count: number })[]> {
	const rows = await db()
		.select({
			id: schema.tags.id,
			userId: schema.tags.userId,
			name: schema.tags.name,
			count: sql<number>`count(${schema.artworkTags.artworkId})`
		})
		.from(schema.tags)
		.leftJoin(schema.artworkTags, eq(schema.artworkTags.tagId, schema.tags.id))
		.where(eq(schema.tags.userId, userId))
		.groupBy(schema.tags.id)
		.orderBy(asc(schema.tags.name));
	return rows;
}

async function syncTags(userId: string, artworkId: string, names: string[]): Promise<void> {
	const database = db();
	await database.delete(schema.artworkTags).where(eq(schema.artworkTags.artworkId, artworkId));
	if (names.length === 0) {
		await pruneUnusedTags(userId);
		return;
	}
	const existing = await database.query.tags.findMany({
		where: and(eq(schema.tags.userId, userId), inArray(schema.tags.name, names))
	});
	const byName = new Map(existing.map((tag) => [tag.name, tag.id]));
	const missing = names
		.filter((name) => !byName.has(name))
		.map((name) => ({ id: newId(), userId, name }));
	if (missing.length > 0) {
		await database.insert(schema.tags).values(missing);
		for (const tag of missing) byName.set(tag.name, tag.id);
	}
	await database
		.insert(schema.artworkTags)
		.values(names.map((name) => ({ artworkId, tagId: byName.get(name)! })));
	await pruneUnusedTags(userId);
}

async function pruneUnusedTags(userId: string): Promise<void> {
	await db()
		.delete(schema.tags)
		.where(
			and(
				eq(schema.tags.userId, userId),
				sql`not exists (select 1 from ${schema.artworkTags} where ${schema.artworkTags.tagId} = ${schema.tags.id})`
			)
		);
}
