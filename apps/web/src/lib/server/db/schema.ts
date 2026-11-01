import { relations, sql } from 'drizzle-orm';
import {
	index,
	integer,
	primaryKey,
	real,
	sqliteTable,
	text,
	uniqueIndex
} from 'drizzle-orm/sqlite-core';

/** Unix epoch milliseconds, stored as an integer. */
const timestamp = (name: string) => integer(name, { mode: 'timestamp_ms' });

const createdAt = () =>
	timestamp('created_at')
		.notNull()
		.default(sql`(unixepoch('subsec') * 1000)`);

export const users = sqliteTable(
	'users',
	{
		id: text('id').primaryKey(),
		email: text('email').notNull(),
		passwordHash: text('password_hash').notNull(),
		name: text('name').notNull(),
		/** URL-safe handle for the public gallery, e.g. /g/{handle}. */
		handle: text('handle'),
		galleryPublic: integer('gallery_public', { mode: 'boolean' }).notNull().default(false),
		galleryTitle: text('gallery_title'),
		galleryIntro: text('gallery_intro'),
		createdAt: createdAt(),
		updatedAt: timestamp('updated_at').notNull()
	},
	(t) => [uniqueIndex('users_email_idx').on(t.email), uniqueIndex('users_handle_idx').on(t.handle)]
);

export const sessions = sqliteTable(
	'sessions',
	{
		/** SHA-256 of the opaque token handed to the browser. */
		id: text('id').primaryKey(),
		userId: text('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		expiresAt: timestamp('expires_at').notNull(),
		createdAt: createdAt()
	},
	(t) => [index('sessions_user_idx').on(t.userId)]
);

export const apiTokens = sqliteTable(
	'api_tokens',
	{
		/** SHA-256 of the token; the raw value is shown once at creation. */
		id: text('id').primaryKey(),
		userId: text('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		name: text('name').notNull(),
		prefix: text('prefix').notNull(),
		lastUsedAt: timestamp('last_used_at'),
		createdAt: createdAt()
	},
	(t) => [index('api_tokens_user_idx').on(t.userId)]
);

import { ARTWORK_STATUSES, DIMENSION_UNITS } from '#lib/domain/artwork.js';

export { ARTWORK_STATUSES, DIMENSION_UNITS } from '#lib/domain/artwork.js';
export type { ArtworkStatus, DimensionUnit } from '#lib/domain/artwork.js';

export const artworks = sqliteTable(
	'artworks',
	{
		id: text('id').primaryKey(),
		userId: text('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		title: text('title').notNull(),
		artist: text('artist'),
		year: integer('year'),
		/** Free-form date text when a single year is not right, e.g. "c. 1920" or "1998–2001". */
		dateText: text('date_text'),
		medium: text('medium'),
		heightCm: real('height_cm'),
		widthCm: real('width_cm'),
		depthCm: real('depth_cm'),
		/** The unit the collector prefers to see; measurements are always stored in cm. */
		dimensionUnit: text('dimension_unit', { enum: DIMENSION_UNITS }).notNull().default('cm'),
		description: text('description'),
		status: text('status', { enum: ARTWORK_STATUSES }).notNull().default('owned'),
		location: text('location'),
		acquiredOn: text('acquired_on'),
		acquiredFrom: text('acquired_from'),
		/** Minor units (cents) in `currency`. */
		purchasePrice: integer('purchase_price'),
		/** Minor units (cents) in `currency`. */
		estimatedValue: integer('estimated_value'),
		currency: text('currency').notNull().default('USD'),
		provenance: text('provenance'),
		notes: text('notes'),
		isPublic: integer('is_public', { mode: 'boolean' }).notNull().default(false),
		primaryImageId: text('primary_image_id'),
		createdAt: createdAt(),
		updatedAt: timestamp('updated_at').notNull()
	},
	(t) => [
		index('artworks_user_idx').on(t.userId),
		index('artworks_user_status_idx').on(t.userId, t.status)
	]
);

export const artworkImages = sqliteTable(
	'artwork_images',
	{
		id: text('id').primaryKey(),
		artworkId: text('artwork_id')
			.notNull()
			.references(() => artworks.id, { onDelete: 'cascade' }),
		/** R2 object key of the full-size image. */
		key: text('key').notNull(),
		/** R2 object key of the ~480px thumbnail. */
		thumbKey: text('thumb_key').notNull(),
		contentType: text('content_type').notNull(),
		width: integer('width').notNull(),
		height: integer('height').notNull(),
		bytes: integer('bytes').notNull(),
		position: integer('position').notNull().default(0),
		createdAt: createdAt()
	},
	(t) => [
		index('artwork_images_artwork_idx').on(t.artworkId),
		uniqueIndex('artwork_images_key_idx').on(t.key)
	]
);

export const tags = sqliteTable(
	'tags',
	{
		id: text('id').primaryKey(),
		userId: text('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		name: text('name').notNull()
	},
	(t) => [uniqueIndex('tags_user_name_idx').on(t.userId, t.name)]
);

export const artworkTags = sqliteTable(
	'artwork_tags',
	{
		artworkId: text('artwork_id')
			.notNull()
			.references(() => artworks.id, { onDelete: 'cascade' }),
		tagId: text('tag_id')
			.notNull()
			.references(() => tags.id, { onDelete: 'cascade' })
	},
	(t) => [primaryKey({ columns: [t.artworkId, t.tagId] })]
);

export const detectorModels = sqliteTable(
	'detector_models',
	{
		id: text('id').primaryKey(),
		userId: text('user_id')
			.notNull()
			.references(() => users.id, { onDelete: 'cascade' }),
		/** R2 key of the ONNX graph. */
		modelKey: text('model_key').notNull(),
		/** JSON object mapping class index (as string) to artwork id. */
		labels: text('labels', { mode: 'json' }).$type<Record<string, string>>().notNull(),
		sha256: text('sha256').notNull(),
		bytes: integer('bytes').notNull(),
		inputSize: integer('input_size').notNull().default(320),
		/** Free-form training metrics (mAP, epochs, dataset size…). */
		metrics: text('metrics', { mode: 'json' }).$type<Record<string, unknown>>(),
		/** Only one model per user is active at a time. */
		isActive: integer('is_active', { mode: 'boolean' }).notNull().default(true),
		createdAt: createdAt()
	},
	(t) => [index('detector_models_user_idx').on(t.userId)]
);

export const usersRelations = relations(users, ({ many }) => ({
	artworks: many(artworks),
	tags: many(tags),
	detectorModels: many(detectorModels)
}));

export const artworksRelations = relations(artworks, ({ one, many }) => ({
	owner: one(users, { fields: [artworks.userId], references: [users.id] }),
	images: many(artworkImages),
	artworkTags: many(artworkTags)
}));

export const artworkImagesRelations = relations(artworkImages, ({ one }) => ({
	artwork: one(artworks, { fields: [artworkImages.artworkId], references: [artworks.id] })
}));

export const tagsRelations = relations(tags, ({ many }) => ({
	artworkTags: many(artworkTags)
}));

export const artworkTagsRelations = relations(artworkTags, ({ one }) => ({
	artwork: one(artworks, { fields: [artworkTags.artworkId], references: [artworks.id] }),
	tag: one(tags, { fields: [artworkTags.tagId], references: [tags.id] })
}));

export type User = typeof users.$inferSelect;
export type Artwork = typeof artworks.$inferSelect;
export type NewArtwork = typeof artworks.$inferInsert;
export type ArtworkImage = typeof artworkImages.$inferSelect;
export type Tag = typeof tags.$inferSelect;
export type DetectorModel = typeof detectorModels.$inferSelect;
