/** Shared artwork vocabulary. Safe to import from both server and browser code. */

export const ARTWORK_STATUSES = ['owned', 'on_loan', 'sold', 'gifted', 'wishlist'] as const;
export type ArtworkStatus = (typeof ARTWORK_STATUSES)[number];

export const STATUS_LABELS: Record<ArtworkStatus, string> = {
	owned: 'Owned',
	on_loan: 'On loan',
	sold: 'Sold',
	gifted: 'Gifted',
	wishlist: 'Wishlist'
};

export const DIMENSION_UNITS = ['cm', 'in'] as const;
export type DimensionUnit = (typeof DIMENSION_UNITS)[number];
