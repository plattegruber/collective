import type { ArtworkWithImages } from '#lib/server/artworks.js';
import { fromMinor } from '#lib/utils/money.js';
import { formatLength } from '#lib/utils/dimensions.js';

/** Turns a stored artwork back into the string values the form expects. */
export function artworkToFormValues(artwork: ArtworkWithImages): Record<string, string> {
	const unit = artwork.dimensionUnit;
	const length = (cm: number | null) => (cm === null ? '' : formatLength(cm, unit));
	const money = (minor: number | null) =>
		minor === null ? '' : String(fromMinor(minor, artwork.currency));
	return {
		title: artwork.title,
		artist: artwork.artist ?? '',
		year: artwork.year === null ? '' : String(artwork.year),
		dateText: artwork.dateText ?? '',
		medium: artwork.medium ?? '',
		height: length(artwork.heightCm),
		width: length(artwork.widthCm),
		depth: length(artwork.depthCm),
		dimensionUnit: unit,
		description: artwork.description ?? '',
		status: artwork.status,
		location: artwork.location ?? '',
		acquiredOn: artwork.acquiredOn ?? '',
		acquiredFrom: artwork.acquiredFrom ?? '',
		purchasePrice: money(artwork.purchasePrice),
		estimatedValue: money(artwork.estimatedValue),
		currency: artwork.currency,
		provenance: artwork.provenance ?? '',
		notes: artwork.notes ?? '',
		tags: artwork.tags.map((tag) => tag.name).join(', '),
		isPublic: artwork.isPublic ? 'on' : ''
	};
}
