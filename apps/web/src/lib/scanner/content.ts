import type { ArtworkListItem } from '#lib/server/artworks.js';
import { byline } from '#lib/utils/format.js';
import type { ScannerArtwork } from './ScannerOverlay.svelte';

/** Trims an artwork down to what the overlay shows; private fields never reach the scanner. */
export function toScannerContent(artworks: ArtworkListItem[]): Record<string, ScannerArtwork> {
	const content: Record<string, ScannerArtwork> = {};
	for (const artwork of artworks) {
		content[artwork.id] = {
			title: artwork.title,
			byline: byline(artwork.artist, artwork.year, artwork.dateText),
			medium: artwork.medium ?? '',
			description: artwork.description ?? ''
		};
	}
	return content;
}

/** Keeps only labels that point at artworks the viewer may see. */
export function restrictLabels(
	labels: Record<string, string>,
	allowed: Set<string>
): Record<string, string> {
	return Object.fromEntries(
		Object.entries(labels).filter(([, artworkId]) => allowed.has(artworkId))
	);
}
