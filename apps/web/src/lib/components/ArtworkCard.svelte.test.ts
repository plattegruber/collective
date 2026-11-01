import { page } from 'vitest/browser';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import ArtworkCard from './ArtworkCard.svelte';
import type { ArtworkListItem } from '#lib/server/artworks.js';

const base: ArtworkListItem = {
	id: 'art_1',
	userId: 'user_1',
	title: 'Night Orchard',
	artist: 'K. Lindqvist',
	year: 2020,
	dateText: null,
	medium: 'Oil on linen',
	heightCm: 61,
	widthCm: 45.7,
	depthCm: null,
	dimensionUnit: 'cm',
	description: null,
	status: 'owned',
	location: null,
	acquiredOn: null,
	acquiredFrom: null,
	purchasePrice: null,
	estimatedValue: null,
	currency: 'USD',
	provenance: null,
	notes: null,
	isPublic: false,
	primaryImageId: null,
	createdAt: new Date(0),
	updatedAt: new Date(0),
	primaryImage: null,
	tags: []
};

describe('ArtworkCard', () => {
	it('links to the artwork with an accessible name built from title and byline', async () => {
		render(ArtworkCard, { artwork: base });
		const link = page.getByRole('link', { name: 'Night Orchard, K. Lindqvist, 2020' });
		await expect.element(link).toHaveAttribute('href', '/app/artworks/art_1');
	});

	it('shows a status badge only for pieces that are not simply owned', async () => {
		render(ArtworkCard, { artwork: { ...base, status: 'on_loan' } });
		await expect.element(page.getByText('On loan')).toBeInTheDocument();
	});

	it('renders the thumbnail from the primary image', () => {
		const { container } = render(ArtworkCard, {
			artwork: {
				...base,
				primaryImage: {
					id: 'img',
					artworkId: 'art_1',
					key: 'users/u/artworks/a/i.jpg',
					thumbKey: 'users/u/artworks/a/i.thumb.jpg',
					contentType: 'image/jpeg',
					width: 10,
					height: 10,
					bytes: 1,
					position: 0,
					createdAt: new Date(0)
				}
			}
		});
		// The image is decorative (alt=""), so it has no accessible role; query the DOM directly.
		expect(container.querySelector('img')?.getAttribute('src')).toBe(
			'/media/users/u/artworks/a/i.thumb.jpg'
		);
	});
});
