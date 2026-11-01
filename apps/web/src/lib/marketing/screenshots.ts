/**
 * Real product screenshots in `static/marketing/`, produced by `scripts/capture-marketing.mjs`
 * (and `scripts/scanner-smoke.mjs` for the scanner) against a seeded account. Set
 * `screenshotsAvailable` to false to fall back to the live mockups.
 */
export interface Screenshot {
	src: string;
	alt: string;
	caption: string;
	width: number;
	height: number;
}

export const screenshotsAvailable = true;

const desktop = { width: 2400, height: 1500 };
const phone = { width: 1170, height: 2532 };

export const screenshots: Screenshot[] = [
	{
		src: '/marketing/dashboard.jpg',
		alt: 'The overview page with collection statistics and recently added artworks',
		caption: 'Overview at a glance',
		...desktop
	},
	{
		src: '/marketing/artworks.jpg',
		alt: 'A grid of artworks with search, status filter and sort controls',
		caption: 'Your whole collection',
		...desktop
	},
	{
		src: '/marketing/artwork.jpg',
		alt: 'An artwork detail page with photo, description and facts',
		caption: 'Every detail, in one place',
		...desktop
	},
	{
		src: '/marketing/scanner.jpg',
		alt: 'The phone scanner recognizing a print and showing its story with emoji reactions',
		caption: 'Point a phone, know the piece',
		...phone
	},
	{
		src: '/marketing/artwork-edit.jpg',
		alt: 'The artwork form with dimensions, acquisition, value and provenance fields',
		caption: 'Record what you paid and what it is worth',
		...desktop
	},
	{
		src: '/marketing/gallery-phone.jpg',
		alt: 'A public gallery page on a phone with a Scan a piece button',
		caption: 'Share a public gallery',
		...phone
	},
	{
		src: '/marketing/artwork-phone.jpg',
		alt: 'A public artwork page on a phone with reactions',
		caption: 'Guests can react',
		...phone
	},
	{
		src: '/marketing/settings.jpg',
		alt: 'Settings with public gallery, recognizer models and API tokens',
		caption: 'Gallery, recognizer and export in Settings',
		...desktop
	},
	{
		src: '/marketing/dashboard-phone.jpg',
		alt: 'The overview page on a phone',
		caption: 'Works on your phone',
		...phone
	},
	{
		src: '/marketing/artworks-phone.jpg',
		alt: 'The artworks grid on a phone',
		caption: 'Browse anywhere',
		...phone
	},
	{
		src: '/marketing/artwork-new.jpg',
		alt: 'The add-artwork form where only the title is required',
		caption: 'Add a piece in under a minute',
		...desktop
	},
	{
		src: '/marketing/login.jpg',
		alt: 'The sign-in page',
		caption: 'Simple sign-in',
		...desktop
	}
];
