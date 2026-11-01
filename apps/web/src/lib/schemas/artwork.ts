import { z } from 'zod';
import { ARTWORK_STATUSES, DIMENSION_UNITS } from '#lib/domain/artwork.js';

export { STATUS_LABELS } from '#lib/domain/artwork.js';

const optionalText = (max: number) => z.string().trim().max(max).optional();

const optionalNumber = (message: string) => z.coerce.number({ error: message }).optional();

const optionalPositive = (message: string) =>
	optionalNumber(message).refine((value) => value === undefined || value >= 0, message);

const optionalDate = z
	.string()
	.regex(/^\d{4}-\d{2}-\d{2}$/, 'Use a full date (YYYY-MM-DD).')
	.optional();

/** Form-shaped input for creating or editing an artwork. */
export const artworkFormSchema = z.object({
	title: z.string('Give the piece a title.').trim().min(1, 'Give the piece a title.').max(200),
	artist: optionalText(200),
	year: optionalNumber('Enter a four-digit year.').refine(
		(value) => value === undefined || (Number.isInteger(value) && value >= 0 && value <= 2100),
		'Enter a four-digit year.'
	),
	dateText: optionalText(80),
	medium: optionalText(200),
	height: optionalPositive('Enter a positive number.'),
	width: optionalPositive('Enter a positive number.'),
	depth: optionalPositive('Enter a positive number.'),
	dimensionUnit: z.enum(DIMENSION_UNITS).default('cm'),
	description: optionalText(5000),
	status: z.enum(ARTWORK_STATUSES).default('owned'),
	location: optionalText(200),
	acquiredOn: optionalDate,
	acquiredFrom: optionalText(200),
	purchasePrice: z.string().trim().max(32).optional(),
	estimatedValue: z.string().trim().max(32).optional(),
	currency: z
		.string()
		.trim()
		.toUpperCase()
		.regex(/^[A-Z]{3}$/, 'Use a three-letter currency code.')
		.default('USD'),
	provenance: optionalText(5000),
	notes: optionalText(5000),
	tags: optionalText(500),
	isPublic: z
		.enum(['on', 'true', 'false'])
		.optional()
		.transform((value) => value === 'on' || value === 'true')
});

export type ArtworkFormInput = z.input<typeof artworkFormSchema>;
export type ArtworkFormData = z.output<typeof artworkFormSchema>;

export function parseTagList(value: string | undefined): string[] {
	if (!value) return [];
	const seen = new Set<string>();
	for (const raw of value.split(',')) {
		const tag = raw.trim().replace(/\s+/g, ' ');
		if (tag) seen.add(tag.slice(0, 40));
	}
	return [...seen].slice(0, 20);
}
