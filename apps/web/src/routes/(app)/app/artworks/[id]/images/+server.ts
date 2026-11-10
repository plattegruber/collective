import { error } from '@sveltejs/kit';
import {
	ALLOWED_IMAGE_TYPES,
	ALLOWED_ORIGINAL_TYPES,
	ImageError,
	MAX_IMAGE_BYTES,
	MAX_ORIGINAL_BYTES,
	addArtworkImage
} from '#lib/server/images.js';
import type { RequestHandler } from './$types';

function requireFile(formData: FormData, name: string): File {
	const value = formData.get(name);
	if (!(value instanceof File) || value.size === 0) error(400, `Missing ${name} image.`);
	if (!ALLOWED_IMAGE_TYPES.has(value.type)) error(415, 'Use a JPEG, PNG, or WebP image.');
	if (value.size > MAX_IMAGE_BYTES) error(413, 'Image is too large (12 MB max).');
	return value;
}

/** The untouched upload is optional and may be a format browsers cannot decode (HEIC). */
function optionalOriginal(formData: FormData): File | null {
	const value = formData.get('original');
	if (!(value instanceof File) || value.size === 0) return null;
	if (!ALLOWED_ORIGINAL_TYPES.has(value.type)) error(415, 'Unsupported original image type.');
	if (value.size > MAX_ORIGINAL_BYTES) error(413, 'Original is too large (30 MB max).');
	return value;
}

function requireDimension(formData: FormData, name: string): number {
	const value = Number(formData.get(name));
	if (!Number.isInteger(value) || value <= 0 || value > 20_000) error(400, `Invalid ${name}.`);
	return value;
}

export const POST: RequestHandler = async ({ request, params, locals }) => {
	if (!locals.user) error(401, 'Sign in to upload.');
	const formData = await request.formData();
	const full = requireFile(formData, 'full');
	const thumb = requireFile(formData, 'thumb');
	const original = optionalOriginal(formData);
	const width = requireDimension(formData, 'width');
	const height = requireDimension(formData, 'height');

	try {
		const image = await addArtworkImage(locals.user.id, params.id, {
			full: { bytes: await full.arrayBuffer(), contentType: full.type, width, height },
			thumb: { bytes: await thumb.arrayBuffer(), contentType: thumb.type },
			original: original
				? { bytes: await original.arrayBuffer(), contentType: original.type }
				: undefined
		});
		return Response.json(
			{ id: image.id, url: `/media/${image.key}`, thumbUrl: `/media/${image.thumbKey}` },
			{ status: 201 }
		);
	} catch (cause) {
		if (cause instanceof ImageError) error(cause.code === 'not_found' ? 404 : 400, cause.message);
		throw cause;
	}
};
