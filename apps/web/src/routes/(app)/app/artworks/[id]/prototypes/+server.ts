import { error } from '@sveltejs/kit';
import { z } from 'zod';
import { base64UrlToBytes } from '#lib/utils/encoding.js';
import {
	RecognitionError,
	decodeEmbedding,
	savePhotoPrototype,
	saveScanMemory
} from '#lib/server/recognition.js';
import type { RequestHandler } from './$types';

const bodySchema = z.object({
	releaseId: z.string().min(1),
	/** base64url of little-endian float32 values. */
	embedding: z.string().min(8),
	source: z.enum(['photo', 'scan']),
	imageId: z.string().min(1).optional(),
	box: z
		.object({ x1: z.number(), y1: z.number(), x2: z.number(), y2: z.number() })
		.nullable()
		.optional()
});

/**
 * The browser computed an embedding (after an upload, during a rebuild, or while scanning)
 * and hands it over. Owner only.
 */
export const POST: RequestHandler = async ({ request, params, locals }) => {
	if (!locals.user) error(401, 'Sign in first.');
	const parsed = bodySchema.safeParse(await request.json().catch(() => null));
	if (!parsed.success) error(400, 'Invalid prototype payload.');
	const { releaseId, source, imageId, box } = parsed.data;
	let embedding: Float32Array;
	try {
		embedding = decodeEmbedding(base64UrlToBytes(parsed.data.embedding));
	} catch {
		error(400, 'Embedding is not valid base64url.');
	}
	try {
		if (source === 'photo') {
			if (!imageId) error(400, 'imageId is required for photo prototypes.');
			const row = await savePhotoPrototype(locals.user.id, params.id, {
				imageId,
				releaseId,
				embedding,
				box: box ?? null
			});
			return Response.json({ id: row.id, saved: true }, { status: 201 });
		}
		const result = await saveScanMemory(locals.user.id, params.id, { releaseId, embedding });
		return Response.json(result, { status: result.saved ? 201 : 200 });
	} catch (cause) {
		if (cause instanceof RecognitionError)
			error(cause.code === 'not_found' ? 404 : 400, cause.message);
		throw cause;
	}
};
