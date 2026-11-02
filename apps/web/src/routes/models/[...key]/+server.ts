import { env } from 'cloudflare:workers';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

/**
 * Serves the global recognizer model files from R2. Keys are content-addressed by release
 * id, so responses are immutable and cache for a year; the browser also keeps a copy in the
 * Cache API (see lib/recognition/assets.ts).
 */
export const GET: RequestHandler = async ({ params, request }) => {
	const key = `models/${params.key}`;
	if (!/^models\/[A-Za-z0-9_-]+\/(detector|embedder)\.onnx$/.test(key)) error(404, 'Not found');
	const object = await env.MEDIA.get(key, {
		onlyIf: request.headers,
		range: request.headers
	});
	if (!object) error(404, 'Not found');
	const headers = new Headers();
	object.writeHttpMetadata(headers);
	headers.set('etag', object.httpEtag);
	headers.set('cache-control', 'public, max-age=31536000, immutable');
	headers.set('cross-origin-resource-policy', 'same-origin');
	if (!('body' in object)) return new Response(null, { status: 304, headers });
	headers.set('content-length', String(object.size));
	return new Response(object.body, { headers });
};
