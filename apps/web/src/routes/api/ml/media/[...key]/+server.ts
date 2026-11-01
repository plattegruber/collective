import { error } from '@sveltejs/kit';
import { env } from 'cloudflare:workers';
import { authenticateApiToken } from '#lib/server/apiTokens.js';
import type { RequestHandler } from './$types';

/** Token-authenticated media download for the training pipeline (owner's objects only). */
export const GET: RequestHandler = async ({ request, params }) => {
	const auth = await authenticateApiToken(request);
	if (!auth) error(401, 'Invalid or missing API token.');
	if (!params.key.startsWith(`users/${auth.userId}/`)) error(404, 'Not found');

	const object = await env.MEDIA.get(params.key);
	if (!object) error(404, 'Not found');
	const headers = new Headers();
	object.writeHttpMetadata(headers);
	headers.set('ETag', object.httpEtag);
	headers.set('Cache-Control', 'private, no-store');
	return new Response(object.body, { headers });
};
