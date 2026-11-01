import { env } from 'cloudflare:workers';
import { error } from '@sveltejs/kit';
import { canReadMedia } from '#lib/server/images.js';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params, locals, request }) => {
	const key = params.key;
	const viewerId = locals.user?.id ?? null;
	if (!(await canReadMedia(key, viewerId))) error(404, 'Not found');

	// Conditional requests are honoured; byte ranges are not (nothing here needs them, and a
	// 206 without Content-Range makes browsers drop images).
	const object = await env.MEDIA.get(key, { onlyIf: request.headers });
	if (!object) error(404, 'Not found');

	const headers = new Headers();
	object.writeHttpMetadata(headers);
	headers.set('ETag', object.httpEtag);
	// Keys are content-addressed per upload, so they never change. Owners can cache hard;
	// public copies get a shorter life so revoking visibility takes effect within a day.
	const isOwner = viewerId !== null && key.startsWith(`users/${viewerId}/`);
	headers.set(
		'Cache-Control',
		isOwner ? 'private, max-age=31536000, immutable' : 'public, max-age=86400'
	);

	if (!('body' in object) || !object.body) return new Response(null, { status: 304, headers });
	return new Response(object.body, { status: 200, headers });
};
