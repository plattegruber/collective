import { error } from '@sveltejs/kit';
import { activateRelease, isReleaseAuthorized } from '#lib/server/recognition.js';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request, params }) => {
	if (!isReleaseAuthorized(request)) error(401, 'A valid release token is required.');
	if (!(await activateRelease(params.id))) error(404, 'Release not found.');
	return Response.json({ ok: true, id: params.id });
};
