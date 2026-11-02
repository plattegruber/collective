import { error } from '@sveltejs/kit';
import { activeRelease, indexStatus, releaseManifest } from '#lib/server/recognition.js';
import type { RequestHandler } from './$types';

/** What the owner's browser needs to (re)build the recognition index: the models and the photos still missing a prototype. */
export const GET: RequestHandler = async ({ locals }) => {
	if (!locals.user) error(401, 'Sign in first.');
	const release = await activeRelease();
	if (!release) return Response.json({ manifest: null, status: null });
	const status = await indexStatus(locals.user.id, release.id);
	return Response.json({ manifest: releaseManifest(release), status });
};
