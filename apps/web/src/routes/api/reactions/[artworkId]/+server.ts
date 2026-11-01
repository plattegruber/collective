import { error } from '@sveltejs/kit';
import { z } from 'zod';
import { isReactionEmoji } from '#lib/reactions/shared.js';
import { canReact, react, reactionCounts } from '#lib/server/reactions.js';
import type { RequestHandler } from './$types';

const bodySchema = z.object({
	emoji: z.string(),
	visitor: z.string().min(8).max(64),
	/** Honeypot: real clients always send an empty string. */
	website: z.string().optional()
});

export const GET: RequestHandler = async ({ params, locals }) => {
	if (!(await canReact(params.artworkId, locals.user?.id ?? null))) error(404, 'Not found');
	return Response.json(await reactionCounts(params.artworkId), {
		headers: { 'cache-control': 'public, max-age=5' }
	});
};

export const POST: RequestHandler = async ({ params, locals, request, getClientAddress }) => {
	if (!(await canReact(params.artworkId, locals.user?.id ?? null))) error(404, 'Not found');
	const parsed = bodySchema.safeParse(await request.json().catch(() => null));
	if (!parsed.success) error(400, 'Invalid reaction.');
	if (parsed.data.website)
		return Response.json({ ok: true, counts: await reactionCounts(params.artworkId) });
	if (!isReactionEmoji(parsed.data.emoji)) error(400, 'Unknown reaction.');

	const outcome = await react(
		params.artworkId,
		parsed.data.emoji,
		parsed.data.visitor,
		getClientAddress()
	);
	if (outcome.status === 'rate_limited') error(429, 'Slow down a little.');
	return Response.json({
		ok: true,
		throttled: outcome.status === 'throttled',
		counts: await reactionCounts(params.artworkId)
	});
};
