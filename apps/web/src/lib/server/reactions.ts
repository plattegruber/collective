import { env } from 'cloudflare:workers';
import { eq } from 'drizzle-orm';
import { REACTION_EMOJIS, type ReactionCount, type ReactionEmoji } from '#lib/reactions/shared.js';
import { bumpCounter, dayStamp, minuteBucket } from './kv.js';
import { db, schema } from './db/index.js';

const countKey = (artworkId: string, emoji: string) => `count:${artworkId}:${emoji}`;
const seenKey = (artworkId: string, visitor: string) => `seen:${artworkId}:${visitor}`;

export async function reactionCounts(artworkId: string): Promise<ReactionCount[]> {
	const values = await Promise.all(
		REACTION_EMOJIS.map((emoji) => env.REACTIONS.get(countKey(artworkId, emoji)))
	);
	return REACTION_EMOJIS.map((emoji, i) => ({ emoji, count: Number(values[i] ?? 0) }));
}

/** Who may react: anyone, as long as the piece is publicly visible or the viewer owns it. */
export async function canReact(artworkId: string, viewerId: string | null): Promise<boolean> {
	const artwork = await db().query.artworks.findFirst({
		where: eq(schema.artworks.id, artworkId),
		columns: { userId: true, isPublic: true },
		with: { owner: { columns: { galleryPublic: true } } }
	});
	if (!artwork) return false;
	if (viewerId && viewerId === artwork.userId) return true;
	return artwork.isPublic && artwork.owner.galleryPublic;
}

export type ReactOutcome = { status: 'ok' } | { status: 'throttled' } | { status: 'rate_limited' };

export async function react(
	artworkId: string,
	emoji: ReactionEmoji,
	visitor: string,
	ip: string
): Promise<ReactOutcome> {
	const seen = seenKey(artworkId, visitor);
	const last = Number(await env.REACTIONS.get(seen));
	if (Number.isFinite(last) && last > 0 && Date.now() - last < 1000) return { status: 'throttled' };
	await env.REACTIONS.put(seen, String(Date.now()), { expirationTtl: 120 });

	const minuteOk = await bumpCounter(`rl:react:${ip}:${minuteBucket()}`, 70, 10);
	const dayOk = await bumpCounter(`rl:reactday:${ip}:${dayStamp()}`, 60 * 60 * 24, 200);
	if (!minuteOk || !dayOk) return { status: 'rate_limited' };

	const key = countKey(artworkId, emoji);
	const current = Number((await env.REACTIONS.get(key)) ?? 0);
	await env.REACTIONS.put(key, String(current + 1));
	return { status: 'ok' };
}
