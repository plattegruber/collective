import { type ReactionCount, type ReactionEmoji } from './shared.js';

const STORAGE_KEY = 'collective.visitor';

/** Stable anonymous visitor id so one guest cannot spam a single piece. */
export function visitorId(): string {
	if (typeof window === 'undefined') return 'server';
	try {
		const existing = window.localStorage.getItem(STORAGE_KEY);
		if (existing) return existing;
		const id = crypto.randomUUID();
		window.localStorage.setItem(STORAGE_KEY, id);
		return id;
	} catch {
		return crypto.randomUUID();
	}
}

export async function fetchReactionCounts(artworkId: string): Promise<ReactionCount[]> {
	const response = await fetch(`/api/reactions/${encodeURIComponent(artworkId)}`);
	if (!response.ok) return [];
	return (await response.json()) as ReactionCount[];
}

export async function sendReaction(
	artworkId: string,
	emoji: ReactionEmoji
): Promise<{ ok: boolean; throttled: boolean; counts?: ReactionCount[] }> {
	const response = await fetch(`/api/reactions/${encodeURIComponent(artworkId)}`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({ emoji, visitor: visitorId(), website: '' })
	});
	const payload = (await response.json().catch(() => null)) as {
		message?: string;
		throttled?: boolean;
		counts?: ReactionCount[];
	} | null;
	if (!response.ok)
		throw new Error(payload?.message ?? `Could not send reaction (${response.status}).`);
	return { ok: true, throttled: Boolean(payload?.throttled), counts: payload?.counts };
}
