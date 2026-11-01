import { env } from 'cloudflare:workers';

/**
 * Fixed-window counter in KV. Returns false once `max` increments happen inside a window.
 * KV is eventually consistent, so treat this as abuse damping rather than a hard quota.
 */
export async function bumpCounter(key: string, ttlSeconds: number, max: number): Promise<boolean> {
	const current = Number(await env.REACTIONS.get(key)) || 0;
	if (current >= max) return false;
	await env.REACTIONS.put(key, String(current + 1), { expirationTtl: Math.max(60, ttlSeconds) });
	return true;
}

export function minuteBucket(now = Date.now()): number {
	return Math.floor(now / 60_000);
}

export function dayStamp(now = new Date()): string {
	return now.toISOString().slice(0, 10);
}
