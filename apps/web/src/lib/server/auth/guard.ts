import { redirect, type RequestEvent } from '@sveltejs/kit';
import type { SessionUser } from './session.js';

/** Returns the signed-in user or redirects to the login page, remembering where they were going. */
export function requireUser(event: Pick<RequestEvent, 'locals' | 'url'>): SessionUser {
	if (event.locals.user) return event.locals.user;
	const next = event.url.pathname + event.url.search;
	redirect(303, `/login?next=${encodeURIComponent(next)}`);
}

/**
 * Only allow redirecting back to same-origin app paths after login. Rejects scheme-relative
 * forms (`//host`, and `/\\host`, which browsers normalise to `//host`).
 */
export function safeNextPath(value: string | null | undefined, fallback = '/app'): string {
	if (!value || !/^\/(?![/\\])/.test(value)) return fallback;
	return value;
}
