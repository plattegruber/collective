import type { Cookies } from '@sveltejs/kit';
import { and, eq, ne } from 'drizzle-orm';
import { dev } from '$app/env';
import { db, schema } from '#lib/server/db/index.js';
import { bytesToBase64Url, sha256Hex } from '#lib/utils/encoding.js';

export const SESSION_COOKIE = 'collective_session';
const SESSION_TTL_MS = 1000 * 60 * 60 * 24 * 30;
/** Sliding expiry: extend when less than this much lifetime remains. */
const RENEW_THRESHOLD_MS = SESSION_TTL_MS / 2;

export type SessionUser = Pick<
	schema.User,
	| 'id'
	| 'email'
	| 'name'
	| 'handle'
	| 'galleryPublic'
	| 'galleryTitle'
	| 'galleryIntro'
	| 'createdAt'
>;

export function generateToken(): string {
	return bytesToBase64Url(crypto.getRandomValues(new Uint8Array(32)));
}

export async function createSession(userId: string): Promise<{ token: string; expiresAt: Date }> {
	const token = generateToken();
	const expiresAt = new Date(Date.now() + SESSION_TTL_MS);
	await db()
		.insert(schema.sessions)
		.values({ id: await sha256Hex(token), userId, expiresAt });
	return { token, expiresAt };
}

export async function validateSession(
	token: string
): Promise<{ user: SessionUser; sessionId: string; renewedUntil: Date | null } | null> {
	const sessionId = await sha256Hex(token);
	const row = await db()
		.select({
			expiresAt: schema.sessions.expiresAt,
			user: {
				id: schema.users.id,
				email: schema.users.email,
				name: schema.users.name,
				handle: schema.users.handle,
				galleryPublic: schema.users.galleryPublic,
				galleryTitle: schema.users.galleryTitle,
				galleryIntro: schema.users.galleryIntro,
				createdAt: schema.users.createdAt
			}
		})
		.from(schema.sessions)
		.innerJoin(schema.users, eq(schema.users.id, schema.sessions.userId))
		.where(eq(schema.sessions.id, sessionId))
		.get();

	if (!row) return null;
	const now = Date.now();
	if (row.expiresAt.getTime() <= now) {
		await db().delete(schema.sessions).where(eq(schema.sessions.id, sessionId));
		return null;
	}

	let renewedUntil: Date | null = null;
	if (row.expiresAt.getTime() - now < RENEW_THRESHOLD_MS) {
		renewedUntil = new Date(now + SESSION_TTL_MS);
		await db()
			.update(schema.sessions)
			.set({ expiresAt: renewedUntil })
			.where(eq(schema.sessions.id, sessionId));
	}

	return { user: row.user, sessionId, renewedUntil };
}

export async function invalidateSession(sessionId: string): Promise<void> {
	await db().delete(schema.sessions).where(eq(schema.sessions.id, sessionId));
}

export async function invalidateUserSessions(userId: string): Promise<void> {
	await db().delete(schema.sessions).where(eq(schema.sessions.userId, userId));
}

/** Signs out every other device while keeping the session that made the request. */
export async function invalidateOtherSessions(
	userId: string,
	keepSessionId: string
): Promise<void> {
	await db()
		.delete(schema.sessions)
		.where(and(eq(schema.sessions.userId, userId), ne(schema.sessions.id, keepSessionId)));
}

export function setSessionCookie(cookies: Cookies, token: string, expiresAt: Date): void {
	cookies.set(SESSION_COOKIE, token, {
		httpOnly: true,
		sameSite: 'lax',
		secure: !dev,
		expires: expiresAt,
		path: '/'
	});
}

export function clearSessionCookie(cookies: Cookies): void {
	cookies.delete(SESSION_COOKIE, { path: '/' });
}
