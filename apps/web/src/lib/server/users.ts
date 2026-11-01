import { env } from 'cloudflare:workers';
import { eq } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { hashPassword } from './auth/password.js';
import { newId } from '#lib/utils/id.js';

export function normalizeEmail(email: string): string {
	return email.trim().toLowerCase();
}

export async function findUserByEmail(email: string) {
	return db().query.users.findFirst({ where: eq(schema.users.email, normalizeEmail(email)) });
}

export async function findUserByHandle(handle: string) {
	return db().query.users.findFirst({ where: eq(schema.users.handle, handle.toLowerCase()) });
}

export async function createUser(input: { email: string; password: string; name: string }) {
	const id = newId();
	const now = new Date();
	await db()
		.insert(schema.users)
		.values({
			id,
			email: normalizeEmail(input.email),
			passwordHash: await hashPassword(input.password),
			name: input.name.trim(),
			updatedAt: now
		});
	return id;
}

/** Removes every object the collector owns from R2, then the user row (rows cascade). */
export async function deleteUserAndData(userId: string): Promise<void> {
	const prefix = `users/${userId}/`;
	let cursor: string | undefined;
	do {
		const listed = await env.MEDIA.list({ prefix, cursor });
		if (listed.objects.length > 0)
			await env.MEDIA.delete(listed.objects.map((object) => object.key));
		cursor = listed.truncated ? listed.cursor : undefined;
	} while (cursor);
	await db().delete(schema.users).where(eq(schema.users.id, userId));
}
