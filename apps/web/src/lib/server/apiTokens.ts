import { and, eq } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { bytesToBase64Url, sha256Hex } from '#lib/utils/encoding.js';

const TOKEN_PREFIX = 'clv_';

/** Creates a bearer token for the ML pipeline. The raw token is only ever returned here. */
export async function createApiToken(
	userId: string,
	name: string
): Promise<{ token: string; id: string }> {
	const secret = bytesToBase64Url(crypto.getRandomValues(new Uint8Array(32)));
	const token = `${TOKEN_PREFIX}${secret}`;
	const id = await sha256Hex(token);
	await db()
		.insert(schema.apiTokens)
		.values({
			id,
			userId,
			name: name.trim().slice(0, 60) || 'Untitled token',
			prefix: token.slice(0, 10)
		});
	return { token, id };
}

export async function listApiTokens(userId: string) {
	return db().query.apiTokens.findMany({
		where: eq(schema.apiTokens.userId, userId),
		columns: { id: true, name: true, prefix: true, lastUsedAt: true, createdAt: true },
		orderBy: (t, { desc }) => desc(t.createdAt)
	});
}

export async function revokeApiToken(userId: string, id: string): Promise<boolean> {
	const result = await db()
		.delete(schema.apiTokens)
		.where(and(eq(schema.apiTokens.id, id), eq(schema.apiTokens.userId, userId)))
		.returning({ id: schema.apiTokens.id });
	return result.length > 0;
}

/** Resolves `Authorization: Bearer clv_…` to a user id, or null. */
export async function authenticateApiToken(
	request: Request
): Promise<{ userId: string; tokenId: string } | null> {
	const header = request.headers.get('authorization') ?? '';
	const [scheme, token] = header.split(' ');
	if (scheme?.toLowerCase() !== 'bearer' || !token?.startsWith(TOKEN_PREFIX)) return null;
	const id = await sha256Hex(token);
	const row = await db().query.apiTokens.findFirst({ where: eq(schema.apiTokens.id, id) });
	if (!row) return null;
	await db()
		.update(schema.apiTokens)
		.set({ lastUsedAt: new Date() })
		.where(eq(schema.apiTokens.id, id));
	return { userId: row.userId, tokenId: id };
}
