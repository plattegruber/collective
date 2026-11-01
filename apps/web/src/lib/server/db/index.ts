import { env } from 'cloudflare:workers';
import { drizzle } from 'drizzle-orm/d1';
import * as schema from './schema.js';

export type Database = ReturnType<typeof createDb>;

export function createDb(d1: D1Database) {
	return drizzle(d1, { schema, casing: 'snake_case' });
}

let cached: Database | undefined;

/** The request-scoped Drizzle client bound to the D1 database. */
export function db(): Database {
	cached ??= createDb(env.DB);
	return cached;
}

export { schema };
