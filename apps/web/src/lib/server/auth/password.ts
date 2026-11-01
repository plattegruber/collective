import { base64UrlToBytes, bytesToBase64Url, timingSafeEqual } from '#lib/utils/encoding.js';

/**
 * PBKDF2-SHA256 via WebCrypto. It is the only KDF available natively on Workers;
 * the iteration count is tuned to stay well inside the CPU budget per request.
 */
const ALGORITHM = 'pbkdf2-sha256';
const ITERATIONS = 100_000;
const KEY_BYTES = 32;
const SALT_BYTES = 16;

async function derive(password: string, salt: Uint8Array, iterations: number): Promise<Uint8Array> {
	const key = await crypto.subtle.importKey(
		'raw',
		new TextEncoder().encode(password),
		'PBKDF2',
		false,
		['deriveBits']
	);
	const bits = await crypto.subtle.deriveBits(
		{ name: 'PBKDF2', hash: 'SHA-256', salt: salt as BufferSource, iterations },
		key,
		KEY_BYTES * 8
	);
	return new Uint8Array(bits);
}

/** Returns `pbkdf2-sha256$<iterations>$<salt>$<hash>` with base64url parts. */
export async function hashPassword(password: string): Promise<string> {
	const salt = crypto.getRandomValues(new Uint8Array(SALT_BYTES));
	const hash = await derive(password, salt, ITERATIONS);
	return [ALGORITHM, ITERATIONS, bytesToBase64Url(salt), bytesToBase64Url(hash)].join('$');
}

export async function verifyPassword(password: string, stored: string): Promise<boolean> {
	const [algorithm, iterationsText, saltText, hashText] = stored.split('$');
	if (algorithm !== ALGORITHM || !iterationsText || !saltText || !hashText) return false;
	const iterations = Number(iterationsText);
	if (!Number.isInteger(iterations) || iterations < 1) return false;
	const expected = base64UrlToBytes(hashText);
	const actual = await derive(password, base64UrlToBytes(saltText), iterations);
	return timingSafeEqual(actual, expected);
}

/** True when a stored hash was produced with weaker parameters than we use today. */
export function needsRehash(stored: string): boolean {
	const [algorithm, iterationsText] = stored.split('$');
	return algorithm !== ALGORITHM || Number(iterationsText) < ITERATIONS;
}
