const ALPHABET = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';

/**
 * URL-safe, collision-resistant identifier (21 chars of base-62, ~125 bits).
 * Uses rejection sampling so every character is equally likely.
 */
export function newId(length = 21): string {
	let out = '';
	const bytes = new Uint8Array(length * 2);
	while (out.length < length) {
		crypto.getRandomValues(bytes);
		for (const b of bytes) {
			if (b < 248 && out.length < length) out += ALPHABET[b % 62];
		}
	}
	return out;
}
