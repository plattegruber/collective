import { describe, expect, it } from 'vitest';
import { safeNextPath } from './guard.js';

describe('safeNextPath', () => {
	it('keeps same-origin paths with their query string', () => {
		expect(safeNextPath('/app/artworks?status=lent')).toBe('/app/artworks?status=lent');
		expect(safeNextPath('/')).toBe('/');
	});

	it('falls back for empty, relative, absolute, and scheme-relative values', () => {
		for (const value of [
			null,
			undefined,
			'',
			'app',
			'https://evil.example',
			'//evil.example',
			'/\\evil.example',
			'/\\\\evil.example'
		]) {
			expect(safeNextPath(value)).toBe('/app');
		}
	});

	it('honours a custom fallback', () => {
		expect(safeNextPath('//evil.example', '/')).toBe('/');
	});
});
