import { describe, expect, it } from 'vitest';

// canReadMedia touches the database; the key grammar it relies on is tested here in isolation.
const KEY = /^users\/([^/]+)\/(artworks|models)\/([^/]+)\//;

describe('media key grammar', () => {
	it('recognises artwork and model keys', () => {
		expect(KEY.exec('users/u1/artworks/a1/img.jpg')?.slice(1)).toEqual(['u1', 'artworks', 'a1']);
		expect(KEY.exec('users/u1/models/m1/model.onnx')?.slice(1)).toEqual(['u1', 'models', 'm1']);
	});
	it('rejects anything else', () => {
		expect(KEY.exec('users/u1/other/x/y')).toBeNull();
		expect(KEY.exec('../users/u1/artworks/a1/x')).toBeNull();
	});
});
