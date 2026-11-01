import { describe, expect, it } from 'vitest';
import { csvCell, toCsv } from './csv.js';

describe('csv', () => {
	it('quotes only when needed and escapes quotes', () => {
		expect(csvCell('plain')).toBe('plain');
		expect(csvCell('a, b')).toBe('"a, b"');
		expect(csvCell('say "hi"')).toBe('"say ""hi"""');
		expect(csvCell('line\nbreak')).toBe('"line\nbreak"');
		expect(csvCell(null)).toBe('');
		expect(csvCell(12.5)).toBe('12.5');
	});

	it('writes a BOM, CRLF line endings and a trailing newline', () => {
		expect(toCsv(['a', 'b'], [[1, 'x,y']])).toBe('\uFEFFa,b\r\n1,"x,y"\r\n');
	});
});
