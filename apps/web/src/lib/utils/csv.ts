/** Minimal RFC 4180 CSV writer. Values with commas, quotes or newlines are quoted. */
export function csvCell(value: unknown): string {
	if (value === null || value === undefined) return '';
	const text = value instanceof Date ? value.toISOString() : String(value);
	return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

/** Byte-order mark so Excel opens the file as UTF-8. */
const BOM = '\uFEFF';

export function toCsv(
	header: readonly string[],
	rows: ReadonlyArray<ReadonlyArray<unknown>>
): string {
	const lines = [header.map(csvCell).join(',')];
	for (const row of rows) lines.push(row.map(csvCell).join(','));
	return `${BOM}${lines.join('\r\n')}\r\n`;
}
