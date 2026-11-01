export function formatDate(value: string | Date | null | undefined, locale = 'en-US'): string {
	if (!value) return '';
	const date = typeof value === 'string' ? new Date(`${value}T00:00:00`) : value;
	if (Number.isNaN(date.getTime())) return '';
	return new Intl.DateTimeFormat(locale, {
		year: 'numeric',
		month: 'short',
		day: 'numeric'
	}).format(date);
}

export function formatRelative(date: Date, now = new Date(), locale = 'en-US'): string {
	const diff = date.getTime() - now.getTime();
	const abs = Math.abs(diff);
	const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
	const minute = 60_000;
	const hour = 60 * minute;
	const day = 24 * hour;
	if (abs < hour) return rtf.format(Math.round(diff / minute), 'minute');
	if (abs < day) return rtf.format(Math.round(diff / hour), 'hour');
	if (abs < 30 * day) return rtf.format(Math.round(diff / day), 'day');
	return formatDate(date, locale);
}

/** "Artist, 1998" / "Artist" / "1998" / "" */
export function byline(
	artist: string | null | undefined,
	year: number | null | undefined,
	dateText?: string | null
): string {
	const when = dateText?.trim() || (year ? String(year) : '');
	return [artist?.trim(), when].filter(Boolean).join(', ');
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
	return `${count} ${count === 1 ? singular : plural}`;
}
