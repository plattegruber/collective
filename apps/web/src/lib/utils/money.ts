/** Amounts are stored as integer minor units (e.g. cents) alongside an ISO-4217 currency code. */

const minorDigitsCache = new Map<string, number>();

export function minorDigits(currency: string): number {
	let digits = minorDigitsCache.get(currency);
	if (digits === undefined) {
		try {
			digits =
				new Intl.NumberFormat('en', { style: 'currency', currency }).resolvedOptions()
					.maximumFractionDigits ?? 2;
		} catch {
			digits = 2;
		}
		minorDigitsCache.set(currency, digits);
	}
	return digits;
}

export function toMinor(amount: number, currency: string): number {
	return Math.round(amount * 10 ** minorDigits(currency));
}

export function fromMinor(minor: number, currency: string): number {
	return minor / 10 ** minorDigits(currency);
}

export function formatMoney(
	minor: number | null | undefined,
	currency: string,
	options: { compact?: boolean; locale?: string } = {}
): string {
	if (minor === null || minor === undefined) return '—';
	const { compact = false, locale = 'en-US' } = options;
	return new Intl.NumberFormat(locale, {
		style: 'currency',
		currency,
		notation: compact ? 'compact' : 'standard',
		maximumFractionDigits: compact ? 1 : minorDigits(currency)
	}).format(fromMinor(minor, currency));
}

/** Parses user input like "1,250.00" or "$1250" into minor units. Returns null for blank input. */
export function parseMoneyInput(
	value: string | undefined | null,
	currency: string
): number | null | undefined {
	if (value === undefined || value === null) return null;
	const cleaned = value.replace(/[^0-9.-]/g, '');
	if (cleaned === '') return null;
	const amount = Number(cleaned);
	if (!Number.isFinite(amount)) return undefined;
	return toMinor(amount, currency);
}

export const COMMON_CURRENCIES = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY', 'CHF', 'MXN'] as const;
