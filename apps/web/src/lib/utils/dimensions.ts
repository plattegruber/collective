import type { DimensionUnit } from '#lib/domain/artwork.js';

const CM_PER_INCH = 2.54;

export function toCm(value: number, unit: DimensionUnit): number {
	return unit === 'in' ? value * CM_PER_INCH : value;
}

export function fromCm(cm: number, unit: DimensionUnit): number {
	return unit === 'in' ? cm / CM_PER_INCH : cm;
}

/** Rounds to a sensible precision for the unit and drops trailing zeros. */
export function formatLength(cm: number, unit: DimensionUnit): string {
	const value = fromCm(cm, unit);
	return Number(value.toFixed(unit === 'in' ? 2 : 1)).toString();
}

/** "61 × 45.7 cm" or "24 × 18 × 1.5 in". Returns an empty string when nothing is set. */
export function formatDimensions(
	dims: { heightCm: number | null; widthCm: number | null; depthCm: number | null },
	unit: DimensionUnit
): string {
	const parts = [dims.heightCm, dims.widthCm, dims.depthCm]
		.filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
		.map((value) => formatLength(value, unit));
	if (parts.length === 0) return '';
	return `${parts.join(' × ')} ${unit}`;
}
