import type { Box } from './types.js';

export function boxArea(box: Box): number {
	return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

/** Fraction of `inner` that lies inside `outer`. */
export function containment(inner: Box, outer: Box): number {
	const ix = Math.max(0, Math.min(inner.x2, outer.x2) - Math.max(inner.x1, outer.x1));
	const iy = Math.max(0, Math.min(inner.y2, outer.y2) - Math.max(inner.y1, outer.y1));
	const area = boxArea(inner);
	return area <= 0 ? 0 : (ix * iy) / area;
}

export function boxIoU(a: Box, b: Box): number {
	const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
	const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
	const inter = ix * iy;
	const union = boxArea(a) + boxArea(b) - inter;
	return union <= 0 ? 0 : inter / union;
}

/** Grows a box by `margin` × its size on every side, clamped to the frame. */
export function expandBox(box: Box, margin: number, width: number, height: number): Box {
	const dx = (box.x2 - box.x1) * margin;
	const dy = (box.y2 - box.y1) * margin;
	return {
		x1: Math.max(0, box.x1 - dx),
		y1: Math.max(0, box.y1 - dy),
		x2: Math.min(width, box.x2 + dx),
		y2: Math.min(height, box.y2 + dy)
	};
}

export function lerpBox(from: Box, to: Box, t: number): Box {
	return {
		x1: from.x1 + (to.x1 - from.x1) * t,
		y1: from.y1 + (to.y1 - from.y1) * t,
		x2: from.x2 + (to.x2 - from.x2) * t,
		y2: from.y2 + (to.y2 - from.y2) * t
	};
}

export function normalizeBox(box: Box, width: number, height: number): Box {
	return { x1: box.x1 / width, y1: box.y1 / height, x2: box.x2 / width, y2: box.y2 / height };
}

export function unionBox(boxes: readonly Box[]): Box {
	if (boxes.length === 0) throw new RangeError('Cannot union zero boxes.');
	return {
		x1: Math.min(...boxes.map((b) => b.x1)),
		y1: Math.min(...boxes.map((b) => b.y1)),
		x2: Math.max(...boxes.map((b) => b.x2)),
		y2: Math.max(...boxes.map((b) => b.y2))
	};
}

export interface ChooseRegionOptions {
	/** Detections smaller than this fraction of the image area are ignored. */
	minRegionFraction: number;
	/** The union must cover at least this fraction of the image to be used. */
	minCoverage: number;
}

export const DEFAULT_CHOOSE_REGION_OPTIONS: ChooseRegionOptions = {
	minRegionFraction: 0.02,
	minCoverage: 0.2
};

/**
 * Where the artwork is in an uploaded photo: the union of every confident, non-trivial
 * detection (so multi-panel works stay together), when that union covers a meaningful part
 * of the photo. Returns null to mean "use the whole photo". Mirrors `locate_artwork` in
 * collective-ml so evaluation measures what the browser does.
 */
export function chooseArtworkRegion(
	regions: readonly { box: Box }[],
	width: number,
	height: number,
	options: ChooseRegionOptions = DEFAULT_CHOOSE_REGION_OPTIONS
): Box | null {
	const area = width * height;
	const boxes = regions
		.map((region) => region.box)
		.filter((box) => boxArea(box) >= options.minRegionFraction * area);
	if (boxes.length === 0) return null;
	const union = unionBox(boxes);
	return boxArea(union) >= options.minCoverage * area ? union : null;
}
