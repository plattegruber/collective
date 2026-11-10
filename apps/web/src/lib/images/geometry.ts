/**
 * Plane geometry for straightening photos: the four corners of a piece on the wall and the
 * projective map that turns them into an upright rectangle.
 */

export interface Point {
	x: number;
	y: number;
}

/** Corners in order: top-left, top-right, bottom-right, bottom-left. */
export type Quad = [Point, Point, Point, Point];

/** Row-major 3×3 matrix with the last entry normalised to 1. */
export type Homography = Float64Array;

export interface Size {
	width: number;
	height: number;
}

function distance(a: Point, b: Point): number {
	return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Orders arbitrary corner points clockwise from the top-left. */
export function orderQuad(points: readonly Point[]): Quad {
	if (points.length !== 4) throw new RangeError('A quad has four corners.');
	const cx = points.reduce((sum, p) => sum + p.x, 0) / 4;
	const cy = points.reduce((sum, p) => sum + p.y, 0) / 4;
	const sorted = [...points].sort(
		(a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
	);
	// atan2 runs from -π at the left, so the first point after sorting sits in the upper-left
	// half-plane when the quad is roughly upright; rotate until top-left leads.
	let start = 0;
	let best = Infinity;
	sorted.forEach((p, i) => {
		const score = p.x + p.y;
		if (score < best) {
			best = score;
			start = i;
		}
	});
	return [0, 1, 2, 3].map((i) => sorted[(start + i) % 4]!) as Quad;
}

export function quadArea(quad: Quad): number {
	let twice = 0;
	for (let i = 0; i < 4; i += 1) {
		const a = quad[i]!;
		const b = quad[(i + 1) % 4]!;
		twice += a.x * b.y - b.x * a.y;
	}
	return Math.abs(twice) / 2;
}

export function quadIsConvex(quad: Quad): boolean {
	let sign = 0;
	for (let i = 0; i < 4; i += 1) {
		const a = quad[i]!;
		const b = quad[(i + 1) % 4]!;
		const c = quad[(i + 2) % 4]!;
		const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
		if (cross === 0) return false;
		const s = Math.sign(cross);
		if (sign === 0) sign = s;
		else if (s !== sign) return false;
	}
	return true;
}

export function boxToQuad(x1: number, y1: number, x2: number, y2: number): Quad {
	return [
		{ x: x1, y: y1 },
		{ x: x2, y: y1 },
		{ x: x2, y: y2 },
		{ x: x1, y: y2 }
	];
}

/** The axis-aligned bounds of a quad, as a quad: a crop without any straightening. */
export function boundsQuad(quad: Quad): Quad {
	const xs = quad.map((p) => p.x);
	const ys = quad.map((p) => p.y);
	return boxToQuad(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys));
}

/**
 * A quad carried from one picture to another of the same piece: each corner keeps its
 * position relative to `from` (where the piece was found in the first picture) inside
 * `to` (where it was found in the second), so it lands on the same physical edges.
 */
export function transferQuad(quad: Quad, from: Quad, to: Quad): Quad {
	const [fa, , fc] = from;
	const [ta, , tc] = to;
	const fw = Math.max(1e-6, fc.x - fa.x);
	const fh = Math.max(1e-6, fc.y - fa.y);
	return quad.map((corner) => ({
		x: ta.x + ((corner.x - fa.x) / fw) * (tc.x - ta.x),
		y: ta.y + ((corner.y - fa.y) / fh) * (tc.y - ta.y)
	})) as Quad;
}

/**
 * The upright size a quad should be warped to. Side lengths are averaged (what document
 * scanners do); when the true aspect ratio is known (the collector recorded the piece's
 * dimensions) it wins, keeping the pixel budget of the longer side.
 */
export function rectifiedSize(quad: Quad, maxEdge: number, aspect?: number | null): Size {
	const [tl, tr, br, bl] = quad;
	let width = Math.max(distance(tl, tr), distance(bl, br));
	let height = Math.max(distance(tl, bl), distance(tr, br));
	if (aspect && Number.isFinite(aspect) && aspect > 0) {
		if (width / height > aspect) height = width / aspect;
		else width = height * aspect;
	}
	const scale = Math.min(1, maxEdge / Math.max(width, height));
	return {
		width: Math.max(1, Math.round(width * scale)),
		height: Math.max(1, Math.round(height * scale))
	};
}

/** Gaussian elimination with partial pivoting on an augmented n×(n+1) system. */
function solveLinear(rows: number[][]): number[] {
	const n = rows.length;
	for (let col = 0; col < n; col += 1) {
		let pivot = col;
		for (let r = col + 1; r < n; r += 1) {
			if (Math.abs(rows[r]![col]!) > Math.abs(rows[pivot]![col]!)) pivot = r;
		}
		if (Math.abs(rows[pivot]![col]!) < 1e-12) throw new RangeError('Degenerate quad.');
		[rows[col], rows[pivot]] = [rows[pivot]!, rows[col]!];
		const head = rows[col]!;
		for (let r = 0; r < n; r += 1) {
			if (r === col) continue;
			const row = rows[r]!;
			const factor = row[col]! / head[col]!;
			if (factor === 0) continue;
			for (let c = col; c <= n; c += 1) row[c] = row[c]! - factor * head[c]!;
		}
	}
	return rows.map((row, i) => row[n]! / row[i]!);
}

/** The homography mapping each `from` corner onto the matching `to` corner. */
export function solveHomography(from: Quad, to: Quad): Homography {
	const rows: number[][] = [];
	for (let i = 0; i < 4; i += 1) {
		const { x, y } = from[i]!;
		const { x: u, y: v } = to[i]!;
		rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
		rows.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
	}
	const h = solveLinear(rows);
	return Float64Array.from([...h, 1]);
}

export function applyHomography(h: Homography, p: Point): Point {
	const w = h[6]! * p.x + h[7]! * p.y + h[8]!;
	return {
		x: (h[0]! * p.x + h[1]! * p.y + h[2]!) / w,
		y: (h[3]! * p.x + h[4]! * p.y + h[5]!) / w
	};
}

/** Intersection of the lines through (a, b) and (c, d); null when parallel. */
export function intersectLines(a: Point, b: Point, c: Point, d: Point): Point | null {
	const r = { x: b.x - a.x, y: b.y - a.y };
	const s = { x: d.x - c.x, y: d.y - c.y };
	const denominator = r.x * s.y - r.y * s.x;
	if (Math.abs(denominator) < 1e-9) return null;
	const t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / denominator;
	return { x: a.x + t * r.x, y: a.y + t * r.y };
}
