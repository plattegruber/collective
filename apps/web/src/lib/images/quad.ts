import type { Box } from '#lib/recognition/types.js';
import {
	boxToQuad,
	intersectLines,
	quadArea,
	quadIsConvex,
	type Point,
	type Quad
} from './geometry.js';
import { downscale, downscaleFactor, luminance, sobel, type Raster } from './raster.js';

export interface QuadDetection {
	quad: Quad;
	/** 0 when the corners are just the box; approaches 1 when all four edges are crisp. */
	confidence: number;
	/** Strength of the top, bottom, left and right edges (0..1); all 0 for the fallback. */
	sides: [number, number, number, number];
}

export interface FindQuadOptions {
	/** Long edge of the analysis image; edges are searched at this resolution. */
	analysisEdge: number;
	/** How far from each side of the box the edge may lie, as a fraction of the box size. */
	band: number;
	/** Maximum tilt of a side: endpoint offset difference over its length. */
	maxTilt: number;
	/** Detections whose weakest side is below this fall back to the box. */
	minConfidence: number;
	/** Which edge wins when a frame offers both: `outer` keeps the frame, `inner` crops to the art. */
	prefer: 'outer' | 'inner';
	/** Candidate edges kept per side for the joint choice. */
	candidates: number;
}

/** Weight of the outside-colour disagreement; wood against mat (~0.08) costs a 30% weaker edge. */
const CONSISTENCY = 4;

export const DEFAULT_FIND_QUAD_OPTIONS: FindQuadOptions = {
	analysisEdge: 960,
	band: 0.22,
	maxTilt: 0.15,
	minConfidence: 0.3,
	prefer: 'outer',
	candidates: 4
};

interface Candidate {
	a: Point;
	b: Point;
	/** Mean gradient across the edge. */
	score: number;
	/** Mean perpendicular offset from the box side, positive towards the outside. */
	outward: number;
	/** Mean colour of the strip just outside the edge (towards the wall). */
	outside: [number, number, number];
}

interface SideSpec {
	along: 'x' | 'y';
	from: number;
	to: number;
	centre: number;
	/** +1 when "outside" means increasing offset, -1 otherwise. */
	outwardSign: 1 | -1;
}

/**
 * The strongest nearly-straight edges along one side of the box. A side is parametrised by
 * its perpendicular offsets at the two ends and scored by the mean gradient across it; the
 * step-2 grid search keeps the best few local maxima (apart by at least 8 px), each refined
 * at step 1. Several candidates per side let the caller resolve "frame or mat?" jointly.
 */
function findSideCandidates(
	small: Raster,
	plane: Float32Array,
	spec: SideSpec,
	bandPx: number,
	maxTilt: number,
	keep: number
): Candidate[] {
	const { width, height } = small;
	const { along, from, to, centre } = spec;
	const limit = along === 'x' ? height - 2 : width - 2;
	const lo = Math.max(1, Math.round(centre - bandPx));
	const hi = Math.min(limit, Math.round(centre + bandPx));
	const span = to - from;
	const samples = Math.max(16, Math.min(256, Math.round(span)));
	const tiltLimit = maxTilt * span;
	const index = (u: number, v: number) => (along === 'x' ? v * width + u : u * width + v);
	const score = (oa: number, ob: number): number => {
		let sum = 0;
		for (let s = 0; s < samples; s += 1) {
			const t = (s + 0.5) / samples;
			const u = Math.round(from + t * span);
			const v = Math.round(oa + t * (ob - oa));
			sum += Math.abs(plane[index(u, v)] ?? 0);
		}
		return sum / samples;
	};

	const grid: { oa: number; ob: number; score: number }[] = [];
	for (let oa = lo; oa <= hi; oa += 2) {
		for (let ob = lo; ob <= hi; ob += 2) {
			if (Math.abs(oa - ob) > tiltLimit) continue;
			grid.push({ oa, ob, score: score(oa, ob) });
		}
	}
	grid.sort((p, q) => q.score - p.score);
	const picked: { oa: number; ob: number; score: number }[] = [];
	for (const g of grid) {
		if (picked.length >= keep) break;
		const mean = (g.oa + g.ob) / 2;
		if (picked.some((p) => Math.abs((p.oa + p.ob) / 2 - mean) < 8)) continue;
		picked.push(g);
	}

	const candidates: Candidate[] = [];
	for (const coarse of picked) {
		let best = coarse;
		for (let oa = Math.max(lo, coarse.oa - 2); oa <= Math.min(hi, coarse.oa + 2); oa += 1) {
			for (let ob = Math.max(lo, coarse.ob - 2); ob <= Math.min(hi, coarse.ob + 2); ob += 1) {
				if (Math.abs(oa - ob) > tiltLimit) continue;
				const value = score(oa, ob);
				if (value > best.score) best = { oa, ob, score: value };
			}
		}
		// Colour of the strip 4..12 px beyond the edge, on the outside.
		const colour: [number, number, number] = [0, 0, 0];
		let count = 0;
		for (let s = 0; s < samples; s += 2) {
			const t = (s + 0.5) / samples;
			const u = Math.round(from + t * span);
			const edge = best.oa + t * (best.ob - best.oa);
			for (let d = 4; d <= 12; d += 2) {
				const v = Math.round(edge + d * spec.outwardSign);
				if (v < 0 || v > limit + 1) continue;
				const p = index(u, v) * 4;
				colour[0] += small.data[p]!;
				colour[1] += small.data[p + 1]!;
				colour[2] += small.data[p + 2]!;
				count += 1;
			}
		}
		// Candidates on the picture's border have no outside to compare; skip them.
		if (count < Math.ceil(samples / 2) * 5 * 0.5) continue;
		const outside = colour.map((c) => c / count) as [number, number, number];
		const point = (u: number, v: number): Point =>
			along === 'x' ? { x: u, y: v } : { x: v, y: u };
		candidates.push({
			a: point(from, best.oa),
			b: point(to, best.ob),
			score: best.score,
			outward: (((best.oa + best.ob) / 2 - centre) * spec.outwardSign) / Math.max(1, bandPx),
			outside
		});
	}
	return candidates;
}

/**
 * How much the outside strips disagree. Lighting falls off across a wall, so brightness is
 * only a weak cue; the chromaticity (colour divided by its mean) tells wood from mat from
 * wall even under a gradient.
 */
function colourSpread(colours: [number, number, number][]): number {
	const chroma = colours.map((c): [number, number, number, number] => {
		const mean = Math.max(1, (c[0] + c[1] + c[2]) / 3);
		return [c[0] / mean, c[1] / mean, c[2] / mean, mean / 255];
	});
	const centre = [0, 1, 2, 3].map((k) => chroma.reduce((s, c) => s + c[k]!, 0) / chroma.length) as [
		number,
		number,
		number,
		number
	];
	let spread = 0;
	for (const c of chroma) {
		spread += Math.hypot(c[0] - centre[0], c[1] - centre[1], c[2] - centre[2]);
		spread += 0.5 * Math.abs(c[3] - centre[3]);
	}
	return spread / chroma.length;
}

/**
 * Locates the four edges of a piece (its frame, canvas or sheet) near the sides of `box`
 * and returns their intersections, in `raster` pixels. Sides are chosen together: a good
 * set has strong edges whose outside strips share one colour (the same wall, mat or frame
 * on every side), with a nudge towards the outer edge so the frame stays in the picture.
 * Falls back to the box itself, with confidence 0, when the weakest side is unconvincing or
 * the corners do not form a sensible quadrilateral.
 */
export function findQuad(
	raster: Raster,
	box: Box,
	options: Partial<FindQuadOptions> = {}
): QuadDetection {
	const opts = { ...DEFAULT_FIND_QUAD_OPTIONS, ...options };
	const fallback: QuadDetection = {
		quad: boxToQuad(box.x1, box.y1, box.x2, box.y2),
		confidence: 0,
		sides: [0, 0, 0, 0]
	};
	const factor = downscaleFactor(raster.width, raster.height, opts.analysisEdge);
	const small = downscale(raster, factor);
	const { width, height } = small;
	const grad = sobel(luminance(small), width, height);

	const x1 = box.x1 / factor;
	const y1 = box.y1 / factor;
	const x2 = box.x2 / factor;
	const y2 = box.y2 / factor;
	const bw = x2 - x1;
	const bh = y2 - y1;
	if (bw < 24 || bh < 24) return fallback;
	// Skip the corners: there the perpendicular edges would pollute the score.
	const inset = 0.08;
	const specs: SideSpec[] = [
		{ along: 'x', from: x1 + bw * inset, to: x2 - bw * inset, centre: y1, outwardSign: -1 },
		{ along: 'x', from: x1 + bw * inset, to: x2 - bw * inset, centre: y2, outwardSign: 1 },
		{ along: 'y', from: y1 + bh * inset, to: y2 - bh * inset, centre: x1, outwardSign: -1 },
		{ along: 'y', from: y1 + bh * inset, to: y2 - bh * inset, centre: x2, outwardSign: 1 }
	];
	// A box that already touches the picture's edge means the photo was cropped to the piece:
	// its edges can only be near the border, so the band shrinks with the room outside the box.
	const rooms = [y1, height - y2, x1, width - x2];
	const sides = specs.map((spec, i) => {
		const dim = i < 2 ? bh : bw;
		const bandPx = Math.min(dim * opts.band, Math.max(dim * 0.05, 1.5 * rooms[i]!));
		return findSideCandidates(
			small,
			i < 2 ? grad.gy : grad.gx,
			spec,
			bandPx,
			opts.maxTilt,
			opts.candidates
		);
	});
	if (sides.some((s) => s.length === 0)) return fallback;

	// Edge strength on an absolute scale: a clean step of ~30 grey levels saturates it, so a
	// busy painting does not make its own frame look weak.
	const reference = 120;
	const strength = (c: Candidate) => Math.min(1, c.score / reference);

	let best: {
		choice: Candidate[];
		total: number;
		confidence: number;
		strengths: number[];
	} | null = null;
	const bias = opts.prefer === 'outer' ? 0.25 : -0.25;
	for (const top of sides[0]!) {
		for (const bottom of sides[1]!) {
			for (const left of sides[2]!) {
				for (const right of sides[3]!) {
					const choice = [top, bottom, left, right];
					const strengths = choice.map(strength);
					const spread = colourSpread(choice.map((c) => c.outside));
					const outward = choice.reduce((s, c) => s + c.outward, 0) / 4;
					const total =
						strengths.reduce((s, v) => s + v, 0) / 4 - CONSISTENCY * spread + bias * outward;
					if (best && total <= best.total) continue;
					const confidence = Math.max(
						0,
						Math.min(Math.min(...strengths), 1 - CONSISTENCY * spread)
					);
					best = { choice, total, confidence, strengths };
				}
			}
		}
	}
	if (!best || best.confidence < opts.minConfidence) return fallback;

	const [top, bottom, left, right] = best.choice as [Candidate, Candidate, Candidate, Candidate];
	const corners = [
		intersectLines(top.a, top.b, left.a, left.b),
		intersectLines(top.a, top.b, right.a, right.b),
		intersectLines(bottom.a, bottom.b, right.a, right.b),
		intersectLines(bottom.a, bottom.b, left.a, left.b)
	];
	if (corners.some((c) => c === null)) return fallback;
	const quad = corners.map((c) => ({
		x: Math.min(width, Math.max(0, c!.x)),
		y: Math.min(height, Math.max(0, c!.y))
	})) as Quad;
	if (!quadIsConvex(quad)) return fallback;
	const areaRatio = quadArea(quad) / (bw * bh);
	if (areaRatio < 0.4 || areaRatio > 1.8) return fallback;

	return {
		quad: quad.map((c) => ({ x: c.x * factor, y: c.y * factor })) as Quad,
		confidence: best.confidence,
		sides: best.strengths as [number, number, number, number]
	};
}
