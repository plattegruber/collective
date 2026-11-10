import type { Point } from './geometry.js';
import {
	boxBlur,
	createRaster,
	downscale,
	downscaleFactor,
	luminance,
	percentile,
	type Raster
} from './raster.js';

/**
 * Reflection removal from several shots of the same piece taken from slightly different
 * spots. Glass reflections are additive and move with the camera while the artwork, once
 * every shot is straightened to the same rectangle, stays put. The cleanest shot is the
 * reference; wherever it is brighter than the darkest shot at that spot by more than a
 * tolerance, the other shots take over with weights that fall off with their own excess.
 * Weights are computed blurred at low resolution, so small misalignments do not show.
 */
export interface FuseOptions {
	/** Long edge of the plane on which weights are computed. */
	analysisEdge: number;
	/** Luminance excess (0..255) treated as noise rather than reflection. */
	tolerance: number;
	/** Softness of the weight fall-off in luminance units. */
	softness: number;
	/** Weight-map blur radius as a fraction of the analysis edge. */
	blur: number;
	/** Largest translation searched at the coarsest level, in pixels there. */
	searchRange: number;
	/** Tiles per side for the local alignment refinement (1 disables it). */
	tiles: number;
	/** Local search radius around the global shift, in pixels at the refinement level. */
	localRange: number;
}

export const DEFAULT_FUSE_OPTIONS: FuseOptions = {
	analysisEdge: 320,
	tolerance: 6,
	softness: 10,
	blur: 0.02,
	searchRange: 12,
	tiles: 5,
	localRange: 4
};

/**
 * A smooth translation field: `shifts` holds one (dx, dy) per node of a `columns`×`rows`
 * grid spanning the picture, in full-resolution pixels; it is interpolated bilinearly.
 */
export interface ShiftField {
	columns: number;
	rows: number;
	shifts: Point[];
}

export function sampleField(
	field: ShiftField,
	width: number,
	height: number,
	x: number,
	y: number
): Point {
	if (field.columns === 1 && field.rows === 1) return field.shifts[0]!;
	const fx = Math.min(field.columns - 1, Math.max(0, ((x + 0.5) / width) * (field.columns - 1)));
	const fy = Math.min(field.rows - 1, Math.max(0, ((y + 0.5) / height) * (field.rows - 1)));
	const x0 = Math.floor(fx);
	const y0 = Math.floor(fy);
	const x1 = Math.min(field.columns - 1, x0 + 1);
	const y1 = Math.min(field.rows - 1, y0 + 1);
	const tx = fx - x0;
	const ty = fy - y0;
	const at = (c: number, r: number) => field.shifts[r * field.columns + c]!;
	const a = at(x0, y0);
	const b = at(x1, y0);
	const c = at(x0, y1);
	const d = at(x1, y1);
	return {
		x: a.x * (1 - tx) * (1 - ty) + b.x * tx * (1 - ty) + c.x * (1 - tx) * ty + d.x * tx * ty,
		y: a.y * (1 - tx) * (1 - ty) + b.y * tx * (1 - ty) + c.y * (1 - tx) * ty + d.y * tx * ty
	};
}

export interface FuseResult {
	raster: Raster;
	/** Index of the shot used as the base. */
	reference: number;
	/** Share of pixels whose reference weight dropped below one half. */
	changed: number;
	/** Per-shot global translation (in full-resolution pixels) that aligns it to the reference. */
	shifts: Point[];
	/** Per-shot translation field after the local refinement. */
	fields: ShiftField[];
}

interface Level {
	factor: number;
	width: number;
	height: number;
	lumas: Float32Array[];
}

function level(shots: Raster[], maxEdge: number): Level {
	const factor = downscaleFactor(shots[0]!.width, shots[0]!.height, maxEdge);
	const smalls = shots.map((shot) => downscale(shot, factor));
	return {
		factor,
		width: smalls[0]!.width,
		height: smalls[0]!.height,
		lumas: smalls.map(luminance)
	};
}

/** Sum of squared differences of `b` shifted by (dx, dy) against `a`, over the overlap. */
function ssd(
	a: Float32Array,
	b: Float32Array,
	width: number,
	height: number,
	dx: number,
	dy: number
): number {
	let sum = 0;
	let count = 0;
	const y0 = Math.max(0, -dy);
	const y1 = Math.min(height, height - dy);
	const x0 = Math.max(0, -dx);
	const x1 = Math.min(width, width - dx);
	for (let y = y0; y < y1; y += 1) {
		for (let x = x0; x < x1; x += 1) {
			const d = a[y * width + x]! - b[(y + dy) * width + (x + dx)]!;
			sum += d * d;
			count += 1;
		}
	}
	return count ? sum / count : Infinity;
}

/** Sum of squared differences over one window of `a` against `b` shifted by (dx, dy). */
function windowSsd(
	a: Float32Array,
	b: Float32Array,
	width: number,
	height: number,
	window: { x0: number; y0: number; x1: number; y1: number },
	dx: number,
	dy: number
): number {
	let sum = 0;
	let count = 0;
	for (let y = window.y0; y < window.y1; y += 1) {
		const sy = y + dy;
		if (sy < 0 || sy >= height) continue;
		for (let x = window.x0; x < window.x1; x += 1) {
			const sx = x + dx;
			if (sx < 0 || sx >= width) continue;
			const d = a[y * width + x]! - b[sy * width + sx]!;
			sum += d * d;
			count += 1;
		}
	}
	return count > 0 ? sum / count : Infinity;
}

/** Gradient magnitude of a luma plane: reflections are smooth, so edges align the artwork. */
function edges(luma: Float32Array, width: number, height: number): Float32Array {
	const out = new Float32Array(luma.length);
	for (let y = 1; y < height - 1; y += 1) {
		for (let x = 1; x < width - 1; x += 1) {
			const i = y * width + x;
			out[i] =
				Math.abs(luma[i + 1]! - luma[i - 1]!) + Math.abs(luma[i + width]! - luma[i - width]!);
		}
	}
	return out;
}

/**
 * Refines a global shift into a field: each tile of a grid searches a small neighbourhood
 * on the edge images, and keeps its own shift only when that clearly beats the global one
 * (tiles under a reflection or on blank paper have nothing to align and stay global).
 */
export function refineShiftField(
	fixed: Float32Array,
	moving: Float32Array,
	width: number,
	height: number,
	global: Point,
	tiles: number,
	range: number
): Point[] {
	const fixedEdges = edges(fixed, width, height);
	const movingEdges = edges(moving, width, height);
	const field: Point[] = [];
	const tileW = width / tiles;
	const tileH = height / tiles;
	for (let r = 0; r < tiles; r += 1) {
		for (let c = 0; c < tiles; c += 1) {
			// Windows overlap their neighbours by half a tile so every node sees real texture.
			const window = {
				x0: Math.max(0, Math.round((c - 0.5) * tileW)),
				y0: Math.max(0, Math.round((r - 0.5) * tileH)),
				x1: Math.min(width, Math.round((c + 1.5) * tileW)),
				y1: Math.min(height, Math.round((r + 1.5) * tileH))
			};
			const base = windowSsd(fixedEdges, movingEdges, width, height, window, global.x, global.y);
			let best = { x: global.x, y: global.y, cost: base };
			for (let dy = -range; dy <= range; dy += 1) {
				for (let dx = -range; dx <= range; dx += 1) {
					if (dx === 0 && dy === 0) continue;
					const cost = windowSsd(
						fixedEdges,
						movingEdges,
						width,
						height,
						window,
						global.x + dx,
						global.y + dy
					);
					if (cost < best.cost) best = { x: global.x + dx, y: global.y + dy, cost };
				}
			}
			field.push(best.cost < base * 0.9 ? { x: best.x, y: best.y } : { x: global.x, y: global.y });
		}
	}
	return field;
}

/** Translation that best aligns `moving` onto `fixed`, searched around `around` within ±range. */
function bestShift(
	fixed: Float32Array,
	moving: Float32Array,
	width: number,
	height: number,
	around: Point,
	range: number
): Point {
	let best = { x: around.x, y: around.y, cost: Infinity };
	for (let dy = around.y - range; dy <= around.y + range; dy += 1) {
		for (let dx = around.x - range; dx <= around.x + range; dx += 1) {
			const cost = ssd(fixed, moving, width, height, dx, dy);
			if (cost < best.cost) best = { x: dx, y: dy, cost };
		}
	}
	return { x: best.x, y: best.y };
}

/** Coarse-to-fine translation estimate of each shot against the reference, in full pixels. */
export function alignShots(shots: Raster[], reference: number, searchRange: number): Point[] {
	const edges = [160, 480, 1600].filter(
		(edge, i, all) => i === 0 || Math.max(shots[0]!.width, shots[0]!.height) > all[i - 1]!
	);
	const levels = edges.map((edge) => level(shots, edge));
	return shots.map((_, i) => {
		if (i === reference) return { x: 0, y: 0 };
		let shift: Point = { x: 0, y: 0 };
		let previousFactor = levels[0]!.factor;
		levels.forEach((lvl, li) => {
			const scaled = {
				x: Math.round((shift.x * previousFactor) / lvl.factor),
				y: Math.round((shift.y * previousFactor) / lvl.factor)
			};
			shift = bestShift(
				lvl.lumas[reference]!,
				lvl.lumas[i]!,
				lvl.width,
				lvl.height,
				scaled,
				li === 0 ? searchRange : 3
			);
			previousFactor = lvl.factor;
		});
		return { x: shift.x * previousFactor, y: shift.y * previousFactor };
	});
}

export function fuseShots(shots: Raster[], options: Partial<FuseOptions> = {}): FuseResult {
	const opts = { ...DEFAULT_FUSE_OPTIONS, ...options };
	if (shots.length === 0) throw new RangeError('Nothing to fuse.');
	const { width, height } = shots[0]!;
	if (shots.some((s) => s.width !== width || s.height !== height))
		throw new RangeError('Shots must share one size.');
	if (shots.length === 1) {
		const none = { x: 0, y: 0 };
		return {
			raster: shots[0]!,
			reference: 0,
			changed: 0,
			shifts: [none],
			fields: [{ columns: 1, rows: 1, shifts: [none] }]
		};
	}

	const analysis = level(shots, opts.analysisEdge);
	const { factor, width: aw, height: ah } = analysis;

	// Exposure: match each shot's darker tones (least affected by reflections) to the darkest.
	const p20 = analysis.lumas.map((l) => percentile(l, 0.2));
	const target = Math.min(...p20);
	const gains = p20.map((p) => Math.min(1.3, Math.max(0.7, target / Math.max(1, p))));

	// The reference is the shot with the least total excess over the per-pixel floor.
	const floor = new Float32Array(aw * ah).fill(Infinity);
	analysis.lumas.forEach((l, i) => {
		for (let p = 0; p < floor.length; p += 1) floor[p] = Math.min(floor[p]!, l[p]! * gains[i]!);
	});
	const excessTotals = analysis.lumas.map((l, i) => {
		let sum = 0;
		for (let p = 0; p < floor.length; p += 1) sum += l[p]! * gains[i]! - floor[p]!;
		return sum;
	});
	const reference = excessTotals.indexOf(Math.min(...excessTotals));

	const shifts = alignShots(shots, reference, opts.searchRange);

	// Local refinement on a mid-resolution level, expressed in full-resolution pixels.
	const fine = level(shots, 640);
	const fields: ShiftField[] = shots.map((_, i) => {
		if (i === reference || opts.tiles <= 1) {
			return { columns: 1, rows: 1, shifts: [shifts[i]!] };
		}
		const global = {
			x: Math.round(shifts[i]!.x / fine.factor),
			y: Math.round(shifts[i]!.y / fine.factor)
		};
		const local = refineShiftField(
			fine.lumas[reference]!,
			fine.lumas[i]!,
			fine.width,
			fine.height,
			global,
			opts.tiles,
			opts.localRange
		);
		return {
			columns: opts.tiles,
			rows: opts.tiles,
			shifts: local.map((p) => ({ x: p.x * fine.factor, y: p.y * fine.factor }))
		};
	});

	// Weight maps at analysis resolution, computed on aligned, exposure-matched lumas.
	const aligned = analysis.lumas.map((l, i) => {
		const out = new Float32Array(aw * ah);
		for (let y = 0; y < ah; y += 1) {
			for (let x = 0; x < aw; x += 1) {
				const shift = sampleField(fields[i]!, width, height, x * factor, y * factor);
				const sy = Math.min(ah - 1, Math.max(0, Math.round(y + shift.y / factor)));
				const sx = Math.min(aw - 1, Math.max(0, Math.round(x + shift.x / factor)));
				out[y * aw + x] = l[sy * aw + sx]! * gains[i]!;
			}
		}
		return out;
	});
	const alignedFloor = new Float32Array(aw * ah).fill(Infinity);
	aligned.forEach((l) => {
		for (let p = 0; p < alignedFloor.length; p += 1)
			alignedFloor[p] = Math.min(alignedFloor[p]!, l[p]!);
	});
	const radius = Math.max(1, Math.round(opts.blur * Math.max(aw, ah)));
	const excess = aligned.map((l) => {
		const e = new Float32Array(aw * ah);
		for (let p = 0; p < e.length; p += 1)
			e[p] = Math.max(0, l[p]! - alignedFloor[p]! - opts.tolerance);
		return boxBlur(e, aw, ah, radius);
	});
	const weights = aligned.map(() => new Float32Array(aw * ah));
	let changed = 0;
	for (let p = 0; p < aw * ah; p += 1) {
		const refWeight = Math.exp(-excess[reference]![p]! / opts.softness);
		let total = refWeight;
		weights[reference]![p] = refWeight;
		aligned.forEach((_, i) => {
			if (i === reference) return;
			const w = (1 - refWeight) * Math.exp(-excess[i]![p]! / opts.softness);
			weights[i]![p] = w;
			total += w;
		});
		for (const w of weights) w[p] = w[p]! / total;
		if (weights[reference]![p]! < 0.5) changed += 1;
	}

	// Composite at full resolution with bilinearly upsampled weights.
	const out = createRaster(width, height);
	const sampleWeight = (w: Float32Array, x: number, y: number): number => {
		const fx = Math.min(aw - 1, Math.max(0, (x + 0.5) / factor - 0.5));
		const fy = Math.min(ah - 1, Math.max(0, (y + 0.5) / factor - 0.5));
		const x0 = Math.floor(fx);
		const y0 = Math.floor(fy);
		const x1 = Math.min(aw - 1, x0 + 1);
		const y1 = Math.min(ah - 1, y0 + 1);
		const tx = fx - x0;
		const ty = fy - y0;
		return (
			w[y0 * aw + x0]! * (1 - tx) * (1 - ty) +
			w[y0 * aw + x1]! * tx * (1 - ty) +
			w[y1 * aw + x0]! * (1 - tx) * ty +
			w[y1 * aw + x1]! * tx * ty
		);
	};
	for (let y = 0; y < height; y += 1) {
		for (let x = 0; x < width; x += 1) {
			let r = 0;
			let g = 0;
			let b = 0;
			let total = 0;
			for (let i = 0; i < shots.length; i += 1) {
				const w = sampleWeight(weights[i]!, x, y);
				if (w < 1e-3) continue;
				const shift = sampleField(fields[i]!, width, height, x, y);
				const sx = Math.min(width - 1, Math.max(0, x + shift.x));
				const sy = Math.min(height - 1, Math.max(0, y + shift.y));
				const x0 = Math.floor(sx);
				const y0 = Math.floor(sy);
				const x1 = Math.min(width - 1, x0 + 1);
				const y1 = Math.min(height - 1, y0 + 1);
				const tx = sx - x0;
				const ty = sy - y0;
				const data = shots[i]!.data;
				const p00 = (y0 * width + x0) * 4;
				const p10 = (y0 * width + x1) * 4;
				const p01 = (y1 * width + x0) * 4;
				const p11 = (y1 * width + x1) * 4;
				const w00 = (1 - tx) * (1 - ty);
				const w10 = tx * (1 - ty);
				const w01 = (1 - tx) * ty;
				const w11 = tx * ty;
				const gain = gains[i]! * w;
				r += (data[p00]! * w00 + data[p10]! * w10 + data[p01]! * w01 + data[p11]! * w11) * gain;
				g +=
					(data[p00 + 1]! * w00 +
						data[p10 + 1]! * w10 +
						data[p01 + 1]! * w01 +
						data[p11 + 1]! * w11) *
					gain;
				b +=
					(data[p00 + 2]! * w00 +
						data[p10 + 2]! * w10 +
						data[p01 + 2]! * w01 +
						data[p11 + 2]! * w11) *
					gain;
				total += w;
			}
			const o = (y * width + x) * 4;
			const norm = total > 0 ? 1 / total : 0;
			out.data[o] = r * norm;
			out.data[o + 1] = g * norm;
			out.data[o + 2] = b * norm;
			out.data[o + 3] = 255;
		}
	}
	return { raster: out, reference, changed: changed / (aw * ah), shifts, fields };
}
