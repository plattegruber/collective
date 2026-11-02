import { error } from '@sveltejs/kit';
import { z } from 'zod';
import {
	RecognitionError,
	activeRelease,
	isReleaseAuthorized,
	listReleases,
	publishRelease,
	releaseManifest
} from '#lib/server/recognition.js';
import { DEFAULT_RECOGNIZER_CONFIG } from '#lib/domain/recognition.js';
import type { RequestHandler } from './$types';

const MAX_MODEL_BYTES = 96 * 1024 * 1024;

const configSchema = z.object({
	mean: z.tuple([z.number(), z.number(), z.number()]),
	std: z.tuple([z.number(), z.number(), z.number()]),
	pooling: z.enum(['cls', 'mean', 'cls+mean']),
	cropMargin: z.number().min(0).max(0.5),
	thresholds: z.object({
		detect: z.number().min(0).max(1),
		match: z.number().min(-1).max(1),
		strong: z.number().min(-1).max(1),
		margin: z.number().min(0).max(1).default(0.08),
		matchReference: z.number().int().min(1).default(7),
		matchSlope: z.number().min(0).max(1).default(0.034),
		memoryNovelty: z.number().min(-1).max(1)
	}),
	maxMemories: z.number().int().min(0).max(64),
	attribution: z.string().max(120)
});

function parseJsonField(value: FormDataEntryValue | null, name: string): unknown {
	if (value === null || value === '') return undefined;
	try {
		return JSON.parse(String(value));
	} catch {
		error(400, `${name} must be valid JSON.`);
	}
}

function requireModel(formData: FormData, name: string): File {
	const value = formData.get(name);
	if (!(value instanceof File) || value.size === 0) error(400, `Missing ${name} file.`);
	if (value.size > MAX_MODEL_BYTES) error(413, `${name} exceeds 96 MB.`);
	return value;
}

function requireInt(formData: FormData, name: string, min: number, max: number): number {
	const value = Number(formData.get(name));
	if (!Number.isInteger(value) || value < min || value > max)
		error(400, `${name} must be an integer between ${min} and ${max}.`);
	return value;
}

/** Public: the active release manifest plus the release history. */
export const GET: RequestHandler = async ({ url }) => {
	const [active, releases] = await Promise.all([activeRelease(), listReleases()]);
	return Response.json({
		active: active ? releaseManifest(active, url.origin) : null,
		releases: releases.map((release) => ({
			id: release.id,
			version: release.version,
			isActive: release.isActive,
			createdAt: release.createdAt,
			metrics: release.metrics,
			notes: release.notes
		}))
	});
};

/**
 * Publish a release (maintainers only; `Authorization: Bearer $RELEASE_TOKEN`). Multipart:
 * version, detector, embedder, detectorInputSize, embedderInputSize, embedderDim,
 * config (JSON, optional: defaults), metrics (JSON, optional), notes, activate ("true"/"false").
 */
export const POST: RequestHandler = async ({ request, url }) => {
	if (!isReleaseAuthorized(request)) error(401, 'A valid release token is required.');
	const formData = await request.formData();
	const version = String(formData.get('version') ?? '').trim();
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(version)) error(400, 'Invalid version.');
	const detector = requireModel(formData, 'detector');
	const embedder = requireModel(formData, 'embedder');
	const configResult = configSchema.safeParse(
		parseJsonField(formData.get('config'), 'config') ?? DEFAULT_RECOGNIZER_CONFIG
	);
	if (!configResult.success) error(400, 'config does not match RecognizerConfig.');
	const metricsRaw = parseJsonField(formData.get('metrics'), 'metrics');
	if (metricsRaw !== undefined && (typeof metricsRaw !== 'object' || metricsRaw === null))
		error(400, 'metrics must be a JSON object.');

	try {
		const release = await publishRelease({
			version,
			detector: {
				bytes: await detector.arrayBuffer(),
				inputSize: requireInt(formData, 'detectorInputSize', 64, 1280)
			},
			embedder: {
				bytes: await embedder.arrayBuffer(),
				inputSize: requireInt(formData, 'embedderInputSize', 64, 1024),
				dim: requireInt(formData, 'embedderDim', 16, 8192)
			},
			config: configResult.data,
			metrics: metricsRaw as Record<string, unknown> | undefined,
			notes: String(formData.get('notes') ?? '').slice(0, 2000) || undefined,
			activate: String(formData.get('activate') ?? 'true') !== 'false'
		});
		return Response.json(releaseManifest(release, url.origin), { status: 201 });
	} catch (cause) {
		if (cause instanceof RecognitionError)
			error(cause.code === 'conflict' ? 409 : 400, cause.message);
		throw cause;
	}
};
