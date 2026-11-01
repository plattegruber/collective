import { error } from '@sveltejs/kit';
import { z } from 'zod';
import { authenticateApiToken } from '#lib/server/apiTokens.js';
import { listModels, modelManifest, publishModel } from '#lib/server/models.js';
import { db, schema } from '#lib/server/db/index.js';
import { eq } from 'drizzle-orm';
import type { RequestHandler } from './$types';

const MAX_MODEL_BYTES = 64 * 1024 * 1024;

const labelsSchema = z.record(z.string().regex(/^\d+$/), z.string().min(1));
const metricsSchema = z.record(z.string(), z.unknown()).optional();

export const GET: RequestHandler = async ({ request, url }) => {
	const auth = await authenticateApiToken(request);
	if (!auth) error(401, 'Invalid or missing API token.');
	const models = await listModels(auth.userId);
	return Response.json({
		models: models.map((model) => ({
			...modelManifest(model, url.origin),
			isActive: model.isActive,
			metrics: model.metrics
		}))
	});
};

/**
 * Publish a trained model. Multipart fields:
 *   model   – the ONNX file
 *   labels  – JSON object {"1": "<artworkId>", …}
 *   inputSize – integer (default 320)
 *   metrics – optional JSON object
 */
export const POST: RequestHandler = async ({ request, url }) => {
	const auth = await authenticateApiToken(request);
	if (!auth) error(401, 'Invalid or missing API token.');

	const formData = await request.formData();
	const model = formData.get('model');
	if (!(model instanceof File) || model.size === 0) error(400, 'Missing model file.');
	if (model.size > MAX_MODEL_BYTES) error(413, 'Model exceeds 64 MB.');

	const labelsResult = labelsSchema.safeParse(JSON.parse(String(formData.get('labels') ?? '{}')));
	if (!labelsResult.success) error(400, 'labels must map class indices to artwork ids.');

	const artworkIds = new Set(
		(
			await db().query.artworks.findMany({
				where: eq(schema.artworks.userId, auth.userId),
				columns: { id: true }
			})
		).map((row) => row.id)
	);
	const unknown = Object.values(labelsResult.data).filter((id) => !artworkIds.has(id));
	if (unknown.length > 0)
		error(400, `Labels reference artworks you do not own: ${unknown.join(', ')}`);

	const inputSize = Number(formData.get('inputSize') ?? 320);
	if (!Number.isInteger(inputSize) || inputSize < 64 || inputSize > 1280)
		error(400, 'inputSize must be between 64 and 1280.');

	const metricsRaw = formData.get('metrics');
	const metrics = metricsRaw ? metricsSchema.parse(JSON.parse(String(metricsRaw))) : undefined;

	const published = await publishModel(auth.userId, {
		onnx: await model.arrayBuffer(),
		labels: labelsResult.data,
		inputSize,
		metrics
	});
	return Response.json(modelManifest(published, url.origin), { status: 201 });
};
