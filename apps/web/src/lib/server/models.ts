import { env } from 'cloudflare:workers';
import { and, desc, eq } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { newId } from '#lib/utils/id.js';
import { sha256Hex } from '#lib/utils/encoding.js';

export async function activeModel(userId: string): Promise<schema.DetectorModel | null> {
	const row = await db().query.detectorModels.findFirst({
		where: and(eq(schema.detectorModels.userId, userId), eq(schema.detectorModels.isActive, true)),
		orderBy: desc(schema.detectorModels.createdAt)
	});
	return row ?? null;
}

export async function listModels(userId: string): Promise<schema.DetectorModel[]> {
	return db().query.detectorModels.findMany({
		where: eq(schema.detectorModels.userId, userId),
		orderBy: desc(schema.detectorModels.createdAt)
	});
}

export interface PublishModelInput {
	onnx: ArrayBuffer;
	labels: Record<string, string>;
	inputSize: number;
	metrics?: Record<string, unknown>;
}

/** Stores a new ONNX model in R2, records it, and makes it the user's active model. */
export async function publishModel(
	userId: string,
	input: PublishModelInput
): Promise<schema.DetectorModel> {
	const id = newId();
	const modelKey = `users/${userId}/models/${id}/model.onnx`;
	const sha256 = await sha256Hex(input.onnx);
	await env.MEDIA.put(modelKey, input.onnx, {
		httpMetadata: { contentType: 'application/octet-stream' },
		customMetadata: { sha256 }
	});
	await db()
		.update(schema.detectorModels)
		.set({ isActive: false })
		.where(eq(schema.detectorModels.userId, userId));
	const [model] = await db()
		.insert(schema.detectorModels)
		.values({
			id,
			userId,
			modelKey,
			labels: input.labels,
			sha256,
			bytes: input.onnx.byteLength,
			inputSize: input.inputSize,
			metrics: input.metrics ?? null,
			isActive: true
		})
		.returning();
	return model!;
}

export async function activateModel(userId: string, modelId: string): Promise<boolean> {
	const model = await db().query.detectorModels.findFirst({
		where: and(eq(schema.detectorModels.id, modelId), eq(schema.detectorModels.userId, userId))
	});
	if (!model) return false;
	await db()
		.update(schema.detectorModels)
		.set({ isActive: false })
		.where(eq(schema.detectorModels.userId, userId));
	await db()
		.update(schema.detectorModels)
		.set({ isActive: true })
		.where(eq(schema.detectorModels.id, modelId));
	return true;
}

export async function deleteModel(userId: string, modelId: string): Promise<boolean> {
	const [deleted] = await db()
		.delete(schema.detectorModels)
		.where(and(eq(schema.detectorModels.id, modelId), eq(schema.detectorModels.userId, userId)))
		.returning({ modelKey: schema.detectorModels.modelKey });
	if (!deleted) return false;
	await env.MEDIA.delete(deleted.modelKey);
	return true;
}

/** The manifest a scanner loads: where the graph lives and how classes map to artwork ids. */
export function modelManifest(model: schema.DetectorModel, baseUrl = '') {
	return {
		id: model.id,
		version: model.createdAt.toISOString(),
		format: 'onnx' as const,
		url: `${baseUrl}/media/${model.modelKey}`,
		sha256: model.sha256,
		inputSize: model.inputSize,
		labels: model.labels
	};
}
