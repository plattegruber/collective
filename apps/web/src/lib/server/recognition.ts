import { env } from 'cloudflare:workers';
import { and, desc, eq, inArray } from 'drizzle-orm';
import { db, schema } from './db/index.js';
import { newId } from '#lib/utils/id.js';
import { bytesToBase64Url, sha256Hex } from '#lib/utils/encoding.js';
import type {
	NormalizedBox,
	PrototypeIndexPayload,
	PrototypeSource,
	RecognizerConfig,
	RecognizerManifest
} from '#lib/domain/recognition.js';
import { cosineSimilarity } from '#lib/recognition/vectors.js';

export class RecognitionError extends Error {
	constructor(
		public readonly code: 'not_found' | 'invalid' | 'conflict',
		message: string
	) {
		super(message);
	}
}

/* ---------------------------------- releases ---------------------------------- */

export async function activeRelease(): Promise<schema.RecognizerRelease | null> {
	const row = await db().query.recognizerReleases.findFirst({
		where: eq(schema.recognizerReleases.isActive, true),
		orderBy: desc(schema.recognizerReleases.createdAt)
	});
	return row ?? null;
}

export async function listReleases(): Promise<schema.RecognizerRelease[]> {
	return db().query.recognizerReleases.findMany({
		orderBy: desc(schema.recognizerReleases.createdAt)
	});
}

export async function getRelease(id: string): Promise<schema.RecognizerRelease | null> {
	const row = await db().query.recognizerReleases.findFirst({
		where: eq(schema.recognizerReleases.id, id)
	});
	return row ?? null;
}

export interface PublishReleaseInput {
	version: string;
	detector: { bytes: ArrayBuffer; inputSize: number };
	embedder: { bytes: ArrayBuffer; inputSize: number; dim: number };
	config: RecognizerConfig;
	metrics?: Record<string, unknown>;
	notes?: string;
	activate: boolean;
}

/** Stores both model files in R2 under models/{id}/ and records the release. */
export async function publishRelease(
	input: PublishReleaseInput
): Promise<schema.RecognizerRelease> {
	const existing = await db().query.recognizerReleases.findFirst({
		where: eq(schema.recognizerReleases.version, input.version)
	});
	if (existing) throw new RecognitionError('conflict', `Version ${input.version} already exists.`);

	const id = newId();
	const detectorKey = `models/${id}/detector.onnx`;
	const embedderKey = `models/${id}/embedder.onnx`;
	const [detectorSha256, embedderSha256] = await Promise.all([
		sha256Hex(input.detector.bytes),
		sha256Hex(input.embedder.bytes)
	]);
	await Promise.all([
		env.MEDIA.put(detectorKey, input.detector.bytes, {
			httpMetadata: { contentType: 'application/octet-stream' },
			customMetadata: { sha256: detectorSha256 }
		}),
		env.MEDIA.put(embedderKey, input.embedder.bytes, {
			httpMetadata: { contentType: 'application/octet-stream' },
			customMetadata: { sha256: embedderSha256 }
		})
	]);
	if (input.activate) {
		await db().update(schema.recognizerReleases).set({ isActive: false });
	}
	const [release] = await db()
		.insert(schema.recognizerReleases)
		.values({
			id,
			version: input.version,
			detectorKey,
			detectorSha256,
			detectorBytes: input.detector.bytes.byteLength,
			detectorInputSize: input.detector.inputSize,
			embedderKey,
			embedderSha256,
			embedderBytes: input.embedder.bytes.byteLength,
			embedderInputSize: input.embedder.inputSize,
			embedderDim: input.embedder.dim,
			config: input.config,
			metrics: input.metrics ?? null,
			notes: input.notes ?? null,
			isActive: input.activate
		})
		.returning();
	return release!;
}

export async function activateRelease(id: string): Promise<boolean> {
	const release = await getRelease(id);
	if (!release) return false;
	await db().update(schema.recognizerReleases).set({ isActive: false });
	await db()
		.update(schema.recognizerReleases)
		.set({ isActive: true })
		.where(eq(schema.recognizerReleases.id, id));
	return true;
}

export function releaseManifest(
	release: schema.RecognizerRelease,
	baseUrl = ''
): RecognizerManifest {
	return {
		id: release.id,
		version: release.version,
		detector: {
			url: `${baseUrl}/${release.detectorKey}`,
			sha256: release.detectorSha256,
			bytes: release.detectorBytes,
			inputSize: release.detectorInputSize
		},
		embedder: {
			url: `${baseUrl}/${release.embedderKey}`,
			sha256: release.embedderSha256,
			bytes: release.embedderBytes,
			inputSize: release.embedderInputSize,
			dim: release.embedderDim
		},
		config: release.config
	};
}

/* --------------------------------- prototypes --------------------------------- */

export function encodeEmbedding(vector: Float32Array): Uint8Array {
	const out = new Uint8Array(vector.length * 4);
	const view = new DataView(out.buffer);
	for (let i = 0; i < vector.length; i += 1) view.setFloat32(i * 4, vector[i]!, true);
	return out;
}

export function decodeEmbedding(bytes: Uint8Array | ArrayBuffer): Float32Array {
	const view =
		bytes instanceof ArrayBuffer
			? new DataView(bytes)
			: new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const out = new Float32Array(view.byteLength / 4);
	for (let i = 0; i < out.length; i += 1) out[i] = view.getFloat32(i * 4, true);
	return out;
}

export interface PrototypeRow {
	id: string;
	artworkId: string;
	imageId: string | null;
	source: PrototypeSource;
	embedding: Float32Array;
}

/** Prototypes of a release for the given artworks. Callers scope `artworkIds` to what the viewer may see. */
export async function listPrototypes(
	releaseId: string,
	artworkIds: string[]
): Promise<PrototypeRow[]> {
	if (artworkIds.length === 0) return [];
	const rows = await db().query.artworkPrototypes.findMany({
		where: and(
			eq(schema.artworkPrototypes.releaseId, releaseId),
			inArray(schema.artworkPrototypes.artworkId, artworkIds)
		),
		columns: { id: true, artworkId: true, imageId: true, source: true, embedding: true }
	});
	return rows.map((row) => ({
		...row,
		embedding: decodeEmbedding(new Uint8Array(row.embedding))
	}));
}

export function toIndexPayload(
	releaseId: string,
	dim: number,
	prototypes: PrototypeRow[]
): PrototypeIndexPayload {
	const artworks: Record<string, string[]> = {};
	for (const prototype of prototypes) {
		(artworks[prototype.artworkId] ??= []).push(
			bytesToBase64Url(encodeEmbedding(prototype.embedding))
		);
	}
	return { releaseId, dim, artworks };
}

async function ownedArtwork(userId: string, artworkId: string) {
	return db().query.artworks.findFirst({
		where: and(eq(schema.artworks.id, artworkId), eq(schema.artworks.userId, userId)),
		columns: { id: true }
	});
}

function requireDim(release: schema.RecognizerRelease, embedding: Float32Array): void {
	if (embedding.length !== release.embedderDim) {
		throw new RecognitionError(
			'invalid',
			`Expected ${release.embedderDim} values, got ${embedding.length}.`
		);
	}
	for (const value of embedding) {
		if (!Number.isFinite(value)) throw new RecognitionError('invalid', 'Embedding has NaN.');
	}
}

/** Records (or replaces) the prototype computed from one of the artwork's photos. */
export async function savePhotoPrototype(
	userId: string,
	artworkId: string,
	input: { imageId: string; releaseId: string; embedding: Float32Array; box: NormalizedBox | null }
): Promise<schema.ArtworkPrototype> {
	const [artwork, release, image] = await Promise.all([
		ownedArtwork(userId, artworkId),
		getRelease(input.releaseId),
		db().query.artworkImages.findFirst({
			where: and(
				eq(schema.artworkImages.id, input.imageId),
				eq(schema.artworkImages.artworkId, artworkId)
			),
			columns: { id: true }
		})
	]);
	if (!artwork) throw new RecognitionError('not_found', 'Artwork not found.');
	if (!release) throw new RecognitionError('not_found', 'Unknown recognizer release.');
	if (!image) throw new RecognitionError('not_found', 'Photo not found.');
	requireDim(release, input.embedding);

	await db()
		.delete(schema.artworkPrototypes)
		.where(
			and(
				eq(schema.artworkPrototypes.imageId, input.imageId),
				eq(schema.artworkPrototypes.releaseId, input.releaseId)
			)
		);
	const [row] = await db()
		.insert(schema.artworkPrototypes)
		.values({
			id: newId(),
			artworkId,
			imageId: input.imageId,
			source: 'photo',
			releaseId: input.releaseId,
			embedding: encodeEmbedding(input.embedding) as unknown as Buffer,
			box: input.box
		})
		.returning();
	return row!;
}

/**
 * Keeps a view captured while scanning, so the recognizer learns the collector's real
 * lighting and angles. Near-duplicates of existing prototypes are ignored and the oldest
 * memory is dropped once the per-artwork cap is reached.
 */
export async function saveScanMemory(
	userId: string,
	artworkId: string,
	input: { releaseId: string; embedding: Float32Array }
): Promise<{ saved: boolean; reason?: 'duplicate' }> {
	const [artwork, release] = await Promise.all([
		ownedArtwork(userId, artworkId),
		getRelease(input.releaseId)
	]);
	if (!artwork) throw new RecognitionError('not_found', 'Artwork not found.');
	if (!release) throw new RecognitionError('not_found', 'Unknown recognizer release.');
	requireDim(release, input.embedding);

	const existing = await listPrototypes(input.releaseId, [artworkId]);
	const novelty = release.config.thresholds.memoryNovelty;
	if (existing.some((row) => cosineSimilarity(row.embedding, input.embedding) >= novelty)) {
		return { saved: false, reason: 'duplicate' };
	}
	const memories = await db().query.artworkPrototypes.findMany({
		where: and(
			eq(schema.artworkPrototypes.artworkId, artworkId),
			eq(schema.artworkPrototypes.releaseId, input.releaseId),
			eq(schema.artworkPrototypes.source, 'scan')
		),
		orderBy: desc(schema.artworkPrototypes.createdAt),
		columns: { id: true }
	});
	const overflow = memories.slice(Math.max(0, release.config.maxMemories - 1));
	if (overflow.length > 0) {
		await db()
			.delete(schema.artworkPrototypes)
			.where(
				inArray(
					schema.artworkPrototypes.id,
					overflow.map((row) => row.id)
				)
			);
	}
	await db()
		.insert(schema.artworkPrototypes)
		.values({
			id: newId(),
			artworkId,
			imageId: null,
			source: 'scan',
			releaseId: input.releaseId,
			embedding: encodeEmbedding(input.embedding) as unknown as Buffer,
			box: null
		});
	return { saved: true };
}

export interface PendingImage {
	artworkId: string;
	imageId: string;
	/** Thumbnail URL the browser can embed from. */
	url: string;
}

export interface IndexStatus {
	images: number;
	indexed: number;
	memories: number;
	pending: PendingImage[];
}

/** Which of the user's photos still lack a prototype for the release. */
export async function indexStatus(userId: string, releaseId: string): Promise<IndexStatus> {
	const artworks = await db().query.artworks.findMany({
		where: eq(schema.artworks.userId, userId),
		columns: { id: true },
		with: { images: { columns: { id: true, thumbKey: true } } }
	});
	const artworkIds = artworks.map((artwork) => artwork.id);
	const prototypes = await listPrototypes(releaseId, artworkIds);
	const indexedImages = new Set(prototypes.filter((p) => p.imageId).map((p) => p.imageId));
	const pending: PendingImage[] = [];
	let images = 0;
	for (const artwork of artworks) {
		for (const image of artwork.images) {
			images += 1;
			if (!indexedImages.has(image.id)) {
				pending.push({ artworkId: artwork.id, imageId: image.id, url: `/media/${image.thumbKey}` });
			}
		}
	}
	return {
		images,
		indexed: images - pending.length,
		memories: prototypes.filter((p) => p.source === 'scan').length,
		pending
	};
}

/** Validates the admin bearer token used to publish global releases. */
export function isReleaseAuthorized(request: Request): boolean {
	const secret = (env as unknown as { RELEASE_TOKEN?: string }).RELEASE_TOKEN;
	if (!secret) return false;
	const header = request.headers.get('authorization') ?? '';
	const token = header.startsWith('Bearer ') ? header.slice(7).trim() : '';
	return token.length > 0 && token === secret;
}
