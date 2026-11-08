import type { RecognizerManifest } from '#lib/domain/recognition.js';
import { fetchModel, type DownloadProgress } from './assets.js';
import { createDetector, type Detector } from './detector.js';
import { createEmbedder, type Embedder } from './embedder.js';
import { loadOrt, type OrtBackend } from './ort.js';

export interface RecognizerModels {
	manifest: RecognizerManifest;
	detector: Detector;
	embedder: Embedder;
	backend: OrtBackend;
}

export type LoadStage =
	{ stage: 'runtime' } | { stage: 'download'; progress: DownloadProgress } | { stage: 'prepare' };

let loaded: { releaseId: string; promise: Promise<RecognizerModels> } | undefined;

/**
 * Loads both models for a release, once per page lifetime. The scanner, the upload page
 * and the settings rebuild all share the sessions; a failure is forgotten so the user can
 * retry.
 */
export function loadRecognizer(
	manifest: RecognizerManifest,
	onStage?: (stage: LoadStage) => void,
	preferred: OrtBackend | 'auto' = 'auto'
): Promise<RecognizerModels> {
	if (loaded && loaded.releaseId === manifest.id) return loaded.promise;
	const promise = (async () => {
		onStage?.({ stage: 'runtime' });
		const runtime = await loadOrt(preferred);
		const total = manifest.detector.bytes + manifest.embedder.bytes;
		let detectorLoaded = 0;
		let embedderLoaded = 0;
		const report = (cached: boolean) =>
			onStage?.({
				stage: 'download',
				progress: { loaded: detectorLoaded + embedderLoaded, total, cached }
			});
		const [detectorBytes, embedderBytes] = await Promise.all([
			fetchModel(manifest.detector, (p) => {
				detectorLoaded = p.loaded;
				report(p.cached);
			}),
			fetchModel(manifest.embedder, (p) => {
				embedderLoaded = p.loaded;
				report(p.cached);
			})
		]);
		onStage?.({ stage: 'prepare' });
		// Sequential on purpose: two sessions initialising at once compete for the thread pool.
		const detector = await createDetector(runtime, detectorBytes, manifest.detector.inputSize);
		const embedder = await createEmbedder(
			runtime,
			embedderBytes,
			manifest.embedder.inputSize,
			manifest.embedder.dim,
			manifest.config
		);
		return { manifest, detector, embedder, backend: runtime.backend };
	})();
	loaded = { releaseId: manifest.id, promise };
	promise.catch(() => {
		if (loaded?.promise === promise) loaded = undefined;
	});
	return promise;
}

export function formatLoadStage(stage: LoadStage): string {
	switch (stage.stage) {
		case 'runtime':
			return 'Starting the recognizer';
		case 'download': {
			if (stage.progress.cached) return 'Loading the recognizer';
			const mb = (n: number) => (n / 1_048_576).toFixed(0);
			return `Downloading the recognizer · ${mb(stage.progress.loaded)} of ${mb(stage.progress.total)} MB`;
		}
		case 'prepare':
			return 'Warming up';
	}
}
