import type * as Ort from 'onnxruntime-web';

export type OrtBackend = 'webgpu' | 'wasm';

export interface OrtRuntime {
	ort: typeof Ort;
	backend: OrtBackend;
	/** Execution providers to request, best first; ORT falls through the list. */
	executionProviders: OrtBackend[];
	/**
	 * True when sessions run in a worker (`env.wasm.proxy`): input buffers are transferred
	 * to it, so callers must hand over a fresh buffer per run.
	 */
	transfersInputs: boolean;
}

/**
 * onnxruntime-web is served from static/ort (staged by scripts/copy-ort-assets.mjs) and
 * imported at runtime so Vite never bundles its multi-megabyte WebAssembly.
 *
 * Two builds are staged: the plain wasm build, and the JSPI build that adds the WebGPU
 * execution provider on browsers with both WebGPU and JS Promise Integration (Chrome and
 * Edge on desktop and Android). Safari takes the wasm path with SIMD and threads, proxied
 * to a worker so a 300 ms embedding never freezes the video or the overlay.
 */
const ORT_BASE = '/ort/';
let pending: Promise<OrtRuntime> | undefined;

/**
 * WebGPU is worth it only when the browser hands out an adapter; headless browsers and some
 * desktops expose `navigator.gpu` without one, and ORT would then stall during session
 * creation instead of failing.
 */
async function supportsWebGpuBuild(): Promise<boolean> {
	if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
	if (typeof (WebAssembly as unknown as { Suspending?: unknown }).Suspending !== 'function')
		return false;
	try {
		const gpu = (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu;
		const adapter = await Promise.race([
			gpu.requestAdapter(),
			new Promise<null>((resolve) => setTimeout(() => resolve(null), 3000))
		]);
		return adapter !== null && adapter !== undefined;
	} catch {
		return false;
	}
}

function threadCount(): number {
	if (typeof crossOriginIsolated === 'undefined' || !crossOriginIsolated) return 1;
	return Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
}

async function importBuild(file: string, proxy: boolean): Promise<typeof Ort> {
	const module = (await import(/* @vite-ignore */ `${ORT_BASE}${file}`)) as typeof Ort;
	module.env.wasm.wasmPaths = ORT_BASE;
	module.env.wasm.numThreads = threadCount();
	module.env.wasm.proxy = proxy;
	module.env.logLevel = 'warning';
	return module;
}

/** The WebGPU provider cannot be proxied; plain wasm can whenever workers exist. */
function canProxy(): boolean {
	return typeof Worker !== 'undefined';
}

/** Loads the best runtime for this browser. A failed attempt is forgotten so retries work. */
export function loadOrt(preferred: OrtBackend | 'auto' = 'auto'): Promise<OrtRuntime> {
	if (pending) return pending;
	const promise: Promise<OrtRuntime> = (async (): Promise<OrtRuntime> => {
		if (preferred !== 'wasm' && (await supportsWebGpuBuild())) {
			try {
				const ort = await importBuild('ort.jspi.min.mjs', false);
				return {
					ort,
					backend: 'webgpu' as const,
					executionProviders: ['webgpu', 'wasm'],
					transfersInputs: false
				};
			} catch (cause) {
				console.warn('WebGPU runtime unavailable, using wasm', cause);
			}
		}
		const proxy = canProxy();
		const ort = await importBuild('ort.wasm.min.mjs', proxy);
		return { ort, backend: 'wasm' as const, executionProviders: ['wasm'], transfersInputs: proxy };
	})();
	pending = promise;
	promise.catch(() => {
		if (pending === promise) pending = undefined;
	});
	return promise;
}

const SESSION_TIMEOUT_MS = 45_000;

function withTimeout<T>(promise: Promise<T>, label: string): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(
			() => reject(new Error(`${label} took too long to start. Reload and try again.`)),
			SESSION_TIMEOUT_MS
		);
		promise.then(resolve, reject).finally(() => clearTimeout(timer));
	});
}

/** Creates a session, falling back to the wasm provider when WebGPU refuses the graph. */
export async function createSession(
	runtime: OrtRuntime,
	bytes: Uint8Array
): Promise<Ort.InferenceSession> {
	const { ort, executionProviders } = runtime;
	try {
		return await withTimeout(
			ort.InferenceSession.create(bytes, { executionProviders, graphOptimizationLevel: 'all' }),
			'The recognizer'
		);
	} catch (cause) {
		if (executionProviders[0] === 'webgpu') {
			console.warn('WebGPU session failed, falling back to wasm', cause);
			return withTimeout(
				ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }),
				'The recognizer'
			);
		}
		throw cause;
	}
}
