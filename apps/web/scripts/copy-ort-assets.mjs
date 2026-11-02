/**
 * onnxruntime-web is loaded at runtime from /ort/ rather than bundled: its WebAssembly
 * binaries are tens of megabytes, exceed Cloudflare's 25 MiB asset limit when Vite emits
 * them, and are only needed by the recognizer. This stages two builds into static/ort: the
 * plain wasm build (every browser) and the JSPI build that adds the WebGPU execution
 * provider on browsers with WebGPU and JS Promise Integration. Both stay under Cloudflare's
 * per-asset limit; the asyncify WebGPU build does not, which is why it is not used.
 */
import { copyFileSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const dist = dirname(require.resolve('onnxruntime-web'));
const target = join(here, '..', 'static', 'ort');

mkdirSync(target, { recursive: true });
for (const file of [
	'ort.wasm.min.mjs',
	'ort-wasm-simd-threaded.mjs',
	'ort-wasm-simd-threaded.wasm',
	'ort.jspi.min.mjs',
	'ort-wasm-simd-threaded.jspi.mjs',
	'ort-wasm-simd-threaded.jspi.wasm'
]) {
	copyFileSync(join(dist, file), join(target, file));
}
console.log(`[copy-ort-assets] staged onnxruntime-web into ${target}`);
