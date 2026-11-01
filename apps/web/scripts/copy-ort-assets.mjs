/**
 * onnxruntime-web is loaded at runtime from /ort/ rather than bundled: its WebAssembly
 * binaries are tens of megabytes, exceed Cloudflare's 25 MiB asset limit when Vite emits
 * them, and are only needed by the scanner. This stages the wasm-only build into static/ort.
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
	'ort-wasm-simd-threaded.wasm'
]) {
	copyFileSync(join(dist, file), join(target, file));
}
console.log(`[copy-ort-assets] staged onnxruntime-web into ${target}`);
