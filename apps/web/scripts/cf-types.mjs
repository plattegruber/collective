/**
 * Regenerates worker-configuration.d.ts from wrangler.jsonc.
 *
 * Wrangler also types the Worker entry as
 * `mainModule: typeof import("./.svelte-kit/cloudflare/_worker")`, which makes svelte-check
 * follow the entire build output (thousands of untyped files). The app never reads
 * `Cloudflare.GlobalProps`, so that line is dropped.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';

const file = 'worker-configuration.d.ts';
// Resolved through npm so this works from the workspace root's hoisted node_modules too.
execFileSync('npx', ['--no-install', 'wrangler', 'types', file], { stdio: 'inherit' });

const generated = readFileSync(file, 'utf8');
const trimmed = generated.replace(/^[ \t]*mainModule: typeof import\(".*"\);\r?\n/m, '');
if (trimmed === generated) {
	console.warn(`[cf-types] no mainModule line found in ${file}; wrangler output may have changed`);
}
writeFileSync(file, trimmed);
