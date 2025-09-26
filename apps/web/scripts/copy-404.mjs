import { cpSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const distDir = join(__dirname, '..', 'dist');
const src = join(distDir, 'index.html');
const dest = join(distDir, '404.html');

if (!existsSync(src)) {
  console.error('[copy-404] dist/index.html not found; did you run vite build?');
  process.exit(1);
}

try {
  cpSync(src, dest);
  console.log('[copy-404] Generated SPA fallback at dist/404.html');
} catch (error) {
  console.error('[copy-404] Failed to copy index.html to 404.html:', error);
  process.exit(1);
}
