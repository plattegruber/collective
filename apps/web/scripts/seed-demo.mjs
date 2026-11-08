#!/usr/bin/env node
/**
 * Seeds a demo account through the app's public HTTP surface (so it also exercises the API).
 *
 *   npm run seed:demo -- --base http://127.0.0.1:3001 --email you@example.com --password '…'
 *
 * Steps: publish a recognizer release (from `--release-dir`, a folder with detector.onnx,
 * embedder.onnx and release.json as written by `collective-ml release --out`, or copied from
 * another installation with `--release-from https://…`), create the account and pieces,
 * upload photos, then index them with the real browser pipeline in headless Chromium
 * (skip with `--no-index`). Thumbnails and downscaled originals are produced with macOS
 * `sips` when available; elsewhere the original file is uploaded for both.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const appDir = resolve(here, '..');
const repoRoot = resolve(appDir, '..', '..');

const args = Object.fromEntries(
	process.argv
		.slice(2)
		.map((arg, i, all) => (arg.startsWith('--') ? [arg.slice(2), all[i + 1] ?? ''] : null))
		.filter(Boolean)
);
const base = (args.base ?? 'http://127.0.0.1:3001').replace(/\/$/, '');
const email = args.email ?? 'demo@collective.local';
const password = args.password ?? 'demo-password-123';
const name = args.name ?? 'Demo Collector';
const handle = args.handle ?? 'demo';
const releaseToken = args['release-token'] ?? readDevVar('RELEASE_TOKEN');
const releaseDir = args['release-dir'] ? resolve(args['release-dir']) : null;
const releaseFrom = (args['release-from'] ?? '').replace(/\/$/, '');
const skipIndex = process.argv.includes('--no-index');

function readDevVar(name) {
	try {
		const line = readFileSync(join(appDir, '.dev.vars'), 'utf8')
			.split('\n')
			.find((l) => l.startsWith(`${name}=`));
		return line ? line.slice(name.length + 1).trim() : undefined;
	} catch {
		return undefined;
	}
}

let cookie = '';
const headers = () => ({ origin: base, ...(cookie ? { cookie } : {}) });

function rememberCookie(response) {
	const set = response.headers.get('set-cookie');
	if (set) cookie = set.split(';')[0];
}

async function action(path, fields) {
	const body = new URLSearchParams(fields);
	const response = await fetch(`${base}${path}`, {
		method: 'POST',
		headers: { ...headers(), accept: 'application/json' },
		body,
		redirect: 'manual'
	});
	rememberCookie(response);
	const payload = await response.json();
	if (payload.type === 'failure') throw new Error(`${path} failed: ${payload.data}`);
	return payload;
}

function hasSips() {
	try {
		execFileSync('sips', ['--help'], { stdio: 'ignore' });
		return true;
	} catch {
		return false;
	}
}

const tmp = mkdtempSync(join(tmpdir(), 'collective-seed-'));
const useSips = hasSips();

function resized(file, maxEdge, suffix) {
	if (!useSips) return file;
	const out = join(tmp, `${basename(file, extname(file))}-${suffix}.jpg`);
	execFileSync(
		'sips',
		[
			'-Z',
			String(maxEdge),
			'-s',
			'format',
			'jpeg',
			'-s',
			'formatOptions',
			'85',
			file,
			'--out',
			out
		],
		{ stdio: 'ignore' }
	);
	return out;
}

function dimensions(file) {
	if (!useSips) return { width: 2400, height: 2400 };
	const text = execFileSync('sips', ['-g', 'pixelWidth', '-g', 'pixelHeight', file]).toString();
	const width = Number(/pixelWidth: (\d+)/.exec(text)?.[1]);
	const height = Number(/pixelHeight: (\d+)/.exec(text)?.[1]);
	return { width, height };
}

async function uploadPhoto(artworkId, file) {
	const full = resized(file, 2400, 'full');
	const thumb = resized(file, 480, 'thumb');
	const { width, height } = dimensions(full);
	const body = new FormData();
	body.set('full', new Blob([readFileSync(full)], { type: 'image/jpeg' }), 'full.jpg');
	body.set('thumb', new Blob([readFileSync(thumb)], { type: 'image/jpeg' }), 'thumb.jpg');
	body.set('width', String(width));
	body.set('height', String(height));
	const response = await fetch(`${base}/app/artworks/${artworkId}/images`, {
		method: 'POST',
		headers: headers(),
		body
	});
	if (!response.ok) throw new Error(`upload failed (${response.status}): ${await response.text()}`);
}

/** Publishes a release unless one is already active. Returns the active manifest or null. */
async function ensureRelease() {
	const current = await fetch(`${base}/api/ml/releases`).then((r) => r.json());
	if (current.active && !releaseDir && !releaseFrom) {
		console.log(`• Recognizer ${current.active.version} already active`);
		return current.active;
	}
	let files;
	if (releaseDir) {
		const meta = JSON.parse(readFileSync(join(releaseDir, 'release.json'), 'utf8'));
		files = {
			meta,
			detector: readFileSync(join(releaseDir, 'detector.onnx')),
			embedder: readFileSync(join(releaseDir, 'embedder.onnx'))
		};
	} else if (releaseFrom) {
		const remote = await fetch(`${releaseFrom}/api/ml/releases`).then((r) => r.json());
		if (!remote.active) throw new Error(`${releaseFrom} has no active release to copy.`);
		const m = remote.active;
		const get = async (url) => Buffer.from(await (await fetch(url)).arrayBuffer());
		files = {
			meta: {
				version: m.version,
				detectorInputSize: m.detector.inputSize,
				embedderInputSize: m.embedder.inputSize,
				embedderDim: m.embedder.dim,
				config: m.config,
				notes: `Copied from ${releaseFrom}`
			},
			detector: await get(m.detector.url),
			embedder: await get(m.embedder.url)
		};
		console.log(`• Downloaded release ${m.version} from ${releaseFrom}`);
	} else {
		console.log(
			'• No recognizer release: pass --release-dir <dir> or --release-from <url> to enable the scanner.'
		);
		return null;
	}
	if (!releaseToken) throw new Error('A release token is required (--release-token or .dev.vars).');
	const body = new FormData();
	body.set('version', files.meta.version);
	body.set('detector', new Blob([files.detector]), 'detector.onnx');
	body.set('embedder', new Blob([files.embedder]), 'embedder.onnx');
	body.set('detectorInputSize', String(files.meta.detectorInputSize));
	body.set('embedderInputSize', String(files.meta.embedderInputSize));
	body.set('embedderDim', String(files.meta.embedderDim));
	if (files.meta.config) body.set('config', JSON.stringify(files.meta.config));
	if (files.meta.metrics) body.set('metrics', JSON.stringify(files.meta.metrics));
	if (files.meta.notes) body.set('notes', files.meta.notes);
	const response = await fetch(`${base}/api/ml/releases`, {
		method: 'POST',
		// Production rejects form POSTs without a matching Origin (SvelteKit's CSRF check).
		headers: { authorization: `Bearer ${releaseToken}`, origin: base },
		body
	});
	if (response.status === 409) {
		console.log(`• Release ${files.meta.version} already exists; keeping it.`);
		return current.active;
	}
	if (!response.ok)
		throw new Error(`release publish failed (${response.status}): ${await response.text()}`);
	const manifest = await response.json();
	console.log(`• Published recognizer ${manifest.version}`);
	return manifest;
}

/** Runs the in-browser indexing exactly as a collector would, from the settings page. */
async function indexPhotos() {
	const { chromium } = await import('playwright');
	const browser = await chromium.launch({ headless: true });
	try {
		const context = await browser.newContext({ baseURL: base });
		const page = await context.newPage();
		await page.goto('/login');
		await page.fill('input[name=email]', email);
		await page.fill('input[name=password]', password);
		await Promise.all([page.waitForURL(/\/app/), page.click('button[type=submit]')]);
		await page.goto('/app/settings');
		// Let hydration finish before clicking, or the click lands on static markup.
		await page.waitForLoadState('networkidle');
		const button = page.getByRole('button', { name: /^Index \d+ photos?$/ });
		if ((await button.count()) === 0) {
			console.log('• Photos already indexed');
			return;
		}
		await button.click();
		await page.getByText(/^Indexed \d+ photos?\.$/).waitFor({ timeout: 240_000 });
		console.log('• Indexed photos in the browser');
	} finally {
		await browser.close();
	}
}

/** Titles already in the account, read through the API with a throwaway token so re-runs are idempotent. */
async function existingTitles() {
	const created = await action('/app/settings?/createToken', { name: 'seed script' });
	const token = /clv_[A-Za-z0-9_-]+/.exec(JSON.stringify(created))?.[0];
	if (!token) return new Set();
	const response = await fetch(`${base}/api/ml/collection`, {
		headers: { authorization: `Bearer ${token}` }
	});
	if (!response.ok) return new Set();
	const payload = await response.json();
	return new Set(payload.artworks.map((artwork) => artwork.title));
}

async function main() {
	console.log(
		`Seeding ${email} at ${base}${useSips ? '' : ' (sips not found: uploading originals as thumbnails)'}`
	);
	const release = await ensureRelease();

	const signup = await fetch(`${base}/signup`, {
		method: 'POST',
		headers: { ...headers(), accept: 'application/json' },
		body: new URLSearchParams({ name, email, password }),
		redirect: 'manual'
	});
	rememberCookie(signup);
	const signupPayload = await signup.json();
	if (signupPayload.type === 'failure') {
		console.log('Account exists; signing in instead.');
		await action('/login', { email, password });
	}

	const collection = JSON.parse(readFileSync(join(appDir, 'seed', 'collection.json'), 'utf8'));
	const existing = await existingTitles();
	for (const piece of collection) {
		if (existing.has(piece.title)) {
			console.log(`• ${piece.title} (already present)`);
			continue;
		}
		const result = await action('/app/artworks/new', {
			title: piece.title,
			artist: piece.artist ?? '',
			year: piece.year ? String(piece.year) : '',
			medium: piece.medium ?? '',
			description: piece.description ?? '',
			status: 'owned',
			currency: 'USD',
			dimensionUnit: 'in',
			isPublic: piece.isPublic ? 'on' : ''
		});
		const artworkId = /\/artworks\/([^/?]+)/.exec(result.location)?.[1];
		const photosDir = resolve(repoRoot, piece.photosDir);
		const photos = existsSync(photosDir)
			? readdirSync(photosDir).filter((f) => /\.(jpe?g|png|webp)$/i.test(f))
			: [];
		for (const photo of photos) await uploadPhoto(artworkId, join(photosDir, photo));
		console.log(`• ${piece.title} (${photos.length} photo${photos.length === 1 ? '' : 's'})`);
	}

	await action('/app/settings?/gallery', {
		handle,
		galleryPublic: 'on',
		galleryTitle: `${name}'s collection`
	});

	if (release && !skipIndex) await indexPhotos();
	console.log(
		`Done. Sign in as ${email} / ${password} → ${base}/app   Public gallery: ${base}/g/${handle}`
	);
}

main().catch((error) => {
	console.error(error.message);
	process.exit(1);
});
