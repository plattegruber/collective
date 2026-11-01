#!/usr/bin/env node
/**
 * Seeds a demo account through the app's public HTTP surface (so it also exercises the API).
 *
 *   npm run seed:demo -- --base http://127.0.0.1:3001 --email you@example.com --password '…'
 *
 * Thumbnails and downscaled originals are produced with macOS `sips` when available;
 * elsewhere the original file is uploaded for both.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, statSync } from 'node:fs';
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

async function main() {
	console.log(
		`Seeding ${email} at ${base}${useSips ? '' : ' (sips not found: uploading originals as thumbnails)'}`
	);
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
	const legacyLabels = JSON.parse(
		readFileSync(join(appDir, 'seed', 'legacy-model', 'labels.json'), 'utf8')
	);
	const idsByLegacyLabel = new Map();

	for (const piece of collection) {
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
		idsByLegacyLabel.set(piece.legacyLabel, artworkId);
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

	const tokenResponse = await action('/app/settings?/createToken', { name: 'seed script' });
	const token = /clv_[A-Za-z0-9_-]+/.exec(JSON.stringify(tokenResponse))?.[0];
	const labels = Object.fromEntries(
		Object.entries(legacyLabels)
			.map(([index, legacy]) => [index, idsByLegacyLabel.get(legacy)])
			.filter(([, id]) => id)
	);
	const modelPath = join(appDir, 'seed', 'legacy-model', 'model.onnx');
	const body = new FormData();
	body.set(
		'model',
		new Blob([readFileSync(modelPath)], { type: 'application/octet-stream' }),
		'model.onnx'
	);
	body.set('labels', JSON.stringify(labels));
	body.set('inputSize', '320');
	body.set('metrics', JSON.stringify({ source: 'legacy', bytes: statSync(modelPath).size }));
	const publish = await fetch(`${base}/api/ml/models`, {
		method: 'POST',
		headers: { authorization: `Bearer ${token}` },
		body
	});
	if (!publish.ok)
		throw new Error(`model publish failed (${publish.status}): ${await publish.text()}`);
	console.log(`• Published legacy model for ${Object.keys(labels).length} pieces`);
	console.log(
		`Done. Sign in as ${email} / ${password} → ${base}/app   Public gallery: ${base}/g/${handle}`
	);
}

main().catch((error) => {
	console.error(error.message);
	process.exit(1);
});
