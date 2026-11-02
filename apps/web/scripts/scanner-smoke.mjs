#!/usr/bin/env node
/**
 * End-to-end scanner check without a phone: Chromium's fake camera plays a video file of a
 * known artwork, the page loads the account's active model, and we wait for the overlay to
 * name the piece. Also saves static/marketing/scanner.jpg.
 *
 *   node scripts/scanner-smoke.mjs --base http://127.0.0.1:3001 --video /path/to/frames.mjpeg --expect "Memory Surface"
 */
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
	process.argv
		.slice(2)
		.map((arg, i, all) => (arg.startsWith('--') ? [arg.slice(2), all[i + 1] ?? ''] : null))
		.filter(Boolean)
);
const base = (args.base ?? 'http://127.0.0.1:3001').replace(/\/$/, '');
const email = args.email ?? 'demo@collective.local';
const password = args.password ?? 'demo-password-123';
const video = args.video;
const expect = args.expect;
const expectUnknown = process.argv.includes('--expect-unknown');
if (!video || (!expect && !expectUnknown))
	throw new Error('--video and --expect <title> (or --expect-unknown) are required');

const browser = await chromium.launch({
	args: [
		'--use-fake-ui-for-media-stream',
		'--use-fake-device-for-media-stream',
		`--use-file-for-fake-video-capture=${video}`
	]
});
try {
	const context = await browser.newContext({
		baseURL: base,
		viewport: { width: 390, height: 844 },
		deviceScaleFactor: 3,
		isMobile: true,
		hasTouch: true,
		permissions: ['camera']
	});
	const login = await context.request.post('/login', {
		form: { email, password },
		headers: { origin: base, accept: 'application/json' },
		maxRedirects: 0
	});
	if ((await login.json()).type !== 'redirect') throw new Error('Sign-in failed');

	const page = await context.newPage();
	const errors = [];
	page.on('pageerror', (error) => errors.push(error.message));
	page.on('console', (message) => {
		if (message.type() === 'error') errors.push(message.text());
	});

	await page.goto('/app/scan', { waitUntil: 'commit' });
	const enable = page.getByRole('button', { name: 'Enable camera' });
	if (await enable.isVisible({ timeout: 3000 }).catch(() => false)) await enable.click();

	const started = Date.now();
	if (expectUnknown)
		await page.getByText('Not in your collection yet').waitFor({ timeout: 60_000 });
	else await page.getByRole('heading', { name: expect }).waitFor({ timeout: 60_000 });
	const seconds = ((Date.now() - started) / 1000).toFixed(1);
	await page.waitForTimeout(600);
	await page.screenshot({
		path: join(
			here,
			'..',
			'static',
			'marketing',
			expectUnknown ? 'scanner-unknown.jpg' : 'scanner.jpg'
		),
		type: 'jpeg',
		quality: 86
	});
	console.log(
		`${expectUnknown ? 'Flagged an unknown piece' : `Recognized "${expect}"`} after ${seconds}s; overlay shown. Console errors: ${errors.length}`
	);
	for (const error of errors) console.log('  ! ' + error);
	process.exitCode = errors.length ? 1 : 0;
} finally {
	await browser.close();
}
