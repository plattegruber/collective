#!/usr/bin/env node
/**
 * Captures the real product screenshots used on the landing page.
 *
 *   node scripts/capture-marketing.mjs --base http://127.0.0.1:3001 --email demo@collective.local --password '…'
 *
 * Writes JPEGs into static/marketing at the sizes documented there. Needs a signed-in
 * account with a few artworks (see `npm run seed:demo`) and Playwright's Chromium
 * (`npx playwright install chromium`).
 */
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '..', 'static', 'marketing');
mkdirSync(out, { recursive: true });

const args = Object.fromEntries(
	process.argv
		.slice(2)
		.map((arg, i, all) => (arg.startsWith('--') ? [arg.slice(2), all[i + 1] ?? ''] : null))
		.filter(Boolean)
);
const base = (args.base ?? 'http://127.0.0.1:3001').replace(/\/$/, '');
const email = args.email ?? 'demo@collective.local';
const password = args.password ?? 'demo-password-123';

const browser = await chromium.launch();

async function signedInContext(options) {
	const context = await browser.newContext({ baseURL: base, ...options });
	const response = await context.request.post('/login', {
		form: { email, password },
		headers: { origin: base, accept: 'application/json' },
		maxRedirects: 0
	});
	const payload = await response.json();
	if (payload.type !== 'redirect') throw new Error(`Sign-in failed: ${JSON.stringify(payload)}`);
	return context;
}

async function shoot(page, path, file) {
	await page.goto(path, { waitUntil: 'networkidle' });
	await page.evaluate(() => document.fonts.ready);
	await page.waitForTimeout(400);
	await page.screenshot({ path: join(out, file), type: 'jpeg', quality: 86 });
	console.log(`• ${file} ← ${path}`);
}

try {
	const desktop = await signedInContext({
		viewport: { width: 1200, height: 750 },
		deviceScaleFactor: 2
	});
	const page = await desktop.newPage();
	await shoot(page, '/app', 'dashboard.jpg');
	await shoot(page, '/app/artworks', 'artworks.jpg');
	const first = await page
		.locator('a[href^="/app/artworks/"]:not([href$="/new"])')
		.first()
		.getAttribute('href');
	if (!first) throw new Error('No artworks found; seed the account first.');
	await shoot(page, first, 'artwork.jpg');
	await shoot(page, `${first}/edit`, 'artwork-edit.jpg');
	await shoot(page, '/app/artworks/new', 'artwork-new.jpg');
	await shoot(page, '/app/settings', 'settings.jpg');
	await desktop.close();

	const signedOut = await browser.newContext({
		baseURL: base,
		viewport: { width: 1200, height: 750 },
		deviceScaleFactor: 2
	});
	const signedOutPage = await signedOut.newPage();
	await shoot(signedOutPage, '/login', 'login.jpg');
	await signedOut.close();

	const phone = await signedInContext({
		viewport: { width: 390, height: 844 },
		deviceScaleFactor: 3,
		isMobile: true,
		hasTouch: true
	});
	const phonePage = await phone.newPage();
	const handle = await phonePage
		.goto('/app/settings')
		.then(() => phonePage.inputValue('input[name="handle"]'));
	if (!handle) throw new Error('Set a gallery handle in Settings first.');
	await shoot(phonePage, `/g/${handle}`, 'gallery-phone.jpg');
	const publicPiece = await phonePage
		.locator(`a[href^="/g/${handle}/"]:not([href$="/scan"])`)
		.first()
		.getAttribute('href');
	if (publicPiece) await shoot(phonePage, publicPiece, 'artwork-phone.jpg');
	await shoot(phonePage, '/app', 'dashboard-phone.jpg');
	await shoot(phonePage, '/app/artworks', 'artworks-phone.jpg');
	await phone.close();

	const og = await browser.newContext({
		baseURL: base,
		viewport: { width: 1200, height: 630 },
		deviceScaleFactor: 1
	});
	const ogPage = await og.newPage();
	await ogPage.goto('/', { waitUntil: 'commit' });
	await ogPage.locator('h1').first().waitFor();
	await ogPage.evaluate(() => document.fonts.ready);
	await ogPage.screenshot({ path: join(out, 'og.jpg'), type: 'jpeg', quality: 88 });
	console.log('• og.jpg ← /');
	await og.close();
} finally {
	await browser.close();
}
