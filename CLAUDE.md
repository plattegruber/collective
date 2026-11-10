# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AGENTS.md holds style, testing, and commit conventions. README.md has setup and deployment steps. Follow both.

## What this is

Collective is an art inventory for personal collectors: accounts, artworks with photos and
provenance, collection stats, a public gallery per collector at `/g/<handle>`, and a phone
scanner that recognizes pieces on the wall with two on-device ONNX models: a class-agnostic
artwork detector and a DINOv3 embedder matched against fingerprints of the collector's own
photos. Nothing is trained per collector. It runs as one Cloudflare Worker (SvelteKit) with D1, R2 and KV.

## Layout

- `apps/web/` – the SvelteKit 3 / Svelte 5 / Tailwind 4 app. The only npm workspace.
- `ml/` – `collective-ml`, Python package (src layout) that builds, evaluates and publishes the global recognizer release (detector + embedder).
- `data/raw/` – photos of the original seven pieces, used only by `npm run seed:demo`.
- `.github/workflows/` – `ci.yml`, `deploy.yml` (main → Cloudflare), `build-recognizer.yml` (manual release build).

There is no test suite for UI flows; verification is `npm run lint && npm run check && npm run test && npm run build` plus clicking through.

## Commands (run in `apps/web` or via root workspace scripts)

- `npm run dev` – Vite dev server with emulated bindings (D1/R2/KV persist under `.wrangler/state`). Pick the port from the console; 3000 may be taken.
- `npm run db:migrate` – apply `drizzle/*.sql` to local D1. Required once before the app works. `db:generate` after schema edits.
- `npm run cf:types` – regenerate `worker-configuration.d.ts` after editing `wrangler.jsonc` (`check` does this too; the script strips Wrangler's `mainModule` line, which would otherwise pull the whole build output into svelte-check). Changing the D1/KV ids in `wrangler.jsonc` also changes which local `.wrangler/state` database the dev server opens, so re-run `db:migrate` and `seed:demo` or copy the old SQLite file.
- `npm run test -- --project server` – Node unit tests only (CI runs this). The `client` project needs `npx playwright install chromium`.
- `npm run seed:demo -- --base http://127.0.0.1:<port> --release-dir ../../ml/workspace/release` – publishes a recognizer release (needs `RELEASE_TOKEN` in `.dev.vars`), creates a demo user with the legacy pieces and photos via HTTP, and indexes the photos in headless Chromium. `--release-from <url>` copies another installation's live release instead.
- `cd ml && pytest && ruff check .` – ML tests need no torch.
- `node scripts/scanner-smoke.mjs --video <mjpeg> --expect "<title>"` – headless end-to-end recognition check (fake camera); `scripts/capture-marketing.mjs` regenerates landing screenshots. Both need a running dev server and the seeded account.

## SvelteKit 3 specifics (differ from most docs and training data)

- No `svelte.config.js`; adapter and kit options live in `vite.config.ts` inside `sveltekit({...})`.
- `$lib` is `#lib` and imports need extensions: `#lib/server/db/index.js`, `#lib/components/Button.svelte`.
- Cloudflare bindings: `import { env } from 'cloudflare:workers'` (typed by `worker-configuration.d.ts`), not `platform.env`.
- `$app/env` (not `$app/environment`), `Response.json()` instead of `json()`, `error(404, 'msg')` with a string, `redirect(303, url)`.
- `onDestroy` runs during SSR in Svelte 5 (`onMount` does not); browser-only lifecycle (camera, ORT, `requestAnimationFrame`, fetches) lives in `$effect`, which never runs on the server.
- ESLint rule `svelte/no-navigation-without-resolve` is configured with `ignoreLinks: true`; `goto()` still needs `resolve()`.

## Architecture notes

- **Auth**: email + password, PBKDF2 via WebCrypto (`src/lib/server/auth/password.ts`), opaque session tokens stored hashed in `sessions` (`session.ts`), sliding 30-day cookie. `hooks.server.ts` populates `locals.user`. `(app)/+layout.server.ts` calls `requireUser`. Signup can be gated with the `INVITE_CODE` secret. Account deletion (`settings?/deleteAccount`) purges the R2 prefix before the cascading row delete.
- **Data**: Drizzle + D1, schema in `src/lib/server/db/schema.ts` (`casing: 'snake_case'`). Money is integer minor units plus a currency; lengths are stored in cm with the collector's display unit. Service modules (`artworks.ts`, `images.ts`, `models.ts`, `reactions.ts`, `apiTokens.ts`) take `userId` first and scope every query.
- **Forms**: Zod 4 schemas in `src/lib/schemas`, parsed by `parseForm` (`src/lib/server/forms.ts`), which returns `fail(400, { errors, values })` for re-rendering. Use `.optional()` rather than unions with `z.undefined()`; the latter does not mark object keys optional.
- **Media**: photos go through the photo studio (`src/lib/components/PhotoStudio.svelte`, logic in `src/lib/images/`): `studio.ts` decodes the file (≤3200px), finds the piece with the recognizer's detector when a release exists, `quad.ts` locates the four edges near the box (candidate lines per side scored by gradient, chosen jointly so their outside strips share one colour, nudged to the outer edge so the frame stays; `prefer: 'inner'` crops to the art; falls back to the box when the weakest side is unconvincing), `warp.ts` straightens by inverse homography (`geometry.ts`), `enhance.ts` white-balances on bright neutral surfaces and lifts the black point conservatively, `fuse.ts` removes glass reflections from several shots taken from slightly different spots (shots straightened to one rectangle, exposure-matched, aligned globally then per tile on edge images; the cleanest shot is the reference and darker shots take over where it is brighter). Extra shots inherit the lead shot's corners via `transferQuad` and only snap locally, otherwise fusing smears the frame. `prepare.ts` encodes full (≤3200px, JPEG 0.9) + 480px thumb and keeps the camera original; all three are POSTed to `/app/artworks/[id]/images` and stored in R2 under `users/{uid}/artworks/{aid}/…` (`original_key` nullable). `/media/[...key]` serves R2 and enforces owner-or-public via `canReadMedia`, which also covers model keys `users/{uid}/models/…`. The scanner's "Add piece" hands a whole still (`scanner/still.ts`, `ImageCapture.takePhoto` where available) to the studio rather than a video-frame crop.
- **Recognition (browser)**: `src/lib/recognition`. Pure, unit-tested modules: `letterbox`, `decode` (class-agnostic regions), `boxes`, `preprocess` (`rgbaToChwAutoLevels`: gray-world white balance + percentile stretch of the detector input, padding excluded; the detector's confidence collapses in dim tungsten light without it, and `collective_ml.runtime.auto_levels` mirrors it so evaluation matches), `matcher` (`PrototypeIndex`: cosine nearest-prototype, best prototype per artwork; `similarityTo` for a named piece), `regionTracker` (IoU association + box smoothing; tracks expire after `maxMissMs`, not a frame count), `tracker` (hysteresis on the final answer with a time constant and half-life, so phones at 3 fps behave like desktops at 30), `signature` (8×8 luminance thumbnails: a region whose content changed is re-embedded at once), `vectors`. Runtime: `ort.ts` picks the JSPI build with the WebGPU provider when `navigator.gpu` hands out an adapter and `WebAssembly.Suspending` exists, else the wasm build proxied to a worker (`env.wasm.proxy`, so inference never blocks the camera or overlay; proxied sessions take ownership of input buffers, hence `transfersInputs`); both builds are staged into `static/ort` by `scripts/copy-ort-assets.mjs`, threads need the COOP/COEP headers set in `hooks.server.ts` and the `_headers` file. `assets.ts` caches model downloads in the Cache API and checks SHA-256; `detector.ts`/`embedder.ts` wrap the sessions (`FrameBuffer` letterboxes, the embedder pads with the dataset mean so padding standardises to zero; pooling `cls+mean` then L2); `recognizer.ts` runs detect (at half the detect threshold: detections below it are *weak* candidates that need a `strong` match) → track → embed one due region per frame (never-embedded first, then strong, then the biggest central region; unmatched regions retry every 350 ms, matched ones refresh every 900 ms or when their signature changes) → match → hysteresis; with nothing detected for 600 ms it embeds the frame centre as a guess. A region that stops matching its previous piece feeds that piece's low similarity back as contradicting evidence, so the overlay drops in one update instead of decaying. `teach.ts` embeds uploaded photos (`locateArtwork`: union of detections covering ≥ 20% of the image, else whole photo) and POSTs prototypes, `rebuildIndex` fills gaps. `loader.ts` keeps one loaded release per page; `?backend=wasm|webgpu` on a scan page forces a runtime and `?debug` shows timings. `Scanner.svelte` draws outlines for strong or matched regions over the `object-cover` video, stores scan memories for owners after a 2.5 s stable match, and offers "Add piece" for unknown art (crop handed over via `scanner/capture.ts` + sessionStorage to the new-artwork flow).
- **Recognition (server)**: `src/lib/server/recognition.ts`. `recognizer_releases` holds the global model pair (R2 keys `models/{id}/{detector,embedder}.onnx`, served by `/models/[...key]` immutable; config = `RecognizerConfig` in `src/lib/domain/recognition.ts`); exactly one is active. `artwork_prototypes` holds float32 vectors (little-endian blobs) per photo (`source: photo`, cascades with the image) or scan memory (`source: scan`, capped by `maxMemories`, deduplicated by `memoryNovelty`). Scan pages load the manifest plus `toIndexPayload` (base64url vectors) scoped to what the viewer may see; `/app/recognizer` reports which photos still lack prototypes. Releases are published by `POST /api/ml/releases` with `Authorization: Bearer $RELEASE_TOKEN` (secret; `.dev.vars` locally) and activated via `/api/ml/releases/[id]/activate`.
- **ML API for collectors**: bearer tokens `clv_…` (hashed in `api_tokens`): `GET /api/ml/collection`, `GET /api/ml/media/<key>`, used by `collective-ml pull` for evaluation.
- **Reactions**: KV counters per artwork and emoji with per-visitor throttle and per-IP limits (`src/lib/server/reactions.ts`), exposed at `/api/reactions/[artworkId]` only for public-or-owned pieces.
- **Marketing**: `src/routes/+page.svelte` composed from `src/lib/marketing/*`. Real screenshots go in `static/marketing/` (see its README) and are enabled via `screenshots.ts`.

## Deployment gotchas

- `wrangler.jsonc` ships with placeholder `database_id` / KV `id`; replace them after `wrangler d1 create collective` and `wrangler kv namespace create REACTIONS`. `PUBLIC_ORIGIN` must be the public URL.
- `deploy.yml` runs lint, check, server tests, build, remote migrations, then `wrangler deploy`; needs repo secrets `CF_API_TOKEN` and `CF_ACCOUNT_ID` (token scopes: Workers Scripts, D1, R2, Workers KV Storage).
- Model files live in R2 and `ml/workspace/` (gitignored), never in git. Changing the recognizer release invalidates nothing server-side: prototypes are keyed by release and rebuilt lazily in collectors' browsers.
