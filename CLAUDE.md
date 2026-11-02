# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AGENTS.md holds style, testing, and commit conventions. README.md has setup and deployment steps. Follow both.

## What this is

Collective is an art inventory for personal collectors: accounts, artworks with photos and
provenance, collection stats, a public gallery per collector at `/g/<handle>`, and a phone
scanner that recognizes pieces with an on-device ONNX detector trained on the collector's
own photos. It runs as one Cloudflare Worker (SvelteKit) with D1, R2 and KV.

## Layout

- `apps/web/` – the SvelteKit 3 / Svelte 5 / Tailwind 4 app. The only npm workspace.
- `ml/` – `collective-ml`, Python package (src layout) that trains and publishes detectors through the app's API.
- `data/raw/` – photos of the original seven pieces, used only by `npm run seed:demo`.
- `.github/workflows/` – `ci.yml`, `deploy.yml` (main → Cloudflare), `train-detector.yml` (manual).

There is no test suite for UI flows; verification is `npm run lint && npm run check && npm run test && npm run build` plus clicking through.

## Commands (run in `apps/web` or via root workspace scripts)

- `npm run dev` – Vite dev server with emulated bindings (D1/R2/KV persist under `.wrangler/state`). Pick the port from the console; 3000 may be taken.
- `npm run db:migrate` – apply `drizzle/*.sql` to local D1. Required once before the app works. `db:generate` after schema edits.
- `npm run cf:types` – regenerate `worker-configuration.d.ts` after editing `wrangler.jsonc`.
- `npm run test -- --project server` – Node unit tests only (CI runs this). The `client` project needs `npx playwright install chromium`.
- `npm run seed:demo -- --base http://127.0.0.1:<port>` – creates a demo user with the legacy pieces, photos and model via HTTP.
- `cd ml && pytest && ruff check .` – ML tests need no torch.
- `node scripts/scanner-smoke.mjs --video <mjpeg> --expect "<title>"` – headless end-to-end recognition check (fake camera); `scripts/capture-marketing.mjs` regenerates landing screenshots. Both need a running dev server and the seeded account.

## SvelteKit 3 specifics (differ from most docs and training data)

- No `svelte.config.js`; adapter and kit options live in `vite.config.ts` inside `sveltekit({...})`.
- `$lib` is `#lib` and imports need extensions: `#lib/server/db/index.js`, `#lib/components/Button.svelte`.
- Cloudflare bindings: `import { env } from 'cloudflare:workers'` (typed by `worker-configuration.d.ts`), not `platform.env`.
- `$app/env` (not `$app/environment`), `Response.json()` instead of `json()`, `error(404, 'msg')` with a string, `redirect(303, url)`.
- `onDestroy`/`onMount` run during SSR in Svelte 5; browser-only lifecycle (camera, ORT, `requestAnimationFrame`) must live in `$effect`.
- ESLint rule `svelte/no-navigation-without-resolve` is configured with `ignoreLinks: true`; `goto()` still needs `resolve()`.

## Architecture notes

- **Auth**: email + password, PBKDF2 via WebCrypto (`src/lib/server/auth/password.ts`), opaque session tokens stored hashed in `sessions` (`session.ts`), sliding 30-day cookie. `hooks.server.ts` populates `locals.user`. `(app)/+layout.server.ts` calls `requireUser`. Signup can be gated with the `INVITE_CODE` secret. Account deletion (`settings?/deleteAccount`) purges the R2 prefix before the cascading row delete.
- **Data**: Drizzle + D1, schema in `src/lib/server/db/schema.ts` (`casing: 'snake_case'`). Money is integer minor units plus a currency; lengths are stored in cm with the collector's display unit. Service modules (`artworks.ts`, `images.ts`, `models.ts`, `reactions.ts`, `apiTokens.ts`) take `userId` first and scope every query.
- **Forms**: Zod 4 schemas in `src/lib/schemas`, parsed by `parseForm` (`src/lib/server/forms.ts`), which returns `fail(400, { errors, values })` for re-rendering. Use `.optional()` rather than unions with `z.undefined()`; the latter does not mark object keys optional.
- **Media**: photos are resized in the browser (`src/lib/images/prepare.ts`, full ≤2400px + 480px thumb) and POSTed to `/app/artworks/[id]/images`; stored in R2 under `users/{uid}/artworks/{aid}/…`. `/media/[...key]` serves R2 and enforces owner-or-public via `canReadMedia`, which also covers model keys `users/{uid}/models/…`.
- **Detector**: pure modules in `src/lib/detector` (`letterbox`, `decode`, `tracker` hysteresis, `preprocess`) are unit-tested; `runtime.ts` lazy-loads `onnxruntime-web` with wasm served from `/ort/` (copied by `scripts/copy-ort-assets.mjs` on `prepare`). Model contract: input `images [1,3,S,S]` float 0..1, outputs `boxes`, `scores`, `labels`. `src/lib/scanner/Scanner.svelte` is shared by `/app/scan` (owner) and `/g/[handle]/scan` (public, labels filtered to public pieces); `?debug` shows raw scores.
- **Models/ML API**: bearer tokens `clv_…` (hashed in `api_tokens`). `GET /api/ml/collection`, `GET /api/ml/media/<key>`, `GET|POST /api/ml/models`. Publishing stores the ONNX in R2 and flips `detector_models.is_active`. Labels map class index (from "1") → artwork id and are validated against ownership.
- **Reactions**: KV counters per artwork and emoji with per-visitor throttle and per-IP limits (`src/lib/server/reactions.ts`), exposed at `/api/reactions/[artworkId]` only for public-or-owned pieces.
- **Marketing**: `src/routes/+page.svelte` composed from `src/lib/marketing/*`. Real screenshots go in `static/marketing/` (see its README) and are enabled via `screenshots.ts`.

## Deployment gotchas

- `wrangler.jsonc` ships with placeholder `database_id` / KV `id`; replace them after `wrangler d1 create collective` and `wrangler kv namespace create REACTIONS`. `PUBLIC_ORIGIN` must be the public URL.
- `deploy.yml` runs lint, check, server tests, build, remote migrations, then `wrangler deploy`; needs repo secrets `CF_API_TOKEN` and `CF_ACCOUNT_ID` (token scopes: Workers Scripts, D1, R2, Workers KV Storage).
- `seed/legacy-model/model.onnx` (15 MB) is the first-generation detector for the demo seed; new models come from `collective-ml` and live in R2, never in git.
