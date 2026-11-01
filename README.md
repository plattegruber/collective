# Collective

An inventory for the art you live with. Collectors sign in, add the pieces they own with
photos and full provenance, keep track of value and location, share a curated public
gallery, and let guests point a phone at any piece on the wall to see its story.

The phone scanner runs a small object detector **in the browser**, trained on the
collector's own photos. Camera frames never leave the device.

## Repository layout

| Path | What it is |
| --- | --- |
| `apps/web/` | The product: SvelteKit 3 + Svelte 5 + Tailwind 4, deployed as a Cloudflare Worker with D1 (database), R2 (photos and models) and KV (reactions, rate limits). |
| `ml/` | `collective-ml`, a Python package that pulls a collection through the API, synthesises a training set, trains an SSDLite320-MobileNetV3 detector, exports ONNX and publishes it back. |
| `data/raw/` | The original photos of the first collection, used only by the demo seed. |
| `.github/workflows/` | `ci.yml` (PRs), `deploy.yml` (main → Cloudflare), `train-detector.yml` (manual training run). |

## Develop

Requires Node 22.17+ and npm. The repo is an npm workspace; run everything from the root or inside `apps/web`.

```bash
npm install
npm run dev                      # http://127.0.0.1:3000 with emulated D1/R2/KV
```

First time only, create the local database:

```bash
cd apps/web
npm run db:migrate               # applies drizzle/ migrations to the local D1
```

Optional: seed a demo account with the original seven pieces, their photos, and the legacy detector so the scanner works immediately.

```bash
npm run seed:demo -- --base http://127.0.0.1:3000 --email you@example.com --password 'choose-one'
```

### Everyday commands (`apps/web`)

| Command | Purpose |
| --- | --- |
| `npm run check` | svelte-check (TypeScript) |
| `npm run lint` / `npm run format` | Prettier + ESLint |
| `npm run test` | Vitest. Pure logic runs in Node; `*.svelte.test.ts` run in headless Chromium (`npx playwright install chromium` once). |
| `npm run db:generate` | Create a migration after editing `src/lib/server/db/schema.ts` |
| `npm run db:migrate` | Apply migrations locally (`db:migrate:remote` for production) |
| `npm run cf:types` | Regenerate `worker-configuration.d.ts` after editing `wrangler.jsonc` |
| `npm run build` / `npm run preview` | Production build / preview with the Cloudflare adapter |
| `node scripts/capture-marketing.mjs --base <url>` | Regenerate the landing-page screenshots from a seeded account |
| `node scripts/scanner-smoke.mjs --base <url> --video <frames.mjpeg> --expect "<title>"` | Drive the scanner with a fake camera and assert a piece is recognized |

## Deploy to Cloudflare

One-time setup with [Wrangler](https://developers.cloudflare.com/workers/wrangler/):

```bash
cd apps/web
npx wrangler login
npx wrangler d1 create collective            # paste database_id into wrangler.jsonc
npx wrangler r2 bucket create collective-media
npx wrangler kv namespace create REACTIONS   # paste id into wrangler.jsonc
npx wrangler secret put INVITE_CODE          # optional: makes sign-up invite-only
```

Set `vars.PUBLIC_ORIGIN` in `wrangler.jsonc` to the public URL, then either run
`npm run db:migrate:remote && npm run deploy`, or push to `main` with the repository
secrets `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` set and let `deploy.yml` do it.

Add `?debug` to any scanner URL to see the strongest raw detection and the smoothed confidence while pointing the phone around.

## Train a recognizer

See [`ml/README.md`](ml/README.md). Short version: create an API token in **Settings → API tokens**, then

```bash
export COLLECTIVE_URL=https://your.domain COLLECTIVE_TOKEN=clv_…
pip install -e './ml' && collective-ml run
```

or trigger **Actions → Train detector** with those two values stored as repository secrets.
