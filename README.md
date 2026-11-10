# Collective

An inventory for the art you live with. Collectors sign in, add the pieces they own with
photos and full provenance, keep track of value and location, share a curated public
gallery, and let guests point a phone at any piece on the wall to see its story.

The phone scanner runs two small models **in the browser**: a class-agnostic artwork
detector and a vision foundation model (built with DINOv3) that turns each piece into a
compact fingerprint. Photos are indexed the moment they are uploaded, so there is nothing
to train, and the scanner keeps learning the views it sees on the collector's walls.
Camera frames never leave the device.

Photos get the same care: the photo studio finds the piece in a phone snapshot, straightens
it, crops to the frame (or the art), corrects the colour cast of the room, and removes glass
reflections from two or three shots taken from slightly different spots. The camera original
is kept alongside. All of it runs in the browser.

## Repository layout

| Path | What it is |
| --- | --- |
| `apps/web/` | The product: SvelteKit 3 + Svelte 5 + Tailwind 4, deployed as a Cloudflare Worker with D1 (database), R2 (photos and models) and KV (reactions, rate limits). |
| `ml/` | `collective-ml`, a Python package that builds the global recognizer release (detector + embedder), evaluates it end to end against a real collection, and publishes it to the app. |
| `data/raw/` | The original photos of the first collection, used only by the demo seed. |
| `.github/workflows/` | `ci.yml` (PRs), `deploy.yml` (main → Cloudflare), `build-recognizer.yml` (manual release build). |

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
secrets `CF_API_TOKEN` and `CF_ACCOUNT_ID` set and let `deploy.yml` do it.

Add `?debug` to any scanner URL to see the runtime backend, per-frame timings and the strongest raw match while pointing the phone around; `?backend=wasm` forces the fallback runtime so both paths can be compared on one device.

## Ship a recognizer release

See [`ml/README.md`](ml/README.md). A release is two global ONNX models plus a manifest;
collectors never train anything. Short version:

```bash
pip install -e './ml[train]'
collective-ml run                          # fetch open data, build both models, assemble ml/workspace/release
export COLLECTIVE_RELEASE_TOKEN=…          # the app's RELEASE_TOKEN secret
collective-ml publish ml/workspace/release
```

To evaluate a release against your own collection first, create an API token in
**Settings → API tokens**, `collective-ml pull`, then `collective-ml evaluate`; the report
suggests the match thresholds that keep false accepts under 2%, and `collective-ml release`
writes them into the manifest. **Actions → Build recognizer** runs the same build on CI.

Locally, `npm run seed:demo -- --release-dir ../../ml/workspace/release` publishes a release
into the dev database and indexes the demo photos through the real browser pipeline.
