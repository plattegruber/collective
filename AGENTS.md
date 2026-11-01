# Repository Guidelines

## Layout
`apps/web/` is the SvelteKit app (routes in `src/routes`, shared code in `src/lib`,
server-only code in `src/lib/server`). `ml/` is the Python training package with a `src`
layout and tests in `ml/tests`. Generated files (`drizzle/*.sql`, `worker-configuration.d.ts`,
`static/ort/`) are produced by scripts; do not hand-edit them.

## Commands
Web: `npm run dev`, `npm run check`, `npm run lint`, `npm run test`, `npm run build` from the
repo root or `apps/web`. Database: edit `src/lib/server/db/schema.ts`, then `npm run db:generate`
and `npm run db:migrate`. Bindings: edit `wrangler.jsonc`, then `npm run cf:types`.
ML: `cd ml && pytest && ruff check .`.

## Style
TypeScript strict, Svelte 5 runes only (`$state`, `$derived`, `$props`, snippets), no legacy
`export let` or stores in components. Imports use `#lib/...` with explicit `.js`/`.svelte`
extensions (SvelteKit 3). Tailwind utility classes with the tokens from `src/routes/layout.css`;
no ad-hoc hex colours. Prettier (tabs, single quotes, 100 cols) and ESLint are the source of
truth; run `npm run format` before committing. Python follows ruff defaults with type hints and
docstrings on public functions.

## Server rules
Bindings come from `import { env } from 'cloudflare:workers'` and only inside
`src/lib/server` or `+server.ts`/`+page.server.ts`. Every query that reads or mutates a
collector's data must be scoped by `userId`. Public pages receive explicit allow-listed
fields; prices, notes, seller and location never reach a public route. Validate form input
with the Zod schemas in `src/lib/schemas` through `parseForm`.

## Testing
Pure modules (`src/lib/detector`, `src/lib/utils`, schema helpers) get Vitest unit tests next
to the file (`*.test.ts`). Component tests use `*.svelte.test.ts` and run in Chromium. Before a
PR: `npm run lint && npm run check && npm run test && npm run build`, and click through sign-in,
add artwork, upload a photo, and the scanner on a phone when touching those areas.

## Commits and PRs
Imperative, Title Case summaries under 72 characters (`Add artwork CSV export`). Describe
manual verification in the PR body; attach screenshots for UI changes; call out schema
migrations and anything that needs a Cloudflare resource or secret.
