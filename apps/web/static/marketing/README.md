# Marketing screenshots

Generated against a signed-in account with a few artworks (`npm run seed:demo` provides one):

- `node scripts/capture-marketing.mjs --base http://127.0.0.1:<port>` — app and gallery pages at desktop (2400 × 1500) and phone (1170 × 2532) sizes plus `og.jpg` (1200 × 630).
- `node scripts/scanner-smoke.mjs --base … --video <frames.mjpeg> --expect "<title>"` — drives the scanner with a fake camera and writes `scanner.jpg` once the piece is recognized.

The landing page lists them in `src/lib/marketing/screenshots.ts`; set `screenshotsAvailable = false`
there to fall back to the live mockups. Re-run the scripts after visual changes so the marketing
page never drifts from the product.
