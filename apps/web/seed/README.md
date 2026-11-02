# Seed data

`collection.json` holds the original seven pieces from the first version of this app,
with photos under `data/raw/assets/2d/<slug>/` at the repo root.

`npm run seed:demo` creates an account from these files against a running dev server and
uploads the photos. Give it a recognizer release and it also publishes that release and
indexes the photos through the real browser pipeline:

```bash
# a release built with collective-ml …
npm run seed:demo -- --base http://127.0.0.1:3001 --release-dir ../../ml/releases/current
# … or the one that is live on another installation
npm run seed:demo -- --base http://127.0.0.1:3001 --release-from https://collective.example.com
```

Publishing needs the `RELEASE_TOKEN` from `.dev.vars` (or `--release-token`).
