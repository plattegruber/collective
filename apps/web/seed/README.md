# Seed data

`collection.json` holds the original seven pieces from the first version of this app,
with photos under `data/raw/assets/2d/<slug>/` at the repo root. `legacy-model/` is the
detector that was trained on them (class index → legacy label).

`npm run seed:demo` creates an account from these files against a running dev server,
uploads the photos, and publishes the legacy model as the account's active recognizer, so
you can try the scanner before training a new model.
