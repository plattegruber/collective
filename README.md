# AR Artwork Gallery

A camera-first web app that recognizes your artwork with an ONNX detector and surfaces rich metadata overlays. The project now separates front-end sources, ML tooling, and generated artifacts so you can tell at a glance what is editable code versus build output.

## Repository Layout
- `apps/web/` – Vite-powered Svelte app (entry HTML at `index.html`, Svelte components in `src/`, deployable assets in `public/`).
- `ml/` – Dataset synthesis, training, and model-export scripts with a shared `requirements.txt`.
- `data/raw/` – Source imagery for dataset generation (`assets/` for foregrounds, `backgrounds/` for augmentation backdrops).
- `artifacts/` – Generated datasets, checkpoints, and other reproducible outputs.
- `archive/mindar/` – Legacy MindAR-based prototype kept for reference.

## Quick Start (Web Client)
1. `cd apps/web`
2. `npm install`
3. `npm run dev`
4. Visit `http://localhost:3000` (or the printed host) and grant camera access.

### Build & Preview
- `npm run build` – Outputs the static bundle to `apps/web/dist/` (GitHub Pages consumes this directory).
- `npm run serve` – Serves the production build locally for smoke-testing.

## Model Pipeline Overview
1. **Synthesize dataset**
   ```bash
   python ml/make_dataset.py --out artifacts/datasets
   ```
   The script reads labeled assets from `data/raw/` and writes COCO-style data to `artifacts/datasets/`.
2. **Train detector**
   ```bash
   python ml/train.py --data artifacts/datasets --out artifacts/models/ssdlite320
   ```
   Produces PyTorch checkpoints, ONNX exports, and label files in `artifacts/models/`.
3. **Package for the web**
   ```bash
   python ml/export_tfjs.py --onnx artifacts/models/ssdlite320/best.onnx --labels artifacts/models/ssdlite320/labels.json
   ```
   Copies the ONNX graph and labels into `apps/web/public/models/` and refreshes `apps/web/public/model-manifest.json` so the site loads the latest assets.

## Deployment Notes
- GitHub Pages expects the built site at `/collective/`. The Vite config sets `base: '/collective/'`; adjust if you host elsewhere.
- The Fly.io/Netlify `project.toml` now points at `apps/web/dist/`. Run `npm run build` before shipping.
- Treat everything in `artifacts/` as disposable; regenerate instead of committing large binaries when possible.

## Troubleshooting
- **Camera blocked** – Ensure you are on HTTPS or `localhost`, and reset browser permissions.
- **Model fails to load** – Confirm `apps/web/public/model-manifest.json` references a valid ONNX or TFJS export inside `apps/web/public/models/`.
- **Blank detections** – Double-check the label keys in `art-content.*.json` match the detector labels generated during training.
