# Repository Guidelines

## Project Structure & Module Organization
The web client lives in `apps/web/`: Svelte components plus entry script in `src/`, static files in `public/`, and the shell at `index.html`. Legacy MindAR experiments sit in `archive/mindar/`. Source imagery for dataset synthesis belongs in `data/raw/` (`assets/` and `backgrounds/`). Generated output stays in `artifacts/` for datasets, checkpoints, and packaged builds. Machine-learning scripts and requirements live in `ml/` and default to those data and artifact paths.

## Build, Test, and Development Commands
Install front-end dependencies with `npm install` inside `apps/web/`. Use `npm run dev` for the Vite server, `npm run build` to emit `apps/web/dist/`, and `npm run serve` to preview that bundle. For pipeline work, create a virtual environment, `pip install -r ml/requirements.txt`, and inspect options with `python ml/make_dataset.py --help`, `python ml/train.py --help`, and `python ml/export_tfjs.py --help`.

## Coding Style & Naming Conventions
Svelte components live in single-file `.svelte` modules with `<script>`, markup, and `<style>` blocks—use 2-space indentation and keep state localized. Extract reusable logic into `src/lib/` modules as needed and avoid manual DOM access when reactive state will do. Vanilla helpers stay as ES modules with `const`/`let` and semicolons. Python code follows PEP 8 with 4-space indentation and argparse-driven CLIs. Preserve the versioned asset naming pattern (`art-content.v#.json`, `model-manifest.json`, etc.) so clients bust caches safely.

## Testing Guidelines
Run `npm run build` after major changes. When tweaking detection logic or overlays, use `npm run dev` and test on a desktop webcam plus one mobile device to confirm camera access and transitions. For ML edits, run `python ml/train.py --epochs 1 --data artifacts/datasets --out artifacts/models/debug` and ensure the exported ONNX loads with `onnxruntime.InferenceSession`.

## Commit & Pull Request Guidelines
Commits keep the imperative, Title Case summaries already in history (`Add detector overlay transition`, `Update manifest hashes`) and stay under 72 characters; add bodies only when needed. Pull requests should summarize intent, list manual verification, and flag work that requires new datasets, retraining, or CDN refreshes. Attach screenshots or recordings for UI changes.

## Asset & Model Management
Treat everything under `artifacts/` as reproducible output—clean it before commits unless a sample is intentionally tracked. Package web-ready detector assets through `apps/web/public/models/` via `ml/export_tfjs.py`. Store new capture assets in `data/raw/` using folder names that match detector labels so the CLI defaults keep working, and update manifests whenever you add or rename art IDs.
