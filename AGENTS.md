# Repository Guidelines

## Project Structure & Module Organization
The web client lives in `apps/web/`: ship-ready files in `public/`, the HTML entry in `index.html`, and runtime code in `src/`. Legacy MindAR experiments were moved to `archive/mindar/` for reference. Source imagery for dataset synthesis belongs in `data/raw/` (split into `assets/` and `backgrounds/`). Generated output now sits in `artifacts/`, which houses datasets, checkpoints, and packaged builds. Machine-learning scripts and requirements live in `ml/` and default to those data and artifact paths.

## Build, Test, and Development Commands
Install front-end dependencies with `npm install` inside `apps/web/`. Use `npm run dev` for the local Vite server, `npm run build` to emit `apps/web/dist/`, and `npm run serve` to preview that bundle. For pipeline work, create a virtual environment, `pip install -r ml/requirements.txt`, and inspect available options with `python ml/make_dataset.py --help`, `python ml/train.py --help`, and `python ml/export_tfjs.py --help`.

## Coding Style & Naming Conventions
JavaScript stays in ES module form with `const`/`let`, 2-space indentation, and semicolons. Keep identifiers camelCase, promote reusable UI helpers to PascalCase, and favor named exports. Python code follows PEP 8 with 4-space indentation and argparse-driven CLIs; add type hints when extending the pipeline. Preserve the versioned asset naming pattern (`art-content.v#.json`, `model-manifest.json`, etc.) so clients bust caches safely.

## Testing Guidelines
Run `npm run build` after every substantive change so Vite and Rollup catch syntax or bundling regressions. When tweaking detection logic or overlays, exercise `npm run dev` and test on at least one mobile device plus a desktop webcam to confirm camera permissions and overlay transitions. For ML edits, run a smoke test such as `python ml/train.py --epochs 1 --data artifacts/datasets --out artifacts/models/debug` and ensure exported ONNX graphs load with `onnxruntime.InferenceSession`.

## Commit & Pull Request Guidelines
Commits keep the imperative, Title Case summaries already in history (`Add detector overlay transition`, `Update manifest hashes`). Stay under 72 characters and include bodies when extra context matters. Pull requests should summarize intent and list manual verification. Flag when teammates must regenerate datasets, retrain models, or refresh CDN assets, and attach screenshots or recordings for UI-facing changes.

## Asset & Model Management
Treat everything under `artifacts/` as reproducible output—clean it before commits unless a sample is intentionally tracked. Package web-ready detector assets through `apps/web/public/models/` via `ml/export_tfjs.py`. Store new capture assets in `data/raw/` using folder names that match detector labels so the CLI defaults keep working, and update manifests whenever you add or rename art IDs.
