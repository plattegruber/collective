# collective-ml

Builds, evaluates and publishes the recognizer that the Collective scanner runs on the
phone. There is nothing per collector to train any more: a release is two global models.

- **Detector** – SSDLite320-MobileNetV3 fine-tuned as a class-agnostic "there is an artwork
  here" model on composites of open paintings and real rooms.
- **Embedder** – a vision foundation model (DINOv3 ViT-S/16 by default, "Built with
  DINOv3") exported to a 224px ONNX graph with weight-only 8-bit quantisation. The browser
  turns every uploaded photo, and every piece it sees while scanning, into a 768-d vector
  and matches by cosine similarity.

Everything the app needs ships as one release: `detector.onnx`, `embedder.onnx` and
`release.json` (input sizes, normalisation, pooling, thresholds, attribution, metrics).

## Install

Python 3.11+. Evaluation and publishing need only the base dependencies; building needs
the `train` extra (install CPU PyTorch first unless you have a GPU build you prefer):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e './ml[train,dev]'
```

## Build a release

```bash
collective-ml fetch-data                 # ~400 WikiArt paintings + ~300 room photos
collective-ml build-detector --epochs 16 # compose 3200 images, train, export, verify
collective-ml build-embedder             # export + quantise DINOv3 ViT-S/16
collective-ml release --release-version 2026.11.02-1
```

Or `collective-ml run --publish` for all of it. The workspace lives in `ml/workspace/`.

## Evaluate against a real collection

Create an API token in the app (Settings → API tokens), then:

```bash
export COLLECTIVE_URL=https://collective.example.com COLLECTIVE_TOKEN=clv_…
collective-ml pull        # photos → ml/workspace/raw/<artworkId>/
collective-ml evaluate    # held-out views on rooms + unknown art as distractors
```

The report prints detection recall, top-1 accuracy, the similarity distributions, and
the thresholds that keep false accepts under 2% (match) and 0.5% (strong). Re-run
`collective-ml release` afterwards and those thresholds are written into `release.json`.

## Publish

Publishing is a maintainer action authenticated with the app's `RELEASE_TOKEN` secret
(`wrangler secret put RELEASE_TOKEN`; `.dev.vars` locally):

```bash
export COLLECTIVE_RELEASE_TOKEN=…
collective-ml publish ml/workspace/release          # or: collective-ml release --publish
```

The app activates the new release; collectors' photos are re-indexed in their browsers
the next time they open the scanner or the settings page.

## Develop

```bash
cd ml
pytest            # unit tests; no torch needed
ruff check . && ruff format --check .
```
