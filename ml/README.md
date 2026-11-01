# collective-ml

Trains a small on-device artwork detector (SSDLite320-MobileNetV3) from the photos in a
Collective account and publishes it back so the phone scanner can recognise your pieces.

## Install

Python 3.11+. Install CPU PyTorch first unless you have a GPU build you prefer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e './ml[dev]'
```

## Get a token

In the app open **Settings → API tokens**, create a token, and export it together with the
app's origin:

```bash
export COLLECTIVE_URL=https://collective.example.com
export COLLECTIVE_TOKEN=clv_…
```

Every artwork you want recognised needs at least one straight-on photo in the app.

## Run locally

```bash
collective-ml run                 # pull → synth → train → export → publish
collective-ml run --no-publish    # stop after export, inspect ml/workspace/<userId>/model/
```

Or step by step:

| Command | What it does |
| --- | --- |
| `collective-ml pull` | Downloads `collection.json` and photos into `ml/workspace/<userId>/raw/<artworkId>/`. Re-runs skip files already present. |
| `collective-ml synth` | Composites each photo onto procedurally generated wall backgrounds with perspective, lighting, blur, glare and occlusion augmentations. Writes a COCO-style dataset and `labels.json`. Pass `--backgrounds-dir` to mix in real wall photos. |
| `collective-ml train` | Fine-tunes the detector, keeps the best checkpoint by validation loss, and writes `metrics.json` (includes mAP@50 on the held-out split). |
| `collective-ml export` | Exports `model.onnx` and verifies it with onnxruntime. Input `images` `[1,3,S,S]` float32 RGB 0..1; outputs `boxes [1,N,4]` (xyxy, input pixels), `scores [1,N]`, `labels [1,N]` int64. |
| `collective-ml publish` | Uploads the model, labels and metrics; the app makes it your active recognizer. |

Useful flags: `--epochs`, `--samples-per-piece`, `--img-size` (synthetic canvas), `--train-size`
(detector input, default 320), `--batch-size`, `--workers`, `--device cuda|mps|cpu`, `--seed`.
A quick smoke run: `collective-ml run --epochs 1 --samples-per-piece 12 --no-publish`.

Class indices start at 1 (0 is background) and map to artwork ids in sorted order, so a new
artwork reshuffles the map — always publish `labels.json` together with the model (the CLI does).

## Run in CI

`.github/workflows/train-detector.yml` is a manual workflow (**Actions → Train detector → Run
workflow**). Add repository secrets `COLLECTIVE_URL` and `COLLECTIVE_TOKEN`. It trains on CPU,
publishes when the `publish` input is ticked, and uploads the model files as an artifact.

## Develop

```bash
cd ml
pytest            # unit tests; no torch needed
ruff check .      # lint
```
