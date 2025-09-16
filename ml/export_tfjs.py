#!/usr/bin/env python3
"""
export_tfjs.py — Convert detector ONNX -> TF.js graph format (if toolchain present),
else package the ONNX for use with onnxruntime-web. Writes a model manifest with
checksums so your frontend can load the right runtime.

Two output modes:
  1) TFJS:   Uses onnx-tf -> SavedModel -> tensorflowjs_converter -> apps/web/public/models/detector/
  2) ONNXWEB: Copies ONNX to apps/web/public/models/detector/model.onnx for onnxruntime-web.

Usage:
  python export_tfjs.py \
    --onnx artifacts/models/ssdlite320/best.onnx \
    --labels artifacts/models/ssdlite320/labels.json \
    --out apps/web/public/models \
    --prefer tfjs   # or onnxweb

Deps (mode-dependent):
  Common:   pip install numpy tqdm
  TFJS:     pip install onnx onnx-tf tensorflow tensorflowjs
  ONNXWEB:  pip install onnx  (for sanity-check)

Frontend loader strategy (pseudocode):
  - Read public/model-manifest.json
  - If manifest["format"] == "tfjs":  tf.loadGraphModel(manifest.path)
  - If manifest["format"] == "onnx":  ort.InferenceSession(manifest.path, {executionProviders:['wasm','webgl','webgpu']})

Note: Converting ONNX->TF SavedModel is not guaranteed for all op sets.
If conversion fails, the script automatically falls back to ONNXWEB unless
--prefer tfjs --strict is used.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import onnx  # type: ignore
except Exception as e:
    onnx = None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_public: Path, fmt: str, rel_path: str, labels_rel: str, extra: dict | None = None):
    out_public.parent.mkdir(parents=True, exist_ok=True)
    version = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    manifest = {
        'version': version,
        'format': fmt,                  # 'tfjs' or 'onnx'
        'path': rel_path,               # web-served relative path
        'labels': labels_rel,
    }
    if extra:
        manifest.update(extra)
    out_public.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest → {out_public}")


def convert_onnx_to_tfjs(onnx_path: Path, tfjs_dir: Path, tmpdir: Path) -> bool:
    """Try ONNX -> TF SavedModel (onnx-tf), then SavedModel -> TFJS (tensorflowjs_converter)."""
    try:
        import onnx
        from onnx_tf.backend import prepare  # type: ignore
    except Exception as e:
        print("[export] onnx-tf not available:", e)
        return False

    saved_model_dir = tmpdir / 'saved_model'
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] Loading ONNX: {onnx_path}")
    model = onnx.load(str(onnx_path))

    print("[export] Converting ONNX → TF (SavedModel)…")
    tf_rep = prepare(model)
    # Export SavedModel
    try:
        tf_rep.export_graph(str(saved_model_dir))
    except Exception as e:
        print("[export] Failed exporting SavedModel:", e)
        return False

    print("[export] Converting SavedModel → TFJS graph…")
    tfjs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'tensorflowjs.converters.converter',
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        '--signature_name', 'serving_default',
        '--saved_model_tags', 'serve',
        str(saved_model_dir), str(tfjs_dir)
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("[export] tensorflowjs_converter failed:", e)
        return False

    model_json = tfjs_dir / 'model.json'
    ok = model_json.exists()
    if ok:
        print(f"[export] TFJS graph written at {model_json}")
    else:
        print("[export] TFJS model.json missing — conversion likely failed")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', type=Path, required=True)
    ap.add_argument('--labels', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('apps/web/public/models'), help='Root models directory for packaged detector assets')
    ap.add_argument('--public', type=Path, default=Path('apps/web/public'), help='Public web root for manifest')
    ap.add_argument('--prefer', choices=['tfjs', 'onnxweb'], default='tfjs')
    ap.add_argument('--strict', action='store_true', help='Fail if preferred path fails (no fallback)')
    args = ap.parse_args()

    # Basic checks
    if not args.onnx.exists():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    if onnx is None:
        print('[warn] onnx python package not found; will not validate graph structure.')

    models_root = args.out
    tfjs_target = models_root / 'detector'
    tfjs_target.mkdir(parents=True, exist_ok=True)

    tmpdir = Path('.tmp_export_tfjs')
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Copy labels alongside model folder for convenience
    labels_target = models_root / 'detector' / 'labels.json'
    shutil.copy2(args.labels, labels_target)

    # Attempt preferred path first
    success = False
    if args.prefer == 'tfjs':
        success = convert_onnx_to_tfjs(args.onnx, tfjs_target, tmpdir)
        if success:
            # Compute checksums for TFJS files
            model_json = tfjs_target / 'model.json'
            # Gather weight shards
            weights = sorted(tfjs_target.glob('group*-shard*of*'))
            # Write manifest for TFJS
            extra = {
                'sha256': {
                    'model.json': sha256_of(model_json) if model_json.exists() else None,
                    'weights': {p.name: sha256_of(p) for p in weights}
                }
            }
            rel_path = str((Path('models') / 'detector' / 'model.json').as_posix())
            labels_rel = str((Path('models') / 'detector' / 'labels.json').as_posix())
            write_manifest(args.public / 'model-manifest.json', 'tfjs', rel_path, labels_rel, extra)
            print('[export] Completed TFJS export.')
        elif args.strict:
            raise SystemExit('[export] TFJS conversion failed and --strict set.')

    if not success:
        # Fallback to ONNXWEB: just copy ONNX to models/detector/model.onnx
        onnx_target = models_root / 'detector' / 'model.onnx'
        shutil.copy2(args.onnx, onnx_target)
        extra = {'sha256': sha256_of(onnx_target)}
        rel_path = str((Path('models') / 'detector' / 'model.onnx').as_posix())
        labels_rel = str((Path('models') / 'detector' / 'labels.json').as_posix())
        write_manifest(args.public / 'model-manifest.json', 'onnx', rel_path, labels_rel, extra)
        print('[export] Packaged ONNX for onnxruntime-web.')

    # Cleanup tmp
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass


if __name__ == '__main__':
    main()
