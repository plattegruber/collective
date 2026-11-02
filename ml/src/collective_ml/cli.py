"""`collective-ml` command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from collective_ml import __version__
from collective_ml.config import (
    ApiSettings,
    ConfigError,
    ReleaseSettings,
    Workspace,
    default_workspace_root,
)


def _add_workspace(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace_root(),
        help="Build folder (default: ml/workspace)",
    )


def _add_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url", help="App origin, e.g. https://collective.example.com (or $COLLECTIVE_URL)"
    )


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=3200, help="Composite images to build")
    parser.add_argument("--img-size", type=int, default=640, help="Composite canvas size")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--train-size", type=int, default=320, help="Detector input size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", help="cuda, mps, or cpu (auto by default)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")


def _add_embedder_args(parser: argparse.ArgumentParser) -> None:
    from collective_ml.embedder import DEFAULT_EMBEDDER

    parser.add_argument("--model", default=DEFAULT_EMBEDDER, help="timm model name")
    parser.add_argument("--embed-size", type=int, default=224, help="Embedder input size")
    parser.add_argument("--pooling", choices=["cls", "mean", "cls+mean"], default="cls+mean")
    parser.add_argument("--no-quantize", action="store_true", help="Ship fp32 weights")


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--per-piece", type=int, default=40, help="Held-out views per artwork")
    parser.add_argument("--distractor-images", type=int, default=150)
    parser.add_argument("--eval-seed", type=int, default=4242)


def _add_release_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--release-version", help="e.g. 2026.11.02-1 (default: today's date)")
    parser.add_argument("--notes", default="", help="Free-form release notes")
    parser.add_argument("--out", type=Path, help="Write the release folder here")
    parser.add_argument("--publish", action="store_true", help="Upload to the app")
    parser.add_argument("--release-token", help="App RELEASE_TOKEN (or $COLLECTIVE_RELEASE_TOKEN)")
    parser.add_argument("--no-activate", action="store_true", help="Publish without activating")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree."""
    parser = argparse.ArgumentParser(
        prog="collective-ml", description="Build, evaluate and publish Collective recognizers."
    )
    parser.add_argument("--version", action="version", version=f"collective-ml {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="Download a collection's photos for evaluation")
    _add_url(p_pull)
    p_pull.add_argument("--token", help="Collector API token (or $COLLECTIVE_TOKEN)")
    _add_workspace(p_pull)

    p_data = sub.add_parser("fetch-data", help="Download open paintings and room photos")
    _add_workspace(p_data)
    p_data.add_argument("--foregrounds", type=int, default=400)
    p_data.add_argument("--rooms", type=int, default=300)

    p_det = sub.add_parser(
        "build-detector", help="Compose a dataset, train and export the detector"
    )
    _add_workspace(p_det)
    _add_train_args(p_det)
    p_det.add_argument("--skip-compose", action="store_true", help="Reuse the existing dataset")

    p_emb = sub.add_parser("build-embedder", help="Export and quantise the embedding backbone")
    _add_workspace(p_emb)
    _add_embedder_args(p_emb)

    p_eval = sub.add_parser("evaluate", help="Score a release against pulled photos")
    _add_workspace(p_eval)
    _add_eval_args(p_eval)
    p_eval.add_argument("--release-dir", type=Path, help="Defaults to the workspace release")

    p_rel = sub.add_parser("release", help="Bundle detector + embedder (and publish)")
    _add_workspace(p_rel)
    _add_url(p_rel)
    _add_release_args(p_rel)

    p_pub = sub.add_parser("publish", help="Upload an assembled release folder")
    _add_url(p_pub)
    p_pub.add_argument("release_dir", type=Path)
    p_pub.add_argument("--release-token")
    p_pub.add_argument("--no-activate", action="store_true")

    p_run = sub.add_parser(
        "run", help="fetch-data, build-detector, build-embedder, evaluate, release"
    )
    _add_workspace(p_run)
    _add_url(p_run)
    _add_train_args(p_run)
    _add_embedder_args(p_run)
    _add_eval_args(p_run)
    _add_release_args(p_run)
    p_run.add_argument("--foregrounds", type=int, default=400)
    p_run.add_argument("--rooms", type=int, default=300)
    return parser


def _build_detector(workspace: Workspace, args: argparse.Namespace) -> None:
    from collective_ml import pipeline
    from collective_ml.composites import CompositeOptions

    workspace.ensure()
    if not args.skip_compose:
        n = pipeline.compose_detector_dataset(
            workspace, CompositeOptions(size=args.img_size, count=args.count, seed=args.seed)
        )
        print(f"composed {n} images")
    metrics = pipeline.train_detector(
        workspace,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.train_size,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
    )
    print(f"detector mAP@50 {float(metrics.get('map50', 0)):.3f} → {workspace.detector_onnx}")


def _build_embedder(workspace: Workspace, args: argparse.Namespace) -> None:
    from collective_ml.embedder import EmbedderSpec, export_embedder

    info = export_embedder(
        EmbedderSpec(
            model=args.model,
            input_size=args.embed_size,
            pooling=args.pooling,
            quantize=not args.no_quantize,
        ),
        workspace.embedder_dir,
    )
    print(
        f"embedder {info.model} dim {info.dim} ({info.bytes / 1e6:.1f} MB) "
        f"→ {workspace.embedder_onnx}"
    )


def _release(workspace: Workspace, args: argparse.Namespace) -> Path:
    import datetime as dt

    from collective_ml.release import assemble_release, publish_release

    version = args.release_version or dt.date.today().strftime("%Y.%m.%d")
    release_dir = assemble_release(
        workspace,
        version,
        detector_input_size=getattr(args, "train_size", 320),
        notes=args.notes,
        out_dir=args.out,
    )
    print(f"release {version} → {release_dir}")
    if args.publish:
        settings = ReleaseSettings.from_env(args.url, args.release_token)
        payload = publish_release(settings, release_dir, activate=not args.no_activate)
        print(f"published {payload['version']} ({payload['id']}) to {settings.base_url}")
    return release_dir


def _evaluate(workspace: Workspace, args: argparse.Namespace) -> None:
    from collective_ml.evaluate import EvalOptions, evaluate_release, format_report

    release_dir = getattr(args, "release_dir", None) or workspace.release_dir
    if not (release_dir / "release.json").exists():
        raise SystemExit(f"{release_dir} has no release.json. Run `collective-ml release` first.")
    report = evaluate_release(
        release_dir,
        workspace.raw_dir,
        workspace.foregrounds_dir,
        workspace.rooms_dir,
        workspace.eval_dir,
        EvalOptions(
            per_piece=args.per_piece,
            distractor_images=args.distractor_images,
            seed=args.eval_seed,
        ),
    )
    print(format_report(report))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    from collective_ml import pipeline

    try:
        workspace = Workspace(args.workspace) if hasattr(args, "workspace") else None
        if args.command == "pull":
            assert workspace is not None
            pipeline.pull(ApiSettings.from_env(args.url, args.token), workspace)
        elif args.command == "fetch-data":
            assert workspace is not None
            pipeline.fetch_data(workspace, args.foregrounds, args.rooms)
        elif args.command == "build-detector":
            assert workspace is not None
            _build_detector(workspace, args)
        elif args.command == "build-embedder":
            assert workspace is not None
            _build_embedder(workspace, args)
        elif args.command == "evaluate":
            assert workspace is not None
            _evaluate(workspace, args)
        elif args.command == "release":
            assert workspace is not None
            _release(workspace, args)
        elif args.command == "publish":
            from collective_ml.release import publish_release

            settings = ReleaseSettings.from_env(args.url, args.release_token)
            payload = publish_release(settings, args.release_dir, activate=not args.no_activate)
            print(f"published {payload['version']} ({payload['id']}) to {settings.base_url}")
        elif args.command == "run":
            assert workspace is not None
            pipeline.fetch_data(workspace, args.foregrounds, args.rooms)
            args.skip_compose = False
            _build_detector(workspace, args)
            _build_embedder(workspace, args)
            args.publish, publish_later = False, args.publish
            _release(workspace, args)
            if workspace.raw_dir.is_dir() and any(workspace.raw_dir.iterdir()):
                args.release_dir = None
                _evaluate(workspace, args)
                # Re-assemble so the release carries the evaluation and its thresholds.
                args.publish = publish_later
                _release(workspace, args)
            elif publish_later:
                args.publish = True
                _release(workspace, args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
