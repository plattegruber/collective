"""`collective-ml` command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from collective_ml import __version__
from collective_ml.config import ApiSettings, ConfigError, Workspace, default_workspace_root


def _add_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url", help="App origin, e.g. https://collective.example.com (or $COLLECTIVE_URL)"
    )
    parser.add_argument(
        "--token", help="API token from Settings → API tokens (or $COLLECTIVE_TOKEN)"
    )


def _add_workspace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace_root(),
        help="Workspace root (default: ml/workspace)",
    )
    parser.add_argument("--user", help="User id subfolder; inferred when only one exists")


def _add_synth_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img-size", type=int, default=640, help="Synthetic canvas size")
    parser.add_argument("--samples-per-piece", type=int, default=160)
    parser.add_argument("--backgrounds-dir", type=Path, help="Optional folder of wall photos")
    parser.add_argument("--seed", type=int, default=1337)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--train-size", type=int, default=320, help="Detector input size (square)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", help="cuda, mps, or cpu (auto by default)")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree."""
    parser = argparse.ArgumentParser(
        prog="collective-ml", description="Train and publish artwork recognizers."
    )
    parser.add_argument("--version", action="version", version=f"collective-ml {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="Download the collection and its photos")
    _add_api_args(p_pull)
    p_pull.add_argument("--workspace", type=Path, default=default_workspace_root())

    p_synth = sub.add_parser("synth", help="Build the synthetic training dataset")
    _add_workspace_args(p_synth)
    _add_synth_args(p_synth)

    p_train = sub.add_parser("train", help="Train the detector")
    _add_workspace_args(p_train)
    _add_train_args(p_train)
    p_train.add_argument("--seed", type=int, default=1337)

    p_export = sub.add_parser("export", help="Export the best checkpoint to ONNX and verify it")
    _add_workspace_args(p_export)

    p_publish = sub.add_parser("publish", help="Upload the ONNX model to the app")
    _add_api_args(p_publish)
    _add_workspace_args(p_publish)

    p_run = sub.add_parser("run", help="pull → synth → train → export → publish")
    _add_api_args(p_run)
    p_run.add_argument("--workspace", type=Path, default=default_workspace_root())
    _add_synth_args(p_run)
    _add_train_args(p_run)
    p_run.add_argument("--no-publish", action="store_true", help="Stop after export")

    return parser


def _synth(workspace: Workspace, args: argparse.Namespace) -> None:
    from collective_ml.synth import SynthOptions, synthesize

    workspace.ensure()
    options = SynthOptions(
        img_size=args.img_size,
        samples_per_piece=args.samples_per_piece,
        seed=args.seed,
        backgrounds_dir=args.backgrounds_dir,
    )
    synthesize(workspace.raw_dir, workspace.dataset_dir, workspace.labels_json, options)


def _train(workspace: Workspace, args: argparse.Namespace) -> None:
    from collective_ml.train import TrainOptions, train

    options = TrainOptions(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.train_size,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        amp=args.amp,
    )
    result = train(workspace.dataset_dir, workspace.model_dir, options)
    print(
        f"best epoch {result.best_epoch}, val loss {result.best_val_loss:.3f}, "
        f"mAP@50 {result.map50:.3f}"
    )


def _export(workspace: Workspace) -> None:
    from collective_ml.export import export_onnx

    if not workspace.checkpoint.exists():
        raise SystemExit(f"{workspace.checkpoint} not found. Run `collective-ml train` first.")
    result = export_onnx(workspace.checkpoint, workspace.onnx_path)
    print(f"exported {result.onnx_path} (input {result.input_size}px)")


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
        if args.command == "pull":
            pipeline.pull(ApiSettings.from_env(args.url, args.token), args.workspace)
        elif args.command == "synth":
            _synth(pipeline.resolve_workspace(args.workspace, args.user), args)
        elif args.command == "train":
            _train(pipeline.resolve_workspace(args.workspace, args.user), args)
        elif args.command == "export":
            _export(pipeline.resolve_workspace(args.workspace, args.user))
        elif args.command == "publish":
            pipeline.publish(
                ApiSettings.from_env(args.url, args.token),
                pipeline.resolve_workspace(args.workspace, args.user),
            )
        elif args.command == "run":
            settings = ApiSettings.from_env(args.url, args.token)
            workspace = pipeline.pull(settings, args.workspace)
            _synth(workspace, args)
            _train(workspace, args)
            _export(workspace)
            if not args.no_publish:
                pipeline.publish(settings, workspace)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
