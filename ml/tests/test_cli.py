import pytest

from collective_ml.cli import build_parser, main


def test_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["run", "--epochs", "3", "--count", "20", "--publish"])
    assert args.command == "run" and args.epochs == 3 and args.count == 20 and args.publish
    release = parser.parse_args(["release", "--release-version", "2026.11.02-1", "--publish"])
    assert release.release_version == "2026.11.02-1" and release.publish
    evaluate = parser.parse_args(["evaluate", "--per-piece", "5"])
    assert evaluate.per_piece == 5


def test_missing_config_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("COLLECTIVE_URL", raising=False)
    monkeypatch.delenv("COLLECTIVE_TOKEN", raising=False)
    monkeypatch.delenv("COLLECTIVE_RELEASE_TOKEN", raising=False)
    assert main(["pull"]) == 2
    assert "COLLECTIVE_TOKEN" in capsys.readouterr().err
    assert main(["publish", "somewhere", "--url", "https://x.test"]) == 2
    assert "COLLECTIVE_RELEASE_TOKEN" in capsys.readouterr().err
