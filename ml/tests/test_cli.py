import pytest

from collective_ml.cli import build_parser, main


def test_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["run", "--epochs", "3", "--samples-per-piece", "20", "--no-publish"])
    assert (
        args.command == "run"
        and args.epochs == 3
        and args.samples_per_piece == 20
        and args.no_publish
    )


def test_missing_config_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv("COLLECTIVE_URL", raising=False)
    monkeypatch.delenv("COLLECTIVE_TOKEN", raising=False)
    assert main(["pull"]) == 2
    assert "COLLECTIVE_URL" in capsys.readouterr().err
