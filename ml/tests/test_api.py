import json
from pathlib import Path

import httpx
import pytest

from collective_ml.api import ApiError, Collection, CollectiveClient, pull_collection
from collective_ml.config import ApiSettings

SETTINGS = ApiSettings(base_url="https://collective.test", token="clv_secret")

COLLECTION = {
    "userId": "u1",
    "origin": "https://collective.test",
    "artworks": [
        {
            "id": "art-b",
            "title": "B",
            "artist": None,
            "year": 2001,
            "status": "owned",
            "images": [
                {
                    "id": "img1",
                    "url": "https://collective.test/api/ml/media/users/u1/artworks/art-b/img1.jpg",
                    "width": 10,
                    "height": 10,
                    "contentType": "image/jpeg",
                }
            ],
        },
        {"id": "art-a", "title": "A", "artist": "X", "year": None, "status": "owned", "images": []},
    ],
}


def make_client(handler):
    return CollectiveClient(SETTINGS, transport=httpx.MockTransport(handler))


def test_fetch_collection_sends_bearer_and_parses():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["path"] = request.url.path
        return httpx.Response(200, json=COLLECTION)

    with make_client(handler) as client:
        collection = client.fetch_collection()

    assert seen == {"auth": "Bearer clv_secret", "path": "/api/ml/collection"}
    assert collection.user_id == "u1"
    assert [a.id for a in collection.with_images()] == ["art-b"]
    assert collection.artworks[0].images[0].extension == "jpg"
    assert Collection.from_payload(collection.to_payload()) == collection


def test_pull_downloads_once_and_skips_existing(tmp_path: Path):
    calls = {"media": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ml/collection":
            return httpx.Response(200, json=COLLECTION)
        calls["media"] += 1
        return httpx.Response(200, content=b"jpegbytes")

    raw = tmp_path / "raw"
    with make_client(handler) as client:
        _, downloaded, skipped = pull_collection(client, raw, tmp_path / "collection.json")
        assert (downloaded, skipped) == (1, 0)
        _, downloaded, skipped = pull_collection(client, raw, tmp_path / "collection.json")
        assert (downloaded, skipped) == (0, 1)

    assert calls["media"] == 1
    assert (raw / "art-b" / "img1.jpg").read_bytes() == b"jpegbytes"
    assert json.loads((tmp_path / "collection.json").read_text())["userId"] == "u1"


def test_publish_sends_multipart(tmp_path: Path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["content_type"] = request.headers["content-type"]
        captured["body"] = request.read()
        return httpx.Response(
            201,
            json={
                "id": "m1",
                "url": "https://collective.test/media/x",
                "sha256": "abc",
                "inputSize": 320,
                "labels": {"1": "art-b"},
            },
        )

    with make_client(handler) as client:
        published = client.publish_model(onnx, {"1": "art-b"}, 320, {"map50": 0.9})

    assert captured["content_type"].startswith("multipart/form-data")
    body = captured["body"]
    assert b'name="model"; filename="model.onnx"' in body
    assert b'name="labels"' in body and b'{"1": "art-b"}' in body
    assert b'name="inputSize"' in body and b"320" in body
    assert b'name="metrics"' in body
    assert published.id == "m1" and published.labels == {"1": "art-b"}


def test_errors_surface_server_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Invalid or missing API token."})

    with make_client(handler) as client, pytest.raises(ApiError) as excinfo:
        client.fetch_collection()
    assert excinfo.value.status == 401
    assert "token" in str(excinfo.value)
