"""HTTP client for the collector-facing ML endpoints (collection listing and photos)."""

from __future__ import annotations

import json
import logging
import mimetypes
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from collective_ml.config import ApiSettings

log = logging.getLogger(__name__)

CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}


@dataclass(frozen=True)
class ImageRef:
    """One stored photo of an artwork."""

    id: str
    url: str
    width: int
    height: int
    content_type: str

    @property
    def extension(self) -> str:
        ext = CONTENT_TYPE_EXTENSIONS.get(self.content_type)
        if ext:
            return ext
        guessed = mimetypes.guess_extension(self.content_type) or ".bin"
        return guessed.lstrip(".")


@dataclass(frozen=True)
class ArtworkRef:
    """An artwork and its photos as returned by `GET /api/ml/collection`."""

    id: str
    title: str
    artist: str | None
    year: int | None
    status: str
    images: tuple[ImageRef, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Collection:
    """A collector's full inventory."""

    user_id: str
    origin: str
    artworks: tuple[ArtworkRef, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Collection:
        """Parse the JSON body of the collection endpoint."""
        artworks = tuple(
            ArtworkRef(
                id=str(item["id"]),
                title=str(item.get("title") or ""),
                artist=item.get("artist"),
                year=item.get("year"),
                status=str(item.get("status") or "owned"),
                images=tuple(
                    ImageRef(
                        id=str(image["id"]),
                        url=str(image["url"]),
                        width=int(image.get("width") or 0),
                        height=int(image.get("height") or 0),
                        content_type=str(image.get("contentType") or "application/octet-stream"),
                    )
                    for image in item.get("images", [])
                ),
            )
            for item in payload.get("artworks", [])
        )
        return cls(
            user_id=str(payload["userId"]),
            origin=str(payload.get("origin") or ""),
            artworks=artworks,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialise back to the API shape so it can be cached on disk."""
        return {
            "userId": self.user_id,
            "origin": self.origin,
            "artworks": [
                {
                    "id": artwork.id,
                    "title": artwork.title,
                    "artist": artwork.artist,
                    "year": artwork.year,
                    "status": artwork.status,
                    "images": [
                        {
                            "id": image.id,
                            "url": image.url,
                            "width": image.width,
                            "height": image.height,
                            "contentType": image.content_type,
                        }
                        for image in artwork.images
                    ],
                }
                for artwork in self.artworks
            ],
        }

    def with_images(self) -> tuple[ArtworkRef, ...]:
        """Artworks that have at least one photo, in a stable order."""
        return tuple(sorted((a for a in self.artworks if a.images), key=lambda a: a.id))


class CollectiveClient:
    """Thin wrapper over the collection endpoints. Pass `transport` to stub it in tests."""

    def __init__(self, settings: ApiSettings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.token}", "User-Agent": "collective-ml"},
            timeout=httpx.Timeout(60.0, read=300.0),
            transport=transport,
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CollectiveClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_collection(self) -> Collection:
        """Download the inventory listing."""
        response = self._client.get("/api/ml/collection")
        _raise_for_status(response)
        return Collection.from_payload(response.json())

    def stream_image(self, image: ImageRef) -> Iterator[bytes]:
        """Yield the raw bytes of one photo."""
        with self._client.stream("GET", image.url) as response:
            _raise_for_status(response)
            yield from response.iter_bytes()

    def download_image(self, image: ImageRef, destination: Path) -> bool:
        """Save a photo to disk. Returns False when the file already existed."""
        if destination.exists() and destination.stat().st_size > 0:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in self.stream_image(image):
                handle.write(chunk)
        tmp.replace(destination)
        return True


class ApiError(RuntimeError):
    """The server answered with an error status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    detail = ""
    try:
        if not response.is_stream_consumed:
            response.read()
        body = response.json()
        detail = str(body.get("message") or body.get("error") or body)
    except (ValueError, httpx.ResponseNotRead):
        detail = response.text[:300] if hasattr(response, "text") else ""
    if response.status_code == 401:
        detail = detail or "Invalid or missing API token."
    raise ApiError(response.status_code, detail)


def pull_collection(
    client: CollectiveClient, raw_dir: Path, collection_json: Path
) -> tuple[Collection, int, int]:
    """
    Download the collection listing and every photo into `raw/<artworkId>/<imageId>.<ext>`.

    Returns the collection plus (downloaded, skipped) counts.
    """
    collection = client.fetch_collection()
    collection_json.parent.mkdir(parents=True, exist_ok=True)
    collection_json.write_text(json.dumps(collection.to_payload(), indent=2))

    downloaded = skipped = 0
    for artwork in collection.artworks:
        if not artwork.images:
            log.warning("Skipping %r (%s): no photos.", artwork.title, artwork.id)
            continue
        for image in artwork.images:
            destination = raw_dir / artwork.id / f"{image.id}.{image.extension}"
            if client.download_image(image, destination):
                downloaded += 1
            else:
                skipped += 1
    return collection, downloaded, skipped
