from __future__ import annotations

import json

from hakyking.models.audio_slice import AudioSlice


MIME_AUDIO_SLICES = "application/x-hakyking-audio-slices"
MIME_AUDIO_FILE = "application/x-hakyking-audio-file"


def encode_audio_slices(slices: list[AudioSlice]) -> bytes:
    payload = [audio_slice.to_dict() for audio_slice in slices]
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def decode_audio_slices(data: bytes) -> list[AudioSlice]:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Audio slice MIME payload must be a list.")
    return [AudioSlice.from_dict(item) for item in payload]


def encode_audio_file(path: str) -> bytes:
    payload = {"path": str(path)}
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def decode_audio_file(data: bytes) -> str:
    payload = json.loads(data.decode("utf-8"))
    if not isinstance(payload, dict) or not payload.get("path"):
        raise ValueError("Audio file MIME payload must contain a path.")
    return str(payload["path"])
