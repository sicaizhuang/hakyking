from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from hakyking.audio.vocal_analysis import (
    PitchTrack,
    cache_source_pitch_track,
    cached_source_pitch_track,
)
from hakyking.models.audio_slice import AudioSlice
from hakyking.runtime import user_data_root


ANALYSIS_CACHE_DIR = user_data_root() / "cache" / "analysis"
ANALYSIS_CACHE_VERSION = 3
ANALYSIS_ALGORITHM_VERSION = "single-pass-long-fast-v2"


def load_cached_slices(
    path: str,
    cache_dir: str | Path = ANALYSIS_CACHE_DIR,
) -> list[AudioSlice] | None:
    identity = _source_identity(path)
    if identity is None:
        return None
    metadata_path, pitch_path = _cache_paths(identity["path"], cache_dir)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not _valid_payload(payload, identity):
        return None

    raw_slices = payload.get("slices")
    if not isinstance(raw_slices, list):
        return None
    try:
        slices = [
            AudioSlice.from_dict(item)
            for item in raw_slices
            if isinstance(item, dict)
        ]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not slices:
        return None

    _restore_pitch_track(identity["path"], pitch_path)
    return slices


def store_cached_slices(
    path: str,
    slices: list[AudioSlice],
    cache_dir: str | Path = ANALYSIS_CACHE_DIR,
) -> None:
    identity = _source_identity(path)
    if identity is None or not slices:
        return
    metadata_path, pitch_path = _cache_paths(identity["path"], cache_dir)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    pitch_track = cached_source_pitch_track(identity["path"])
    if pitch_track is not None and pitch_track.f0_hz.size:
        _write_pitch_track_atomic(pitch_path, pitch_track)

    payload = {
        "cache_version": ANALYSIS_CACHE_VERSION,
        "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
        "source": identity,
        "slices": [audio_slice.to_dict() for audio_slice in slices],
        "pitch_track": pitch_path.name if pitch_path.exists() else None,
    }
    _write_json_atomic(metadata_path, payload)


def invalidate_cached_slices(
    path: str,
    cache_dir: str | Path = ANALYSIS_CACHE_DIR,
) -> None:
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        resolved = str(path)
    metadata_path, pitch_path = _cache_paths(resolved, cache_dir)
    for candidate in (metadata_path, pitch_path):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def _source_identity(path: str) -> dict[str, Any] | None:
    try:
        source = Path(path).expanduser().resolve(strict=True)
        stat = source.stat()
    except (OSError, RuntimeError, ValueError):
        return None
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _cache_paths(path: str, cache_dir: str | Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(path.casefold().encode("utf-8")).hexdigest()
    root = Path(cache_dir)
    return root / f"{digest}.json", root / f"{digest}.pitch.npz"


def _valid_payload(payload: object, identity: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("cache_version") != ANALYSIS_CACHE_VERSION:
        return False
    if payload.get("algorithm_version") != ANALYSIS_ALGORITHM_VERSION:
        return False
    source = payload.get("source")
    if not isinstance(source, dict):
        return False
    return (
        str(source.get("path", "")).casefold() == str(identity["path"]).casefold()
        and source.get("size") == identity["size"]
        and source.get("mtime_ns") == identity["mtime_ns"]
    )


def _restore_pitch_track(path: str, pitch_path: Path) -> None:
    try:
        with np.load(pitch_path, allow_pickle=False) as payload:
            backend_value = payload["backend"]
            backend = str(np.asarray(backend_value).reshape(-1)[0])
            pitch_track = PitchTrack(
                times=np.asarray(payload["times"], dtype=np.float64),
                f0_hz=np.asarray(payload["f0_hz"], dtype=np.float32),
                confidence=np.asarray(payload["confidence"], dtype=np.float32),
                backend=backend,
            )
    except (OSError, KeyError, TypeError, ValueError):
        return
    if (
        pitch_track.times.shape == pitch_track.f0_hz.shape
        and pitch_track.f0_hz.shape == pitch_track.confidence.shape
    ):
        cache_source_pitch_track(path, pitch_track)


def _write_pitch_track_atomic(path: Path, pitch_track: PitchTrack) -> None:
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                times=np.asarray(pitch_track.times, dtype=np.float64),
                f0_hz=np.asarray(pitch_track.f0_hz, dtype=np.float32),
                confidence=np.asarray(pitch_track.confidence, dtype=np.float32),
                backend=np.asarray([pitch_track.backend]),
            )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _temporary_path(path: Path) -> Path:
    return path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
