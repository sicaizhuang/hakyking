from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hakyking.audio.gain import MAX_GAIN_DB, MIN_GAIN_DB
from hakyking.models.audio_slice import AudioSlice
from hakyking.models.project import ProjectModel, TrackModel, TrackRole, TrackType
from hakyking.views.workspace import AudioSliceGraphicsItem, WorkspaceView


PROJECT_FORMAT = "hakyking.project"
PROJECT_VERSION = 4


@dataclass(frozen=True)
class RestoredSlice:
    audio_slice: AudioSlice
    track_index: int
    x: float
    y: float
    width: float
    height: float
    target_midi_note: int | None
    target_duration: float
    n_steps: float
    rate: float
    gain_db: float
    pitch_flatten_amount: float
    formant_shift: float
    protect_transients: bool
    pitch_control_points: list[dict[str, float]]
    pitch_vibrato_regions: list[dict[str, float | str]]
    pitch_shape_regions: list[dict[str, float | str]]
    track_reference: bool
    reference_editable: bool
    missing_source: bool


@dataclass(frozen=True)
class LoadedProject:
    project: ProjectModel
    slices: list[RestoredSlice]
    missing_paths: list[str]
    recovered_from: str | None = None
    migrated_from_version: int | None = None


class ProjectManager:
    """Serializes Hakyking project metadata without touching source media files."""

    def save(self, path: str | Path, project: ProjectModel, workspace: WorkspaceView) -> Path:
        output_path = self.with_haky_suffix(path)
        payload = self.build_project_dict(project, workspace)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(output_path, payload)
        return output_path

    def load(self, path: str | Path) -> LoadedProject:
        input_path = Path(path)
        try:
            payload = self._read_payload(input_path)
            return self.project_from_dict(payload)
        except ValueError as primary_error:
            backup_path = self.backup_path(input_path)
            if not backup_path.is_file():
                raise
            try:
                recovered = self.project_from_dict(self._read_payload(backup_path))
            except ValueError:
                raise primary_error
            return LoadedProject(
                project=recovered.project,
                slices=recovered.slices,
                missing_paths=recovered.missing_paths,
                recovered_from=str(backup_path),
                migrated_from_version=recovered.migrated_from_version,
            )

    @staticmethod
    def backup_path(path: str | Path) -> Path:
        return Path(f"{Path(path)}.bak")

    def _write_json_atomic(self, output_path: Path, payload: dict[str, Any]) -> None:
        temporary_path = Path(f"{output_path}.tmp")
        backup_path = self.backup_path(output_path)
        backup_temporary_path = Path(f"{backup_path}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            if output_path.is_file():
                shutil.copy2(output_path, backup_temporary_path)
                os.replace(backup_temporary_path, backup_path)
            os.replace(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            backup_temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_payload(input_path: Path) -> dict[str, Any]:
        try:
            with input_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid .haky project JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise ValueError(f"Cannot read .haky project: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Invalid .haky project: root must be a JSON object.")
        if payload.get("format") not in (None, PROJECT_FORMAT):
            raise ValueError("Invalid .haky project: unsupported format.")
        return payload

    def build_project_dict(
        self,
        project: ProjectModel,
        workspace: WorkspaceView,
    ) -> dict[str, Any]:
        return {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "project": {
                "title": project.title,
                "bpm": project.bpm,
                "sample_rate": project.sample_rate,
                "selected_track_index": project.selected_track_index,
                "material_folders": list(dict.fromkeys(project.material_folders)),
            },
            "tracks": [
                self._track_to_dict(index, track)
                for index, track in enumerate(project.tracks)
            ],
            "slices": [
                self._slice_item_to_dict(item)
                for item in sorted(
                    workspace.slice_items(),
                    key=lambda slice_item: (
                        slice_item.track_index,
                        slice_item.scenePos().x(),
                        slice_item.scenePos().y(),
                    ),
                )
            ],
            "material_slice_overrides": {
                path: [audio_slice.to_dict() for audio_slice in slices]
                for path, slices in sorted(project.material_slice_overrides.items())
                if path and slices
            },
        }

    def project_from_dict(self, payload: dict[str, Any]) -> LoadedProject:
        raw_version = payload.get("version", 1)
        try:
            source_version = int(raw_version)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid .haky project: version must be an integer.") from exc
        if source_version < 1:
            raise ValueError("Invalid .haky project: version must be positive.")
        if source_version > PROJECT_VERSION:
            raise ValueError(
                f"Unsupported .haky project version {source_version}; "
                f"this build supports up to version {PROJECT_VERSION}."
            )
        project_payload = self._dict_value(payload.get("project"))
        project = ProjectModel(
            title=str(project_payload.get("title", "Untitled Hakyking Project")),
            bpm=self._float_value(project_payload.get("bpm"), 120.0, minimum=20.0, maximum=400.0),
            sample_rate=self._int_value(
                project_payload.get("sample_rate"),
                44100,
                minimum=8000,
                maximum=192000,
            ),
            selected_track_index=self._optional_int(
                project_payload.get("selected_track_index")
            ),
            material_folders=self._string_list(project_payload.get("material_folders", [])),
        )

        track_payloads = payload.get("tracks", [])
        if not isinstance(track_payloads, list):
            raise ValueError("Invalid .haky project: tracks must be a list.")
        project.tracks = [
            self._track_from_dict(self._dict_value(track_payload), index)
            for index, track_payload in enumerate(track_payloads)
        ]
        if not project.tracks:
            project.bootstrap_default_tracks()

        if (
            project.selected_track_index is None
            or project.selected_track_index < 0
            or project.selected_track_index >= len(project.tracks)
        ):
            project.selected_track_index = min(1, max(0, len(project.tracks) - 1))

        project.material_slice_overrides = self._material_slice_overrides_from_dict(
            payload.get("material_slice_overrides", {})
        )

        slice_payloads = payload.get("slices", [])
        if not isinstance(slice_payloads, list):
            raise ValueError("Invalid .haky project: slices must be a list.")

        restored_slices: list[RestoredSlice] = []
        missing_paths: list[str] = []
        for index, slice_payload in enumerate(slice_payloads):
            restored_slice = self._restored_slice_from_dict(
                self._dict_value(slice_payload),
                fallback_index=index,
                track_count=len(project.tracks),
            )
            if restored_slice.missing_source:
                missing_paths.append(restored_slice.audio_slice.source_path)
            restored_slices.append(restored_slice)

        return LoadedProject(
            project=project,
            slices=restored_slices,
            missing_paths=sorted(set(missing_paths)),
            migrated_from_version=(
                source_version if source_version < PROJECT_VERSION else None
            ),
        )

    @staticmethod
    def _material_slice_overrides_from_dict(value: Any) -> dict[str, list[AudioSlice]]:
        if not isinstance(value, dict):
            return {}
        overrides: dict[str, list[AudioSlice]] = {}
        for source_path, raw_slices in value.items():
            if not isinstance(source_path, str) or not source_path.strip():
                continue
            if not isinstance(raw_slices, list):
                continue
            restored: list[AudioSlice] = []
            for index, raw_slice in enumerate(raw_slices):
                if not isinstance(raw_slice, dict):
                    continue
                try:
                    audio_slice = AudioSlice.from_dict(raw_slice)
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if (
                    audio_slice.source_path != source_path
                    or not math.isfinite(audio_slice.start_time)
                    or not math.isfinite(audio_slice.end_time)
                    or audio_slice.end_time <= audio_slice.start_time
                ):
                    continue
                start_time = max(0.0, audio_slice.start_time)
                end_time = max(start_time, audio_slice.end_time)
                if end_time <= start_time:
                    continue
                restored.append(
                    AudioSlice(
                        source_path=source_path,
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        midi_note=audio_slice.midi_note,
                        f0_hz=audio_slice.f0_hz,
                        pitch_confidence=audio_slice.pitch_confidence,
                        analysis_backend=audio_slice.analysis_backend,
                    )
                )
            if restored:
                overrides[source_path] = restored
        return overrides

    @staticmethod
    def with_haky_suffix(path: str | Path) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".haky":
            output_path = output_path.with_suffix(".haky")
        return output_path

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                continue
            normalized = str(Path(entry).expanduser())
            if normalized not in result:
                result.append(normalized)
        return result

    def _track_to_dict(self, index: int, track: TrackModel) -> dict[str, Any]:
        return {
            "track_index": index,
            "name": track.name,
            "role": track.role.value,
            "track_type": track.track_type.value,
            "is_frozen": track.locked,
            "is_locked": track.locked,
            "is_muted": track.muted,
            "is_solo": track.solo,
            "clip_path": track.clip_path,
            "clip_start": track.clip_start,
            "clip_duration": track.clip_duration,
            "clip_editable": track.clip_editable,
        }

    def _slice_item_to_dict(self, item: AudioSliceGraphicsItem) -> dict[str, Any]:
        parameters = item.current_render_parameters()
        return {
            "file_path": item.audio_slice.source_path,
            "original_start": item.audio_slice.start_time,
            "original_end": item.audio_slice.end_time,
            "slice_index": item.audio_slice.index,
            "midi_note": item.audio_slice.midi_note,
            "f0_hz": item.audio_slice.f0_hz,
            "pitch_confidence": item.audio_slice.pitch_confidence,
            "analysis_backend": item.audio_slice.analysis_backend,
            "track_index": item.track_index,
            "x": item.scenePos().x(),
            "y": item.scenePos().y(),
            "width": item.rect().width(),
            "height": item.rect().height(),
            "target_midi_note": item.target_midi_note,
            "target_duration": item.target_duration,
            "n_steps": parameters.n_steps,
            "rate": parameters.rate,
            "gain_db": parameters.gain_db,
            "pitch_flatten_amount": parameters.pitch_flatten_amount,
            "formant_shift": parameters.formant_shift,
            "protect_transients": parameters.protect_transients,
            "pitch_control_points": item.pitch_control_points_payload(),
            "pitch_vibrato_regions": item.pitch_vibrato_regions_payload(),
            "pitch_shape_regions": item.pitch_shape_regions_payload(),
            "edit_model": item.edit_model.to_project_payload(),
            "is_track_reference": item.is_track_reference,
            "reference_editable": item.reference_editable,
        }

    def _track_from_dict(self, payload: dict[str, Any], fallback_index: int) -> TrackModel:
        track_type = self._enum_value(
            TrackType,
            payload.get("track_type"),
            TrackType.VOCAL_SLICE,
        )
        role = self._enum_value(
            TrackRole,
            payload.get("role"),
            TrackRole.MAIN if track_type == TrackType.MASTER_BGM else TrackRole.AUX,
        )
        locked = bool(payload.get("is_frozen", payload.get("is_locked", False)))
        if track_type == TrackType.MASTER_BGM:
            locked = True
        return TrackModel(
            name=str(payload.get("name", f"Track {fallback_index + 1}")),
            role=role,
            track_type=track_type,
            muted=bool(payload.get("is_muted", False)),
            solo=bool(payload.get("is_solo", False)),
            locked=locked,
            clip_path=str(payload.get("clip_path", "") or ""),
            clip_start=self._float_value(payload.get("clip_start"), 0.0, minimum=0.0),
            clip_duration=self._float_value(payload.get("clip_duration"), 0.0, minimum=0.0),
            clip_editable=bool(payload.get("clip_editable", False)),
        )

    def _restored_slice_from_dict(
        self,
        payload: dict[str, Any],
        fallback_index: int,
        track_count: int = 0,
    ) -> RestoredSlice:
        edit_model_payload = self._dict_value(payload.get("edit_model"))
        clip_payload = self._dict_value(edit_model_payload.get("clip"))
        automation_payload = self._dict_value(edit_model_payload.get("pitch_automation"))
        effects_payload = self._dict_value(edit_model_payload.get("effects"))
        source_path = str(
            clip_payload.get(
                "source_path",
                payload.get("file_path", payload.get("source_path", "")),
            )
        ).strip()
        original_start = self._float_value(
            clip_payload.get(
                "source_start",
                payload.get("original_start", payload.get("start_time")),
            ),
            0.0,
            minimum=0.0,
        )
        original_end = self._float_value(
            clip_payload.get(
                "source_end",
                payload.get("original_end", payload.get("end_time")),
            ),
            original_start + 0.001,
            minimum=original_start,
        )
        audio_slice = AudioSlice(
            source_path=source_path,
            index=self._int_value(
                payload.get("slice_index", payload.get("index")),
                fallback_index,
                minimum=0,
            ),
            start_time=original_start,
            end_time=max(original_start, original_end),
            midi_note=self._optional_int(payload.get("midi_note")),
            f0_hz=self._optional_float(payload.get("f0_hz")),
            pitch_confidence=self._optional_float(payload.get("pitch_confidence")),
            analysis_backend=(
                None
                if payload.get("analysis_backend") is None
                else str(payload.get("analysis_backend"))
            ),
        )
        target_duration = self._float_value(
            clip_payload.get("target_duration", payload.get("target_duration")),
            max(0.001, audio_slice.duration),
            minimum=0.001,
        )
        track_index = self._int_value(
            clip_payload.get("track_index", payload.get("track_index")),
            0,
            minimum=0,
        )
        if track_count > 0 and track_index >= track_count:
            track_index = min(1, max(0, track_count - 1))
        pitch_flatten_amount = self._float_value(
            effects_payload.get(
                "pitch_flatten_amount",
                payload.get("pitch_flatten_amount"),
            ),
            0.0,
            minimum=0.0,
            maximum=1.0,
        )
        formant_shift = self._float_value(
            effects_payload.get("formant_shift", payload.get("formant_shift")),
            0.0,
            minimum=-12.0,
            maximum=12.0,
        )
        return RestoredSlice(
            audio_slice=audio_slice,
            track_index=track_index,
            x=self._float_value(payload.get("x"), 0.0, minimum=0.0),
            y=self._float_value(payload.get("y"), 0.0, minimum=0.0),
            width=self._float_value(
                payload.get("width"),
                target_duration * 260.0,
                minimum=1.0,
            ),
            height=self._float_value(payload.get("height"), 30.0, minimum=1.0),
            target_midi_note=self._optional_int(
                clip_payload.get("pitch_center_midi", payload.get("target_midi_note"))
            ),
            target_duration=max(0.001, target_duration),
            n_steps=self._float_value(payload.get("n_steps"), 0.0),
            rate=self._float_value(payload.get("rate"), 1.0, minimum=0.05, maximum=20.0),
            gain_db=self._float_value(
                effects_payload.get("gain_db", payload.get("gain_db")),
                0.0,
                minimum=MIN_GAIN_DB,
                maximum=MAX_GAIN_DB,
            ),
            pitch_flatten_amount=pitch_flatten_amount,
            formant_shift=formant_shift,
            protect_transients=bool(
                effects_payload.get(
                    "protect_transients",
                    payload.get("protect_transients", True),
                )
            ),
            pitch_control_points=self._pitch_control_points_from_value(
                automation_payload.get(
                    "control_points",
                    payload.get("pitch_control_points", []),
                )
            ),
            pitch_vibrato_regions=self._pitch_vibrato_regions_from_value(
                automation_payload.get(
                    "vibrato_regions",
                    payload.get("pitch_vibrato_regions", []),
                )
            ),
            pitch_shape_regions=self._pitch_shape_regions_from_value(
                payload.get("pitch_shape_regions", [])
            ),
            track_reference=bool(payload.get("is_track_reference", False)),
            reference_editable=bool(payload.get("reference_editable", False)),
            missing_source=not self._path_exists(source_path),
        )

    @staticmethod
    def _dict_value(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return value

    @staticmethod
    def _pitch_control_points_from_value(value: Any) -> list[dict[str, float]]:
        if not isinstance(value, list):
            return []
        points: list[dict[str, float]] = []
        for entry in value:
            try:
                if isinstance(entry, dict):
                    raw_x = entry.get("x", entry.get("ratio", 0.0))
                    raw_offset = entry.get("offset", entry.get("semitones", 0.0))
                else:
                    raw_x = entry[0]
                    raw_offset = entry[1]
                x_value = max(0.0, min(1.0, float(raw_x)))
                offset = max(-24.0, min(24.0, float(raw_offset)))
            except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                continue
            points.append({"x": x_value, "offset": offset})
        points.sort(key=lambda point: point["x"])
        return points

    @staticmethod
    def _pitch_vibrato_regions_from_value(value: Any) -> list[dict[str, float | str]]:
        if not isinstance(value, list):
            return []
        regions: list[dict[str, float | str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            try:
                start = max(0.0, min(1.0, float(entry.get("start", 0.0))))
                end = max(0.0, min(1.0, float(entry.get("end", 1.0))))
                if end < start:
                    start, end = end, start
                if end - start <= 1e-5:
                    continue
                cycles = max(0.0, float(entry.get("cycles", 0.0)))
                depth = max(0.0, min(12.0, float(entry.get("depth", 0.0))))
                phase = float(entry.get("phase", 0.0)) % 1.0
            except (TypeError, ValueError, OverflowError):
                continue
            waveform = str(entry.get("waveform", "sine"))
            if waveform not in {"sine", "triangle", "square"}:
                waveform = "sine"
            if cycles <= 0.0 or depth <= 0.0:
                continue
            regions.append(
                {
                    "start": start,
                    "end": end,
                    "cycles": cycles,
                    "depth": depth,
                    "phase": phase,
                    "waveform": waveform,
                }
            )
        regions.sort(key=lambda region: (float(region["start"]), float(region["end"])))
        return regions

    @staticmethod
    def _pitch_shape_regions_from_value(value: Any) -> list[dict[str, float | str]]:
        if not isinstance(value, list):
            return []
        valid_shapes = {
            "linear",
            "smooth",
            "ease_in",
            "ease_out",
            "s_curve",
            "instant",
        }
        regions: list[dict[str, float | str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            try:
                start = max(0.0, min(1.0, float(entry.get("start", 0.0))))
                end = max(0.0, min(1.0, float(entry.get("end", 1.0))))
                if end < start:
                    start, end = end, start
            except (TypeError, ValueError, OverflowError):
                continue
            if end - start <= 1e-5:
                continue
            shape = str(entry.get("shape", "linear"))
            if shape not in valid_shapes:
                shape = "linear"
            regions.append(
                {
                    "start": round(start, 6),
                    "end": round(end, 6),
                    "shape": shape,
                }
            )
        regions.sort(key=lambda region: (float(region["start"]), float(region["end"])))
        return regions

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _int_value(
        value: Any,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            number = default
        if minimum is not None:
            number = max(minimum, number)
        if maximum is not None:
            number = min(maximum, number)
        return number

    @staticmethod
    def _float_value(
        value: Any,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = default
        if not math.isfinite(number):
            number = default
        if minimum is not None:
            number = max(minimum, number)
        if maximum is not None:
            number = min(maximum, number)
        return number

    @staticmethod
    def _path_exists(path: str) -> bool:
        if not path:
            return False
        try:
            return Path(path).exists()
        except OSError:
            return False

    @staticmethod
    def _enum_value(enum_type, value: Any, default):
        try:
            return enum_type(value)
        except Exception:
            return default
