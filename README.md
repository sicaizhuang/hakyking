# Hakyking

## Overview / 概览

Hakyking is an experimental desktop audio slicing and pitch-editing host for
Windows. It is aimed at vocal editing, syllable-oriented arrangement, pitch
automation, and fast playback experiments.

Hakyking 是一个面向 Windows 的实验性桌面音频切片与音高编辑工具，
主要用于人声编辑、音节式编排、音高自动化和快速试听实验。

The project is released as a snapshot for learning and community improvement.
It is not a Melodyne replacement, and audio quality, platform support, and UI
stability still need work.

项目以学习快照的形式公开，欢迎社区继续改进。它并不是 Melodyne 的
替代品，音频质量、平台支持和界面稳定性仍有待完善。

## What is included / 包含内容

- PyQt5 desktop interface with material browser, piano roll, timeline, tracks,
  slicing, playback, and project save/load.
- Non-destructive audio edit models for clips, pitch automation, control
  points, and parameterized vibrato regions.
- Optional pitch/time engines using Praat PSOLA, Rubber Band, PyWorld, and
  Librosa-backed analysis paths.
- Unit tests and deterministic synthetic audio quality checks.
- Windows PyInstaller build configuration.

Personal audio, downloaded pitch models, FFmpeg, Rubber Band binaries, build
outputs, logs, caches, and local project files are intentionally not included.

## Requirements / 运行要求

- Python 3.11 or newer
- Windows is the primary supported platform
- FFmpeg and FFprobe on `PATH` for video audio extraction
- Rubber Band on `PATH` for Rubber Band pitch/time processing

Install Python dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the application:

```powershell
python -m hakyking.main
```

On Windows, `run_hakyking.bat` is also available. The application stores
settings, cache, and autosave data in the per-user Hakyking data directory.
Set `HAKYKING_DATA_DIR` to override that location.

## Tests / 测试

The public test suite does not require private audio. The three acceptance
sample slots are documented in `docs/ACCEPTANCE_SAMPLES.md`; place your own
licensed fixtures under `tests/fixtures/acceptance/` if you want to run that
audio-specific gate.

```powershell
python -m compileall hakyking dev_tools
python -m unittest discover -s tests -v
python dev_tools\audio_quality_audit.py
```

The optional acceptance runner reports missing local fixtures as `SKIP`.
GitHub CI runs the portable public smoke checks; the full local suite also
includes model-contract tests that are best run in the project's supported
Windows environment.

## Build a Windows executable / 打包 Windows 可执行文件

Install PyInstaller in the active environment, then run:

```powershell
powershell -ExecutionPolicy Bypass -File dev_tools\build_windows_exe.ps1 -Clean
```

The build is an `onedir` distribution. External audio tools are not bundled;
keep their licenses and installation separate from this repository.

## Project map / 项目结构

- `hakyking/audio/`: reading, analysis, playback, DSP, and export
- `hakyking/models/`: project, clip, scale, and non-destructive edit models
- `hakyking/controllers/`: worker orchestration and UI coordination
- `hakyking/views/`: Qt widgets and interaction views
- `tests/`: unit and contract tests
- `docs/`: architecture and feature-freeze notes

The current edit-model boundary is described in `docs/EDIT_MODEL.md`. The
project is deliberately frozen around the existing V/B/N pitch-editing tools
while reliability and maintainability catch up.

## Contributing / 贡献

Bug reports with a short reproduction, environment details, and a traceback
are especially useful. See `CONTRIBUTING.md` before opening a pull request.

## License / 许可

Hakyking is available under the MIT License. See `LICENSE`.
