# Contributing to Hakyking

Hakyking is an experimental audio application. Small, reproducible changes
are easier to review than broad UI rewrites.

## Before opening a pull request

- Keep personal audio, model files, caches, logs, build outputs, and absolute
  machine paths out of commits.
- Add or update a focused test for behavior that can be tested without an
  audio device.
- Run compileall, the unit suite, and the synthetic audio quality audit.
- Include the operating system, Python version, dependency versions, and a
  traceback for runtime bugs.

## Audio and algorithm changes

Explain the input assumptions, expected duration/pitch behavior, fallback
engine, and any external command-line dependency. Do not add a binary model
or media file without documenting its source and redistribution license.

## Pull requests

Describe the user-visible change, test commands and results, and known
limitations. Changes to the edit model should update `docs/EDIT_MODEL.md`.
