# Architecture Map

Hakyking follows a small MVC-style split. Keep UI gestures, undo commands,
worker threads, and DSP algorithms separate when extending it.

## Layers

- `hakyking/main.py`: application bootstrap and Qt setup.
- `hakyking/views/`: Qt widgets, painting, drag/drop, keyboard and mouse intent.
- `hakyking/controllers/`: signal wiring, workers, project actions, playback,
  rendering dispatch, and undo registration.
- `hakyking/models/`: project state and non-destructive edit models.
- `hakyking/audio/`: probing, reading, slicing, pitch analysis, DSP, playback,
  waveform generation, and export.
- `hakyking/commands.py`: `QUndoCommand` implementations for user-visible edits.

## Edit Model

`hakyking/models/audio_edit.py` is the canonical boundary:

- A clip owns source interval, timeline placement, duration, gain, and pitch
  center.
- Pitch automation owns control points and parameterized vibrato regions.
- A control point is an editing marker; it does not cut audio.
- A render request is a read-only snapshot consumed by DSP and playback.

The existing V/B/N tools are intentionally distinct during the feature freeze:

- V selects, moves, and box-selects existing control points.
- B adds or removes control points.
- N edits parameterized vibrato on a selected continuous curve segment.

Persistent edits must enter the global undo stack.

## Threading Rule

Expensive file I/O, waveform analysis, pitch detection, slicing, rendering,
and export belong in `QThread` workers. Widgets should never synchronously
decode or analyze a long media file.

## Extension Points

- Add a new DSP engine under `hakyking/audio/` and route it through the common
  render request rather than assembling parameters inside a widget.
- Add a new user edit in `hakyking/commands.py` and cover undo/redo first.
- Put new persistent project fields in the model and project serializer before
  adding view state.
- Keep optional external tools and model weights outside the source repository;
  document their versions and licenses in `THIRD_PARTY_NOTICES.md`.
