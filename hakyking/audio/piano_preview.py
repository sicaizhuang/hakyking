from __future__ import annotations

import math

import numpy as np

from hakyking.audio.playback import prepare_playback_audio


def synthesize_piano_note(
    midi_note: int,
    sample_rate: int = 44100,
    velocity: float = 0.82,
) -> np.ndarray:
    """Generate a compact piano-like preview tone.

    This is not a replacement for a real SoundFont/SFZ piano, but it avoids the
    buzzy sine-wave reference tone by modeling a hammer transient, detuned string
    partials, and piano-like decay.
    """

    midi_note = max(0, min(127, int(midi_note)))
    frequency = 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))
    duration = _duration_for_frequency(frequency)
    sample_count = max(64, int(round(sample_rate * duration)))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate)

    low_note_factor = max(0.0, min(1.0, (180.0 - frequency) / 140.0))
    high_note_factor = max(0.0, min(1.0, (frequency - 900.0) / 1100.0))
    brightness = 1.0 - 0.42 * low_note_factor - 0.28 * high_note_factor

    # Piano-roll preview is primarily a tuning reference, so keep partials
    # harmonically exact instead of adding realistic string inharmonicity.
    partials = (
        (1.0, 1.18, 0.0),
        (2.0, 0.34 * brightness, 0.0),
        (3.0, 0.17 * brightness, 0.0),
        (4.0, 0.09 * brightness, 0.0),
        (5.0, 0.05 * brightness, 0.0),
        (6.0, 0.03 * brightness, 0.0),
    )

    tone = np.zeros(sample_count, dtype=np.float32)
    for multiple, level, detune in partials:
        partial_frequency = frequency * multiple * (1.0 + detune)
        partial_decay = math.exp(-0.18 * multiple) * (1.0 + 1.2 * low_note_factor)
        envelope = np.exp(-t / max(0.08, partial_decay)).astype(np.float32)
        tone += (level * envelope * np.sin(2.0 * math.pi * partial_frequency * t)).astype(np.float32)

    hammer = _deterministic_hammer_noise(sample_count, sample_rate, midi_note)
    attack = 1.0 - np.exp(-t / 0.006)
    body_decay = np.exp(-t / max(0.35, duration * 0.48))
    release_decay = np.exp(-np.maximum(0.0, t - duration * 0.72) / 0.16)
    shaped = tone * attack * body_decay * release_decay + hammer * 0.58

    peak = float(np.max(np.abs(shaped))) if shaped.size else 0.0
    if peak > 0:
        shaped = shaped / peak
    shaped = shaped * max(0.05, min(1.0, float(velocity))) * 0.55
    return prepare_playback_audio(shaped.astype(np.float32), sample_rate, fade_ms=5.0)


def _duration_for_frequency(frequency: float) -> float:
    if frequency < 90:
        return 1.05
    if frequency < 220:
        return 0.82
    if frequency < 880:
        return 0.62
    return 0.42


def _deterministic_hammer_noise(
    sample_count: int,
    sample_rate: int,
    midi_note: int,
) -> np.ndarray:
    noise_length = min(sample_count, max(8, int(round(sample_rate * 0.018))))
    if noise_length <= 0:
        return np.zeros(sample_count, dtype=np.float32)

    rng = np.random.default_rng(seed=10_000 + midi_note)
    noise = rng.normal(0.0, 1.0, noise_length).astype(np.float32)
    envelope = np.exp(-np.arange(noise_length, dtype=np.float32) / (sample_rate * 0.004))
    hammer = np.zeros(sample_count, dtype=np.float32)
    hammer[:noise_length] = noise * envelope * 0.045
    return hammer
