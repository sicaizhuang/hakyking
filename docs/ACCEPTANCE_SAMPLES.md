# Optional Acceptance Samples

The original development project used three local recordings:

1. A five-syllable short vocal phrase.
2. A repeated three-word vocal phrase.
3. A long vocal or BGM file for streaming and non-blocking checks.

Those recordings are not redistributed here. Do not commit copyrighted or
private audio. To run the full local acceptance gate, provide your own
licensed files with these names:

```text
tests/fixtures/acceptance/haya_kunalu.wav
tests/fixtures/acceptance/hakimi_x3.wav
tests/fixtures/acceptance/sarilang_long_vocal.wav
```

Update `qa/acceptance_samples.json` with their measured metadata and expected
slice baseline. Without the files, `dev_tools/acceptance_samples.py` reports
each case as `SKIP` and the public unit suite remains runnable.
