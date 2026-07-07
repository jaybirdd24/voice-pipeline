# PRD: Personalised Multilingual Storytelling Voice Pipeline

> Status: ready-for-agent
> Produced via /to-prd after grilling sessions on 2026-07-06/07.

## Problem Statement

A parent or caregiver cannot always be present to read stories to their child. Generic TTS voices on storytelling devices lack the personal connection of a familiar voice, and it is unknown how much that connection matters. Recording enough audio to train a quality personalised TTS voice the conventional way (hours of clean studio-style recordings, per language) is too burdensome for a real caregiver to do.

The research question: **can a personalised TTS voice, trained from a small amount of caregiver audio, be integrated into a multilingual storytelling device — and how does it affect perceived engagement, perceived understanding, and user preference compared with a generic TTS voice?**

## Solution

A pipeline that turns ~1 minute of caregiver reference audio into a deployable personalised voice:

1. Qwen3-TTS voice-clones the caregiver from the short reference and mass-generates a synthetic training dataset (hundreds of clips) with an automated quality-control gate.
2. Piper is fine-tuned on that synthetic dataset, producing a small, fast model.
3. The Piper model is exported and deployed to a Raspberry Pi 5 storytelling device, where a child selects a preset story via buttons and hears it read live in the caregiver's voice.

The same pipeline runs per language (English, Mandarin, Cantonese), producing one Piper model per language. If a trained Piper voice is not good enough, the manual fallback is pre-generated Qwen3-TTS audio for the preset stories, cached on the device.

## User Stories

1. As a caregiver, I want to record only ~1 minute of my voice, so that creating a personalised voice doesn't demand hours of recording sessions.
2. As a caregiver, I want my cloned voice to read stories in each language I speak, so that my child hears me in the languages we use at home.
3. As a child, I want to press a button on the device and hear a story in my caregiver's voice, so that story time feels familiar even when they're away.
4. As a child, I want story playback to start promptly after selection, so that the device feels responsive.
5. As a researcher, I want to generate a synthetic training dataset from a reference clip with one command, so that dataset creation is repeatable per language and per voice.
6. As a researcher, I want degenerate TTS outputs (screech tails, over-long collapses) automatically quarantined, so that training data is never poisoned by generation failures.
7. As a researcher, I want quarantined clips logged with a machine-readable reason, so that I can audit the QC gate and recalibrate its thresholds as data accumulates.
8. As a researcher, I want dataset generation to resume where it left off, so that an interrupted multi-hour batch doesn't restart from zero.
9. As a researcher, I want to train a Piper model from a generated dataset with one command, so that training runs are consistent and logged.
10. As a researcher, I want each training run recorded in the training notes, so that hyperparameters and outcomes are traceable for the report.
11. As a researcher, I want to export a trained model to a Pi-ready bundle with one command, so that deployment is mechanical.
12. As a researcher, I want to A/B the personalised voice against a generic device voice, so that I can run the engagement/understanding/preference study.
13. As a researcher, I want to pre-generate and cache Qwen3-TTS audio for the preset stories, so that a high-quality fallback exists if a Piper voice disappoints.
14. As a researcher, I want cached story audio keyed on content and language, so that each story is only synthesised once.
15. As a demo presenter, I want the device to work fully offline at the viva, so that the demo doesn't depend on network or a running server.
16. As a demo presenter, I want to manually switch a language between the Piper voice and the cached fallback audio, so that I control quality trade-offs on the day.
17. As a developer/agent, I want each pipeline stage testable at a stable seam, so that stages can be built and verified independently.
18. As a developer/agent, I want all voice-, language- and path-specific values in one config file, so that adding a language or a new caregiver voice requires no code changes.
19. As a study participant, I want the two voice conditions presented identically apart from the voice itself, so that comparisons are fair.

## Implementation Decisions

- **Synthetic data generator:** Qwen3-TTS (0.6B base model) voice-cloning, selected after comparative experiments against CosyVoice2 and GPT-SoVITS. Reference for English: the 32 kHz ~65 s "Milo" reading with verbatim matched transcript.
- **Dataset format:** LJSpeech layout (`metadata.csv` with `stem|text` + `wavs/`) at 22050 Hz — the exact format Piper training consumes. This directory is the contract between generation and training.
- **QC gate (in the generator):** two independent checks — a hard duration cap (default 11 s, catches autoregressive over-generation) and a screech-tail detector (spectral-centroid jump combined with spectral-flux drop, catching stuck-token collapse). Rejected clips are quarantined to a `rejected/` directory and logged to `rejects.csv` with reasons; they never enter `metadata.csv`. Thresholds were calibrated on observed failures and should be revisited as rejects accumulate.
- **Text corpus filtering:** sentences capped at 20 words (long utterances trigger collapse) and terminal punctuation enforced (clean stop cue for the autoregressive model).
- **Per-language configuration:** one YAML config holding reference audio, matched transcript, corpus, espeak voice, pretrained checkpoint, and bundle path per language (`en`, `zh`, `yue`). One Piper model per language; no cross-lingual model.
- **Training:** fine-tune from an existing Piper medium checkpoint using the established training scripts and gradient-accumulation setup. English currently phonemizes with a British espeak voice inherited from earlier runs — flagged for review against the en-AU target.
- **Device:** Raspberry Pi 5, fully offline at demo time. Interaction is preset-story selection via buttons; the underlying speak interface still accepts arbitrary text (used for testing and future free-text input).
- **Fallback:** offline preparation only. Qwen3-TTS pre-generates audio for each preset story; files are cached keyed on hash(text + language) and copied to the device. Switching a language between live Piper and cached fallback is a manual configuration choice, not automatic quality routing.
- **Evaluation:** small-N preliminary user study comparing the personalised voice against a generic device TTS voice on perceived engagement, perceived understanding, and preference.

## Testing Decisions

- Test only external behaviour at three seams, approved by the developer:
  1. **Dataset directory** — given a corpus and reference, generation produces valid `metadata.csv`/`wavs/` entries; QC-failing audio lands in quarantine, never in metadata. Prior art: the QC gate was validated against a real batch containing one known-bad clip (caught) and four known-good clips (passed).
  2. **Fallback story cache** — given a story text and language, a cached WAV exists after prep and is reused (not regenerated) on repeat runs.
  3. **Pi speak interface** — given a voice bundle and text, a valid audible WAV is produced.
- Training and export wrappers are verified by smoke runs (dataset validation script, short training run, ONNX loads and synthesises one sentence) rather than unit tests.
- Model inference in tests: stub the synthesizer for fast tests of orchestration/caching logic; use the real model only in explicit smoke tests.
- A good test asserts on observable outputs (files, formats, audio properties), never on internal implementation details.

## Out of Scope

- Automatic quality-gated fallback routing (fallback selection is manual).
- A live fallback server in the demo (LAN service may exist as a dev tool but is not on the critical path).
- Free-text web UI on the device (device UI is preset-story buttons; free text remains available at the CLI seam).
- LLM story generation (stories are pre-written).
- Fine-tuning or modifying Qwen3-TTS itself.
- The superseded CosyVoice2 and GPT-SoVITS pipelines (preserved in the experimental workspace, not part of this repo).
- Formal large-N evaluation; the study is a small-N preliminary.

## Further Notes

- Hard deadline: university viva/conference in 2–4 weeks (mid/late July 2026). English end-to-end is the minimum viable demo; Mandarin and Cantonese are additive.
- Known risks: Piper/espeak Cantonese (`yue`) support is uncertain — investigate before recording; Qwen3-TTS clone quality on Mandarin/Cantonese references is untested; user-study recruitment inside the window forces small N.
- Mandarin and Cantonese caregiver recordings do not exist yet; the recording step uses the existing recording-studio tooling.
- The experimental workspace (`p4p`) holds large artifacts (models, datasets, checkpoints) referenced by config — they are inputs to this repo, not contents of it.
