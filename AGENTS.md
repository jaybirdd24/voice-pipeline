# AGENTS.md

## Agent skills

This repo uses [mattpocock/skills](https://github.com/mattpocock/skills), vendored under `.claude/skills/`.

### Issue tracker

GitHub Issues via the `gh` CLI. PRs are not a triage surface (solo repo, no external contributors). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label strings throughout (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`, `bug`, `enhancement`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. `CONTEXT.md` and `docs/adr/` don't exist yet — created lazily by `/domain-modeling` as terms and decisions get resolved. See `docs/agents/domain.md`.

## Tests

```
.venv/bin/python -m pytest
```

The repo-local `.venv` (create with `python3 -m venv .venv && .venv/bin/pip install pytest numpy scipy soundfile pyyaml`) is deliberately light: tests stub the Qwen3-TTS synthesizer at its seam, so no torch/qwen-tts needed. Real-model smoke runs use the Qwen3 venv at `/home/jay/p4p/qwen3_tts_test/.venv/` instead. Tests live in `tests/` and assert only at the approved seams (dataset directory, fallback cache, Pi speak interface).
