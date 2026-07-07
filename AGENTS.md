# AGENTS.md

## Agent skills

This repo uses [mattpocock/skills](https://github.com/mattpocock/skills), vendored under `.claude/skills/`.

### Issue tracker

GitHub Issues via the `gh` CLI. PRs are not a triage surface (solo repo, no external contributors). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label strings throughout (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`, `bug`, `enhancement`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo. `CONTEXT.md` and `docs/adr/` don't exist yet — created lazily by `/domain-modeling` as terms and decisions get resolved. See `docs/agents/domain.md`.
