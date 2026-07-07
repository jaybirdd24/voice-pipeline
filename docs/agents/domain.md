# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout: single-context

This is a single-context repo — one pipeline, one domain.

```
/
├── CONTEXT.md          (not yet created — created lazily by /domain-modeling)
├── docs/adr/            (not yet created — created lazily by /domain-modeling)
├── docs/PRD.md
├── docs/agents/
├── generate_dataset.py
├── train_piper.sh
├── export_piper.sh
├── fallback/
├── pi/
└── evaluation/
```

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, if it exists.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

Neither exists yet. Proceed silently — `/domain-modeling` (reached via `/grill-with-docs` or `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## Use the glossary's vocabulary

Once `CONTEXT.md` exists, use its terms in issue titles, refactor proposals, and test names rather than drifting to synonyms.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding.
