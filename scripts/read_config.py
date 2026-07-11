#!/usr/bin/env python3
"""Read values from config.yaml as shell variable assignments.

Usage (from a bash wrapper):

    eval "$(python3 scripts/read_config.py --config config.yaml --lang en \
        DATASETS_DIR=datasets_dir \
        ESPEAK_VOICE=languages.{lang}.espeak_voice)"

Each VAR=dotted.key argument prints one shell-quoted `VAR='value'` line.
`{lang}` inside a key is substituted from --lang. A missing or empty key is a
hard error (exit 1, nothing printed for it), so an unconfigured language fails
loudly instead of propagating blank paths.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--lang", default=None, help="Substituted for {lang} in keys")
    parser.add_argument("mappings", nargs="+", metavar="VAR=dotted.key")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    lines = []
    for mapping in args.mappings:
        var, sep, key = mapping.partition("=")
        if not sep or not var or not key:
            print(f"ERROR: bad mapping {mapping!r} (expected VAR=dotted.key)", file=sys.stderr)
            return 1
        if "{lang}" in key:
            if not args.lang:
                print(f"ERROR: key {key!r} requires --lang", file=sys.stderr)
                return 1
            key = key.replace("{lang}", args.lang)

        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                print(f"ERROR: key not found in {args.config}: {key}", file=sys.stderr)
                return 1
            node = node[part]
        if node is None or (isinstance(node, str) and not node.strip()):
            print(
                f"ERROR: config key is empty: {key} (fill it in {args.config})",
                file=sys.stderr,
            )
            return 1
        lines.append(f"{var}={shlex.quote(str(node))}")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
