# Agent Rules

This directory contains rule sets used by AI coding agents working on this
repository. The files are vendored copies (not external references) so the
rules are always available regardless of network access or changes upstream.

## Contents

- `clean-code.mini.md` — readability and local-reasoning rules.
- `a-philosophy-of-software-design.mini.md` — module/API design rules.
- `LICENSE` — MIT license covering the two `*.mini.md` rule files.

## Source and attribution

The two `*.mini.md` files were copied verbatim from the MIT-licensed
`ciembor/agent-rules-books` project:

- Upstream: https://github.com/ciembor/agent-rules-books
- Vendored from commit: `af756d4c2413d687119666f76db21582ec7c3618`
- Files:
  - `clean-code/clean-code.mini.md`
  - `a-philosophy-of-software-design/a-philosophy-of-software-design.mini.md`

The accompanying `LICENSE` file is the upstream MIT license and applies to
those two vendored files. Project-specific guidance lives in the top-level
`AGENTS.md` and is covered by this repository's own license.

## Updating

To refresh the vendored rules, replace the `*.mini.md` files with the latest
versions from upstream and update the commit SHA above. Do not edit the
vendored files in place; project-specific overrides belong in `AGENTS.md`.
