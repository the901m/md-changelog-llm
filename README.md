# md-changelog-llm

A lightweight Python utility that tracks changes across local Markdown note vaults, parses intelligent line-level diffs, and uses a local LLM server to automatically write concise daily changelogs.

Designed specifically to work with local models (like Gemma or Llama via `llama-server`) without relying on heavy external dependencies.

---

## Features

* **SQLite State Tracking:** Monitors local directory file states and hashes without requiring a full Git setup.
* **Smart Diff Processing:** Automatically categorizes line modifications, task completions, new file creations, moves, and deletions instead of raw line dumps.
* **Lazy LLM Bootstrapping:** Scans for changes first. If no relevant file changes exist, the local LLM server process is never launched.
* **Fuzzy Rename Detection:** Distinguishes genuine file moves/renames from structural deletions using similarity thresholds.
* **Dry-Run Mode:** Test prompt payloads and output strings with `--dry-run` without modifying the database or invoking the LLM.
* **Local Inference Compatibility:** Includes native support for OpenAI-compatible local endpoints (`llama-server`) with configurable sampling parameters (temperature, top_p, min_p, top_k).

---

## Prerequisites

* Python 3.8+
* `llama-server` (or any local OpenAI-compatible API server)

---

## Usage

### Standard Run
Scans target directory, generates summary entries via the local LLM, and appends results to your changelog file:

```bash
python3 gitloggerllm.py
```

### Dry Run
Inspect the generated diffs, payload JSON structure, and mock output without executing LLM inference or writing to SQLite:
Bash
```bash
python3 gitloggerllm.py --dry-run
```

## Configuration
Edit the top parameters inside gitloggerllm.py to match your local setup:

- TARGET_DIR: Path to your Markdown notes or Obsidian vault.
- CHANGELOG_FILE: Output file path for daily changelog entries.
- DB_FILE: Path to the SQLite snapshot database.
- LLAMA_SERVER_CMD: Launch command for your local llama-server binary and GGUF model path.
- LLAMA_URL: Local endpoint URL (default: http://localhost:8080/v1/chat/completions).
