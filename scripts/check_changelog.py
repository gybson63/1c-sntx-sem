#!/usr/bin/env python3
"""Pre-commit: require CHANGELOG.md update when application code changes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = "CHANGELOG.md"
CODE_PREFIX = "src/sntx_sem/"


def _git_output(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def staged_files(repo_root: Path = ROOT) -> list[str]:
    output = _git_output(
        ["diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=repo_root,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def staged_diff(path: str, repo_root: Path = ROOT) -> str:
    return _git_output(["diff", "--cached", "--", path], cwd=repo_root)


def check_changelog(*, repo_root: Path = ROOT) -> str | None:
    """Return an error message when application code changes without CHANGELOG."""
    files = staged_files(repo_root)
    code_changed = any(f.replace("\\", "/").startswith(CODE_PREFIX) for f in files)
    if not code_changed:
        return None

    if not staged_diff(CHANGELOG, repo_root).strip():
        return (
            "CHANGELOG.md must be updated when changing src/sntx_sem/.\n"
            "Add an entry under ## [Unreleased] (Added / Changed / Fixed / Removed)."
        )
    return None


def main() -> int:
    error = check_changelog()
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
