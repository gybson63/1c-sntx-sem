#!/usr/bin/env python3
"""Подготовка релиза: changelog, версия, заметки для GitHub Release."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
SECTION_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _parse_version(raw: str) -> str:
    version = raw.removeprefix("v")
    if not VERSION_RE.match(version):
        msg = f"invalid semver: {raw!r} (expected X.Y.Z)"
        raise SystemExit(msg)
    return version


def _pyproject_version() -> str:
    for line in _read_text(PYPROJECT).splitlines():
        stripped = line.strip()
        if stripped.startswith("version ="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    msg = "version not found in pyproject.toml"
    raise SystemExit(msg)


def _set_pyproject_version(new_version: str) -> None:
    lines = _read_text(PYPROJECT).splitlines(keepends=True)
    updated = False
    for index, line in enumerate(lines):
        if line.strip().startswith("version ="):
            lines[index] = f'version = "{new_version}"\n'
            updated = True
            break
    if not updated:
        msg = "version field not found in pyproject.toml"
        raise SystemExit(msg)
    _write_text(PYPROJECT, "".join(lines))


def _split_changelog(text: str) -> tuple[str, dict[str, str]]:
    header_lines: list[str] = []
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        match = SECTION_RE.match(line.rstrip("\n"))
        if match:
            if current_key is not None:
                sections[current_key] = "".join(current_lines)
            current_key = match.group("version")
            current_lines = [line]
            continue
        if current_key is None:
            header_lines.append(line)
        else:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "".join(current_lines)

    return "".join(header_lines), sections


def _section_has_content(section: str) -> bool:
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            return True
    return False


def extract_notes(version: str) -> str:
    _, sections = _split_changelog(_read_text(CHANGELOG))
    body = sections.get(version)
    if body is None:
        msg = f"section ## [{version}] not found in CHANGELOG.md"
        raise SystemExit(msg)
    return body.strip() + "\n"


def cmd_notes(args: argparse.Namespace) -> None:
    print(extract_notes(_parse_version(args.version)), end="")


def cmd_check(_: argparse.Namespace) -> None:
    header, sections = _split_changelog(_read_text(CHANGELOG))
    if not header.strip():
        raise SystemExit("CHANGELOG.md header is empty")

    unreleased = sections.get("Unreleased", "")
    if not _section_has_content(unreleased):
        raise SystemExit("[Unreleased] has no entries — nothing to release")

    package_version = _pyproject_version()
    print(f"OK: pyproject version={package_version}, [Unreleased] has entries")


def cmd_prepare(args: argparse.Namespace) -> None:
    new_version = _parse_version(args.version)
    current_version = _pyproject_version()

    header, sections = _split_changelog(_read_text(CHANGELOG))
    unreleased = sections.get("Unreleased")
    if unreleased is None:
        raise SystemExit("## [Unreleased] section not found in CHANGELOG.md")
    if not _section_has_content(unreleased):
        raise SystemExit("[Unreleased] is empty — add entries before release")

    if new_version in sections:
        raise SystemExit(f"section ## [{new_version}] already exists in CHANGELOG.md")

    if tuple(map(int, new_version.split("."))) <= tuple(map(int, current_version.split("."))):
        raise SystemExit(
            f"new version {new_version} must be greater than pyproject version {current_version}"
        )

    today = date.today().isoformat()
    released_section = f"## [{new_version}] - {today}\n" + unreleased.split("\n", 1)[1]
    empty_unreleased = "## [Unreleased]\n\n### Added\n\n### Changed\n\n### Fixed\n\n### Removed\n\n"

    ordered: list[str] = ["Unreleased", new_version]
    ordered.extend(key for key in sections if key not in {"Unreleased", new_version})

    new_sections = {**sections, "Unreleased": empty_unreleased, new_version: released_section}
    body = header + "".join(new_sections[key] for key in ordered)
    _write_text(CHANGELOG, body)
    _set_pyproject_version(new_version)

    print(f"Prepared release v{new_version} (was v{current_version})")
    print()
    print("Next steps:")
    print("  git add CHANGELOG.md pyproject.toml")
    print(f'  git commit -m "chore(release): v{new_version}"')
    print(f'  git tag -a v{new_version} -m "Release v{new_version}"')
    print("  git push origin main --tags")
    print()
    print("GitHub Actions создаст Release автоматически после push тега.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    notes = sub.add_parser("notes", help="extract release notes for GitHub Release")
    notes.add_argument("version", help="version (0.2.0 or v0.2.0)")
    notes.set_defaults(func=cmd_notes)

    check = sub.add_parser("check", help="verify [Unreleased] and version")
    check.set_defaults(func=cmd_check)

    prepare = sub.add_parser("prepare", help="finalize CHANGELOG and bump pyproject version")
    prepare.add_argument("version", help="new semver (0.2.0 or v0.2.0)")
    prepare.set_defaults(func=cmd_prepare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
