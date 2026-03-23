#!/usr/bin/env python3
"""
Append a release entry to docs/changelog.md with the current package version and date.

Usage (from repo root):
    python scripts/update_changelog.py "Brief summary of changes"
    CHANGELOG_SUMMARY="Summary" python scripts/update_changelog.py

The script reads the version from the installed hst123 package (setuptools-scm)
and inserts a new line under "## Releases" in docs/changelog.md.
"""
import os
import sys
from datetime import date


def get_version():
    try:
        import hst123
        return hst123.__version__
    except Exception:
        pass
    try:
        from setuptools_scm import get_version as _get
        return _get()
    except Exception:
        return "0.0.0+unknown"


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    changelog_path = os.path.join(repo_root, "docs", "changelog.md")

    summary = (
        sys.argv[1].strip() if len(sys.argv) > 1
        else os.environ.get("CHANGELOG_SUMMARY", "").strip()
    )
    if not summary:
        print("Usage: python scripts/update_changelog.py \"Brief summary of changes\"", file=sys.stderr)
        print("   or: CHANGELOG_SUMMARY=\"Summary\" python scripts/update_changelog.py", file=sys.stderr)
        sys.exit(1)

    version = get_version()
    today = date.today().isoformat()
    line = f"- **v{version.lstrip('v')}** ({today}) — {summary}\n"

    with open(changelog_path, "r") as f:
        content = f.read()

    marker = "## Releases\n"
    if marker not in content:
        print("Changelog has no '## Releases' section.", file=sys.stderr)
        sys.exit(1)

    # Insert after "## Releases" and any intro lines (e.g. italic note); before existing bullets
    head, _, rest = content.partition(marker)
    rest_lines = rest.split("\n")
    insert_at = 0
    for i, row in enumerate(rest_lines):
        if row.strip().startswith("- **"):
            insert_at = i
            break
        if row.strip().startswith("*") or not row.strip():
            insert_at = i + 1
    rest_lines.insert(insert_at, line.rstrip())
    new_rest = "\n".join(rest_lines)

    with open(changelog_path, "w") as f:
        f.write(head + marker + new_rest)

    print(f"Added to docs/changelog.md: v{version} ({today}) — {summary}")


if __name__ == "__main__":
    main()
