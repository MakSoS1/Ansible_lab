#!/usr/bin/env python3
"""Fail CI if any E-CUP workflow can create a GitHub Release/prerelease.

Submission distribution is private-Hugging-Face-only. Historical workflows are
scanned too, because a manually dispatched old workflow must not be able to
reintroduce a Release.
"""
from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path('.github/workflows')

# Intentionally scoped to workflow YAML files; this source file contains the
# policy patterns themselves and must not scan itself.
FORBIDDEN = {
    'gh release create': re.compile(r'\bgh\s+release\s+create\b', re.I),
    'softprops/action-gh-release': re.compile(r'softprops/action-gh-release', re.I),
    'ncipollo/release-action': re.compile(r'ncipollo/release-action', re.I),
    'actions/create-release': re.compile(r'actions/create-release', re.I),
    'PyGithub create_git_release/create_release': re.compile(r'\bcreate_(?:git_)?release\s*\(', re.I),
    'REST create-release POST': re.compile(r'(?:curl|gh\s+api)[^\n]{0,300}(?:-X|--method)\s*(?:POST|post)[^\n]{0,300}/releases\b', re.I),
}


def main() -> None:
    offenders: list[str] = []
    if not WORKFLOWS.exists():
        raise SystemExit(f'missing workflow directory: {WORKFLOWS}')

    for path in sorted(WORKFLOWS.glob('*.y*ml')):
        text = path.read_text(encoding='utf-8')
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                offenders.append(f'{path}: {label}')

    if offenders:
        print('E-CUP GitHub Release policy: FAILED')
        for item in offenders:
            print(f' - {item}')
        raise SystemExit(1)

    print('E-CUP GitHub Release policy: OK (private Hugging Face only)')


if __name__ == '__main__':
    main()
