#!/usr/bin/env python3
"""Confirm the user-guide pages exist and their local links resolve."""
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "NOTICE",
    "AI_USE.md",
    "docs/notes/commands.md",
    "docs/notes/README.md",
    "environment.yml",
    "requirements.txt",
    "docs/index.html",
    "docs/guide/index.html",
    "docs/guide/commands.html",
    "docs/guide/fmf.html",
    "docs/guide/control.html",
    "docs/guide/simplify.html",
    "docs/guide/libraries.html",
    "docs/html/index.html",
    "docs/html/assets/style.css",
    ".gitignore",
    ".gitlab-ci.yml",
    ".github/workflows/ci.yml",
]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)


def main():
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        print("missing files:")
        for rel in missing:
            print(" ", rel)
        return 1
    errors = []
    pages = list((ROOT / "docs").rglob("*.html"))
    for page in pages:
        text = page.read_text(encoding="utf-8")
        if "<title>" not in text or "</html>" not in text:
            errors.append(f"{page.relative_to(ROOT)}: not a complete html document")
        parser = Links()
        parser.feed(text)
        for href in parser.hrefs:
            if href.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = (page.parent / href.split("#", 1)[0]).resolve()
            if not target.exists():
                errors.append(f"{page.relative_to(ROOT)}: broken link {href}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"docs ok ({len(pages)} html pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
