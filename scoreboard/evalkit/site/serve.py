"""Local preview server for a built scorecard site. It applies the site's `_headers` file the way Netlify does
(path patterns with a trailing `*`), serves `404.html` with status 404 for a missing path, and binds 127.0.0.1
only. For the launch check (item D9); it deploys nothing. No em dashes.

Usage: python -m scoreboard.evalkit.site.serve <site folder> [port]
"""
from __future__ import annotations

import http.server
import sys
from pathlib import Path

TYPES = {".md": "text/markdown; charset=utf-8", ".txt": "text/plain; charset=utf-8",
         ".webmanifest": "application/manifest+json; charset=utf-8", ".svg": "image/svg+xml",
         ".json": "application/json", ".xml": "application/xml", ".html": "text/html; charset=utf-8"}


def parse_headers(text: str) -> list[tuple[str, list[tuple[str, str]]]]:
    rules, cur = [], None
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if not ln.startswith((" ", "\t")):
            cur = (ln.strip(), [])
            rules.append(cur)
        elif cur is not None and ":" in ln:
            k, v = ln.strip().split(":", 1)
            cur[1].append((k.strip(), v.strip()))
    return rules


def matches(pattern: str, path: str) -> bool:
    if pattern == "/*":
        return True
    if pattern.startswith("/*."):
        return path.endswith(pattern[2:])
    return path == pattern


def handler(root: Path):
    rules = parse_headers((root / "_headers").read_text(encoding="utf-8")) if (root / "_headers").exists() else []

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

        def send_head(self):
            path = self.path.split("?", 1)[0]
            f = root / path.lstrip("/")
            if f.is_dir():
                f = f / "index.html"
            if not f.resolve().is_relative_to(root.resolve()) or not f.is_file():
                f, status = root / "404.html", 404
            else:
                status = 200
            body = f.read_bytes()
            self.send_response(status)
            ctype = TYPES.get(f.suffix, self.guess_type(str(f)))
            extra = [(k, v) for pat, hs in rules if matches(pat, path) for k, v in hs]
            if not any(k.lower() == "content-type" for k, _ in extra):
                self.send_header("Content-Type", ctype)
            for k, v in extra:
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            import io
            return io.BytesIO(body)

    return H


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root, port = Path(argv[0]).resolve(), int(argv[1]) if len(argv) > 1 else 8795
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler(root)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
