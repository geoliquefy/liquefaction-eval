"""Serve the blind form on 127.0.0.1 and write each grade to local JSON.

Routes:
  GET  /                      the form (form/index.html)
  GET  /packet/<path>         files from the packet folder (manifest, items, tiles, RULE.md)
  POST /api/start             {"grader": G}: the first start of a name issues its token and stores it in
                              grades/<grader>/TOKEN; a later start of that name must send the same token
  GET  /api/grades?grader=G   this grader's own latest grades (to resume); needs the X-Grader-Token header
  POST /api/grade             one grade; needs the token; validated, appended to grades/<grader>/<queue_id>.jsonl
  POST /api/lock?grader=G     needs the token; writes grades/<grader>/LOCKED; later saves are refused

Independence (audit design 4.1 item 6): a name is bound to a token at its first start, so a second grader who
types the first grader's name cannot read, write or lock that grader's grades. The form keeps the token in the
browser. A grader who changes browser asks the custodian for the token in grades/<grader>/TOKEN.

Every POST must be Content-Type application/json, and a POST that carries an Origin header must come from this
server's own origin. So a web page open in the grader's browser cannot write or lock grades.

The server binds to 127.0.0.1 only. It never reads a label file: it knows only the packet and the grades folder.
"""
from __future__ import annotations

import hmac
import json
import re
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
FORM = HERE / "form" / "index.html"

EJECTA = ("YES", "NO", "CANNOT TELL")
PROBS = (0.05, 0.25, 0.5, 0.75, 0.95)
SEVERITY = ("none", "minor", "moderate", "severe", "CANNOT TELL")
REASONS = ("R1", "R2", "R3", "R4", "R5", "R6")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class GradeError(ValueError):
    pass


def validate(g: dict, queue: set) -> dict:
    """Return the clean grade or raise GradeError. Fields follow audit design section 4.3."""
    grader, qid = g.get("grader", ""), g.get("queue_id", "")
    if not NAME.match(str(grader)):
        raise GradeError("grader name: letters, digits, _ or -, at most 32")
    if qid not in queue:
        raise GradeError(f"unknown queue id {qid!r}")
    out = {"grader": grader, "queue_id": qid}
    cg = bool(g.get("cannot_grade"))
    out["cannot_grade"] = cg
    if cg:
        if g.get("reason") not in REASONS:
            raise GradeError("CANNOT GRADE needs a reason code R1 to R6")
        if g["reason"] == "R6" and not str(g.get("reason_text", "")).strip():
            raise GradeError("reason R6 needs a written reason")
        out["reason"], out["reason_text"] = g["reason"], str(g.get("reason_text", ""))[:500]
    else:
        if g.get("ejecta") not in EJECTA:
            raise GradeError("ejecta must be YES, NO or CANNOT TELL")
        try:
            p = float(g.get("probability"))
        except (TypeError, ValueError):
            raise GradeError("probability is required") from None
        if not any(abs(p - q) < 1e-9 for q in PROBS):
            raise GradeError("probability must be 0.05, 0.25, 0.50, 0.75 or 0.95")
        if g.get("severity") not in SEVERITY:
            raise GradeError("severity must be none, minor, moderate, severe or CANNOT TELL")
        out.update(ejecta=g["ejecta"], probability=p, severity=g["severity"])
    conf = g.get("confidence")
    if conf not in (1, 2, 3, "1", "2", "3"):
        raise GradeError("confidence must be 1, 2 or 3")
    if g.get("recognise") not in ("yes", "no"):
        raise GradeError("say whether you recognise the site or its published record (yes or no)")
    ev = g.get("evidence_used") or []
    if not isinstance(ev, list) or not all(isinstance(e, str) for e in ev):
        raise GradeError("evidence_used must be a list of tile names")
    out.update(confidence=int(conf), recognise=g["recognise"], evidence_used=ev[:20],
               notes=str(g.get("notes", ""))[:2000])
    try:
        out["seconds"] = max(0, min(int(g.get("seconds", 0)), 6 * 3600))
    except (TypeError, ValueError):
        out["seconds"] = 0
    return out


def write_grade(grades_dir: Path, grade: dict) -> Path:
    d = grades_dir / grade["grader"]
    if (d / "LOCKED").exists():
        raise GradeError("this grader has locked all grades")
    d.mkdir(parents=True, exist_ok=True)
    grade = {**grade, "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    p = d / f"{grade['queue_id']}.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(grade) + "\n")
    return p


def latest_grades(grades_dir: Path, grader: str) -> dict:
    d = grades_dir / grader
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.jsonl")):
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        if lines:
            g = json.loads(lines[-1])
            out[g["queue_id"]] = g
    return out


def start_grader(grades_dir: Path, grader: str, token: str = "") -> str:
    """Return the grader's token. The first start of a name issues one; later starts must present it."""
    if not NAME.match(str(grader)):
        raise GradeError("grader name: letters, digits, _ or -, at most 32")
    d = grades_dir / grader
    tf = d / "TOKEN"
    if tf.exists():
        if not check_token(grades_dir, grader, token):
            raise PermissionError("this grader name is taken; use its token (the custodian has it)")
        return tf.read_text().strip()
    d.mkdir(parents=True, exist_ok=True)
    t = secrets.token_urlsafe(24)
    tf.write_text(t + "\n")
    return t


def check_token(grades_dir: Path, grader: str, token: str) -> bool:
    if not NAME.match(str(grader)):
        return False
    tf = grades_dir / grader / "TOKEN"
    return bool(token) and tf.exists() and hmac.compare_digest(tf.read_text().strip(), str(token))


def make_handler(packet_dir: Path, grades_dir: Path):
    packet_dir, grades_dir = Path(packet_dir).resolve(), Path(grades_dir).resolve()
    queue = set(json.loads((packet_dir / "manifest.json").read_text())["queue"])

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(200, FORM.read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/packet/"):
                p = (packet_dir / u.path[len("/packet/"):]).resolve()
                if packet_dir not in p.parents or grades_dir in p.parents or not p.is_file():
                    return self._json(404, {"error": "not found"})
                ctype = {".json": "application/json", ".png": "image/png", ".md": "text/plain; charset=utf-8"}
                return self._send(200, p.read_bytes(), ctype.get(p.suffix, "application/octet-stream"))
            if u.path == "/api/grades":
                g = parse_qs(u.query).get("grader", [""])[0]
                if not NAME.match(g):
                    return self._json(400, {"error": "bad grader name"})
                if not check_token(grades_dir, g, self.headers.get("X-Grader-Token", "")):
                    return self._json(403, {"error": "wrong or missing grader token"})
                return self._json(200, {"grades": latest_grades(grades_dir, g),
                                        "locked": (grades_dir / g / "LOCKED").exists()})
            return self._json(404, {"error": "not found"})

        def _origin_ok(self) -> bool:
            o = self.headers.get("Origin")
            if o is None:
                return True  # not a browser request (curl, tests)
            port = self.server.server_address[1]
            return o in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")

        def do_POST(self):
            u = urlparse(self.path)
            if not self._origin_ok():
                return self._json(403, {"error": "cross-origin request refused"})
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return self._json(415, {"error": "Content-Type must be application/json"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n > 100_000:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "not JSON"})
            if not isinstance(body, dict):
                return self._json(400, {"error": "expected a JSON object"})
            token = self.headers.get("X-Grader-Token", "")
            try:
                if u.path == "/api/start":
                    return self._json(200, {"token": start_grader(grades_dir, body.get("grader", ""), token)})
                if u.path == "/api/grade":
                    if not check_token(grades_dir, str(body.get("grader", "")), token):
                        raise PermissionError("wrong or missing grader token")
                    g = validate(body, queue)
                    write_grade(grades_dir, g)
                    return self._json(200, {"ok": True, "queue_id": g["queue_id"]})
                if u.path == "/api/lock":
                    g = parse_qs(u.query).get("grader", [""])[0]
                    if not NAME.match(g):
                        raise GradeError("bad grader name")
                    if not check_token(grades_dir, g, token):
                        raise PermissionError("wrong or missing grader token")
                    (grades_dir / g / "LOCKED").write_text(datetime.now(timezone.utc).isoformat())
                    return self._json(200, {"ok": True})
            except GradeError as e:
                return self._json(400, {"error": str(e)})
            except PermissionError as e:
                return self._json(403, {"error": str(e)})
            return self._json(404, {"error": "not found"})

    return H


def serve(packet_dir, grades_dir, port=8765):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(packet_dir, grades_dir))
    print(f"grader form at http://127.0.0.1:{port}/  (grades go to {Path(grades_dir).resolve()}; Ctrl+C stops)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
