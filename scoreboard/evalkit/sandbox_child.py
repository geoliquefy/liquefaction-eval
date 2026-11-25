"""B4 sandbox, child side (ruling 59a, wave 13; eval M5). Runs ONE forward entry once and exits. No em dashes.

The parent (`evalkit.sandbox`) starts this file as a script, `python -I -B sandbox_child.py JOB.json`, in a fresh
process with a scrubbed environment, its working folder set to a read-only copy of the type's /inputs/ folder. The
file imports only the standard library at the top, so nothing of the kit runs before the parent's job is read.

```
 1. trusted setup (no entry code yet): read JOB.json, import numpy and the kit's /inputs/ reader
 2. build FrozenInputs from the /inputs/ copy ONLY (event.json, sites.csv, package/cpt_list.csv, shakemap/grid.xml)
 3. drop the kit from sys.path, install the audit hook (it cannot be removed)
 4. import the entry module (a copy of the registered file) and call it; any refused event ends the process at once
 5. write the values to /outputs/.evalkit-result.json (O_EXCL, no symlink follow) and exit 0
```

The audit hook allows: reads under the /inputs/ copy, the job folder, the entry code copy, /outputs/ and this
interpreter's own Python install (its real sys.prefix, base_prefix, exec_prefix and base_exec_prefix, added here in
every mode, so an honest import works in the container too); writes only under /outputs/. It refuses, and ends the
process at once with exit code VIOLATION_EXIT and one line "SANDBOX REFUSED {json}" on stderr: a read or directory
listing elsewhere, a write elsewhere, any socket or network event, a new process, ctypes, gc object walks and os.kill.
Review 1 (S1) added: an open of a folder (a folder handle is what a dir_fd call needs, and the open event does not
carry dir_fd), a relative path with a '..' part, sqlite3.connect to a file outside /outputs/ (sqlite opens its file
in C, with no open event), sqlite3 extension loading, and the dbm C modules. The entry cannot catch this: the hook
calls os._exit before the event returns to it. The hook alone is not a boundary against a hostile pure-Python
entry (C code that opens files without an audit event is out of its sight); the parent refuses to run without a
kernel layer unless the caller allows the audit layer alone (sandbox.py).
"""
import json
import os
import sys

VIOLATION_EXIT = 86
RESULT = ".evalkit-result.json"
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_NET_PREFIXES = ("socket.", "urllib.", "http.client.", "ftplib.", "smtplib.", "poplib.", "imaplib.", "nntplib.",
                 "telnetlib.", "webbrowser.")
_PROC_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                "pty.spawn", "os.startfile", "os.kill", "os.killpg", "signal.pthread_kill", "_posixsubprocess")
_TAMPER_EVENTS = ("gc.get_objects", "gc.get_referrers", "gc.get_referents", "sys.remote_exec")
_WRITE_PATH_EVENTS = {"os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1), "os.chmod": (0,),
                      "os.chown": (0,), "os.chflags": (0,), "os.utime": (0,), "os.truncate": (0,), "os.symlink": (1,),
                      "os.link": (0, 1), "os.setxattr": (0,), "os.removexattr": (0,), "os.lchmod": (0,),
                      "os.lchown": (0,), "os.lchflags": (0,), "os.mkfifo": (0,), "os.mknod": (0,)}
_REFUSED_IMPORTS = ("_dbm", "_gdbm", "dbm.ndbm", "dbm.gnu")      # C modules that open files with no open event
_READ_PATH_EVENTS = {"os.listdir": 0, "os.scandir": 0, "os.chdir": 0, "os.listxattr": 0, "os.getxattr": 0,
                     "os.fwalk": 0, "glob.glob": 0}


def _real(p):
    if p is None:
        p = "."
    p = os.fsdecode(os.fspath(p))
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(os.getcwd(), p))


def _under(p, roots):
    return any(p == r or p.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def _dotdot(p):
    """True for a relative path with a '..' part. Such a path resolves against a dir_fd the hook cannot see."""
    if p is None or isinstance(p, int):
        return False
    try:
        s = os.fsdecode(os.fspath(p))
    except TypeError:
        return False
    return not os.path.isabs(s) and ".." in s.replace("\\", "/").split("/")


def interpreter_roots():
    """This interpreter's own install folders (real paths). The child adds them in every mode (review 1, S2)."""
    return sorted({os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)})


def _install_hook(read_roots, write_roots):
    """Add the audit hook. Its state lives in this closure only (no module global), so entry code holds no handle."""
    read_roots, write_roots = tuple(read_roots), tuple(write_roots)
    busy = [False]

    def refuse(kind, event, path=None):
        msg = {"kind": kind, "event": event, "path": None if path is None else str(path)[:300]}
        try:
            os.write(2, ("SANDBOX REFUSED " + json.dumps(msg, sort_keys=True) + "\n").encode("utf-8", "replace"))
        finally:
            os._exit(VIOLATION_EXIT)

    def hook(event, args):
        if busy[0]:
            return
        busy[0] = True
        try:
            if event == "open":
                path, mode, flags = (tuple(args) + (None, None, None))[:3]
                if isinstance(path, int):
                    return
                if _dotdot(path):
                    refuse("relative path with '..'", event, path)
                p = _real(path)
                if os.path.isdir(p):
                    refuse("folder handle (dir_fd paths are not checked)", event, path)
                write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                        (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
                if write and not _under(p, write_roots) and p != os.devnull:
                    refuse("write outside /outputs/", event, path)
                if not write and not _under(p, read_roots) and p != os.devnull:
                    refuse("read outside /inputs/", event, path)
            elif event.startswith(_NET_PREFIXES):
                refuse("network or socket", event, args[1] if event == "socket.connect" and len(args) > 1 else None)
            elif event.startswith(_PROC_EVENTS):
                refuse("new process or signal", event)
            elif event.startswith("ctypes."):
                refuse("ctypes", event)
            elif event in _TAMPER_EVENTS:
                refuse("object walk", event)
            elif event in _WRITE_PATH_EVENTS:
                for i in _WRITE_PATH_EVENTS[event]:
                    if i < len(args) and _dotdot(args[i]):
                        refuse("relative path with '..'", event, args[i])
                    if i < len(args) and not isinstance(args[i], int) and \
                            not _under(_real(args[i]), write_roots):
                        refuse("write outside /outputs/", event, args[i])
            elif event in _READ_PATH_EVENTS:
                a = args[0] if args else None
                if _dotdot(a):
                    refuse("relative path with '..'", event, a)
                if not isinstance(a, int) and not _under(_real(a), read_roots):
                    refuse("read outside /inputs/", event, a)
            elif event == "sqlite3.connect":
                db = args[0] if args else None
                if isinstance(db, (str, bytes, os.PathLike)):
                    s = os.fsdecode(os.fspath(db))
                    if s not in ("", ":memory:") and (s.startswith("file:") or _dotdot(s) or
                                                       not _under(_real(s), write_roots)):
                        refuse("sqlite3 file outside /outputs/", event, s)
            elif event in ("sqlite3.enable_load_extension", "sqlite3.load_extension"):
                refuse("sqlite3 extension loading", event)
            elif event == "import" and args and args[0] in _REFUSED_IMPORTS:
                refuse("dbm C module (opens files with no audit event)", event, args[0])
        finally:
            busy[0] = False

    sys.addaudithook(hook)


def _load_entry(path, module_name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    # 1. trusted setup: the kit's /inputs/ reader, from the kit path the parent names, before any entry code runs
    import types
    import numpy as np
    kit = job["kit_root"]
    for name, sub in (("scoreboard", "scoreboard"), ("scoreboard.evalkit", os.path.join("scoreboard", "evalkit"))):
        m = types.ModuleType(name)          # the package __init__ files are not run: the reader needs none of them
        m.__path__ = [os.path.join(kit, sub)]
        sys.modules[name] = m
    sys.modules["scoreboard"].evalkit = sys.modules["scoreboard.evalkit"]
    from scoreboard.evalkit import forward as F
    # 2. the entry's inputs, from the /inputs/ copy only
    inp = F.inputs_from_folder(job["inputs"], job["input_type"], job.get("manifest_sha256", ""))
    r = job["runner"]
    # 3. no kit path on sys.path; the entry code folder first; then the hook
    roots = list(job["read_roots"]) + interpreter_roots()      # review 1, S2: the stdlib in every mode
    keep = [p for p in sys.path if _under(_real(p), roots)]
    sys.path[:] = [job["code_dir"]] + keep if job.get("code_dir") else keep
    _install_hook(roots, [job["outputs"]])
    del job_path
    # 4. the entry
    params = dict(r.get("params") or {})
    if r["kind"] == "builtin":
        p = F._builtin(r["name"], inp, params)
    else:
        mod = _load_entry(job["code_file"], r["module"])
        p = getattr(mod, r["function"])(inp, **params)
    vals = [float(v) for v in np.asarray(p, dtype=float).ravel()]
    shape = list(np.asarray(p).shape)
    # 5. the result, written once; an existing file or a planted link is refused by the open itself
    fd = os.open(os.path.join(job["outputs"], RESULT), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"shape": shape, "values": vals}, f)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except SystemExit:
        raise
    except BaseException as exc:          # an entry error: one line for the parent, then exit 3
        os.write(2, ("ENTRY ERROR " + json.dumps({"type": type(exc).__name__, "message": str(exc)[:500]}) + "\n")
                 .encode("utf-8", "replace"))
        os._exit(3)
