"""B4 sandbox, parent side (founder ruling 59a, wave 13; eval M5; rule book B4 and R21). No em dashes.

Before this item a forward entry ran inside the kit's own Python process and could read any file the kit could. Now
`forward.predict` runs every entry (builtin and python) through `run_entry`, once per run, each run in a new process:

```
 kit process (predict)                                   child process (sandbox_child.py, python -I -B)
 ---------------------                                   ----------------------------------------------
 copy inputs/<type>/ --> TEMP/run-N/inputs  (read only)  cwd = TEMP/run-N/inputs
 check the copy against the frozen table                 sees: /inputs/ copy, its job file, the entry code copy,
 copy the entry module file --> TEMP/code-N/  (hashed)         the Python install; writes: TEMP/run-N/outputs only
 empty TEMP/run-N/outputs                                env: PATH=/usr/bin:/bin, HOME=TEMP/run-N, TMPDIR=outputs
 macOS: sandbox-exec profile (kernel layer)              audit hook: a read, list or write elsewhere, a socket, a
 read outputs/.evalkit-result.json  <------------------- new process, ctypes or an object walk ends the process
 delete TEMP                                             with "SANDBOX REFUSED {kind, event, path}", exit 86
```

Two layers on macOS. (1) The audit hook in the child (as the D8 label audit does) names what the entry tried and
stops it at once. (2) The kernel layer, `sandbox-exec` with a deny profile: no network, no process fork, no write
outside outputs, no file read in the home folder, the kit folder or the temp folders except the run's own folders,
the kit's `scoreboard/` code and the Python install. The kernel layer catches what Python's audit events do not see
(C code, a child process). Fail closed (review 1, S1): if `sandbox-exec` cannot start here (an outer sandbox, Linux),
process mode is refused, unless the caller passes `allow_audit_only=True`; the predictions record then names that
choice (`sandbox.kernel_layer`, `sandbox.audit_only`). The audit layer alone is not a boundary against a hostile
pure-Python entry. A kernel-layer denial that the entry catches is NOT a refusal: the entry got nothing, but it runs
on and is scored. Only an audit-layer catch, or an uncaught kernel denial, is SANDBOX REFUSED.

Container mode (`mode="container"`): the same child runs in the kit image with `--network none`, a read-only root,
`/inputs` mounted read-only, `/outputs` the only writable mount, all capabilities dropped, no new privileges, user
nobody, 8 CPU and 32 GB (R21). `container_argv` builds that command; this session's tests check the argv only (no
docker daemon ran), and a live test runs when EVALKIT_SANDBOX_IMAGE names a built image. The container also has a
64 MB tmpfs on /tmp (the read-only root needs one scratch folder); the audit hook refuses an entry write there, and
the tmpfs is gone when the run ends, so "writes only /outputs/" holds for the hook, not for the mounts.

What the sandbox does not guarantee. It is not a security boundary against a hostile native extension on the
process path (the audit layer sees Python events only; the kernel layer is the backstop, and it allows reads outside
the home, kit and temp folders, such as /etc and /usr). File metadata (stat, exists) is not refused by the audit
layer. CPU, memory and the 12 h limit are enforced only in container mode (process mode has a wall-clock timeout).
A python entry is one module file: the registered file is copied alone, so a relative import of a sibling file
fails. The entry may import any package of the Python install. Builtin entries run the kit's own code in the child.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import package as _package
from .spec import SpecError

CHILD = Path(__file__).with_name("sandbox_child.py")
KIT_ROOT = Path(__file__).resolve().parents[2]
VIOLATION_EXIT = 86            # sandbox_child.VIOLATION_EXIT
RESULT = ".evalkit-result.json"
MODES = ("process", "container")
DEFAULT_TIMEOUT_S = 12 * 3600  # R21: 12 h per run
CONTAINER_CPUS, CONTAINER_MEMORY = "8", "32g"
CONTAINER_PYTHON = "/opt/venv-score/bin/python"   # container/Dockerfile: the scoring venv


class SandboxRefused(RuntimeError):
    """An entry tried to read outside /inputs/, write outside /outputs/, open a socket or start a process."""

    def __init__(self, entry: str, kind: str, event: str, path: str | None):
        self.entry, self.kind, self.event, self.path = entry, kind, event, path
        where = f" ({path})" if path else ""
        super().__init__(f"SANDBOX REFUSED: entry {entry!r}: {kind}: {event}{where}. The entry is not run and not "
                         f"scored (ruling 59a: an entry sees only /inputs/ and writes only /outputs/)")

    def record(self) -> dict:
        return {"name": self.entry, "reason": f"SANDBOX REFUSED: {self.kind}", "event": self.event,
                "path": self.path}


@dataclass(frozen=True)
class Layers:
    mode: str
    kernel_layer: str           # "macos sandbox-exec", "container", or "none (...)"
    audit_layer: str = "python audit hook in the child (sandbox_child.py)"

    def record(self) -> dict:
        d = {"mode": self.mode, "kernel_layer": self.kernel_layer, "audit_layer": self.audit_layer,
             "inputs": "a read-only copy of /inputs/, checked against the frozen table",
             "outputs": "/outputs/ only"}
        if self.kernel_layer.startswith("none"):
            d["audit_only"] = ("allowed by the caller (allow_audit_only=True); the audit layer alone does not stop "
                               "a hostile pure-Python entry")
        return d


_SBX_OK: dict = {}


def _sandbox_exec_ok() -> bool:
    """True when /usr/bin/sandbox-exec can start a process here (it cannot inside some outer sandboxes)."""
    if "ok" not in _SBX_OK:
        exe = Path("/usr/bin/sandbox-exec")
        ok = False
        if sys.platform == "darwin" and exe.exists():
            try:
                ok = subprocess.run([str(exe), "-p", "(version 1)(allow default)", "/usr/bin/true"],
                                    capture_output=True, timeout=30).returncode == 0
            except (OSError, subprocess.SubprocessError):
                ok = False
        _SBX_OK["ok"] = ok
    return _SBX_OK["ok"]


def layers(mode: str = "process", *, allow_audit_only: bool = False) -> Layers:
    """The layers a run gets. Fails closed: process mode with no kernel layer is refused unless allow_audit_only."""
    if mode not in MODES:
        raise SpecError(f"sandbox mode must be one of {MODES}, got {mode!r}")
    if mode == "container":
        return Layers(mode, "container")
    if _sandbox_exec_ok():
        return Layers(mode, "macos sandbox-exec")
    if not allow_audit_only:
        raise SpecError("sandbox: no kernel layer here (sandbox-exec cannot start), and the audit layer alone does "
                        "not stop a hostile pure-Python entry; use container mode, or pass allow_audit_only=True "
                        "(sandbox_opts) to run with the audit layer alone and have the record say so")
    return Layers(mode, "none (sandbox-exec not available here; audit layer only)")


def _real(p) -> str:
    return os.path.realpath(str(p))


def _python_roots() -> list[str]:
    return sorted({_real(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)})


def sbpl_profile() -> str:
    """The macOS kernel-layer profile. Paths come in as -D parameters, so no path is quoted into the text."""
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        "(deny file-write*)",
        '(allow file-write* (subpath (param "OUT")) (literal "/dev/null"))',
        '(deny file-read-data (subpath (param "HOME")) (subpath (param "KIT")) (subpath "/private/var/folders")'
        ' (subpath "/private/tmp") (subpath "/Volumes"))',
        '(allow file-read-data (subpath (param "RUN")) (subpath (param "CODE")) (subpath (param "KITCODE"))'
        ' (subpath (param "PY0")) (subpath (param "PY1")) (subpath (param "PY2")) (subpath (param "PY3")))',
    ]) + "\n"


def _chmod_tree(root: Path, file_mode: int, dir_mode: int) -> None:
    for dp, dns, fns in os.walk(root):
        for fn in fns:
            os.chmod(os.path.join(dp, fn), file_mode, follow_symlinks=False)
        os.chmod(dp, dir_mode)


def _rmtree(p: Path) -> None:
    def fix(func, path, _exc):
        try:
            os.chmod(os.path.dirname(path), 0o755)
            os.chmod(path, 0o755 if os.path.isdir(path) else 0o644)
        except OSError:
            pass
        func(path)
    if p.exists():
        shutil.rmtree(p, onexc=fix)


def module_file(mod: str) -> Path | None:
    """The source file of a dotted module, found folder by folder on sys.path. Imports nothing (review 1, B1).

    `importlib.util.find_spec("pkg.entry")` imports `pkg` first, so `pkg/__init__.py` would run in the kit process.
    This walk asks a fresh FileFinder in each folder for one name at a time and follows the folders the finder
    names, as the import system does, but loads no module. A parent package's `__init__.py` never runs: not here,
    and not in the sandbox, which runs a copy of the module file alone. A name that resolves to anything but a
    `.py` source file (an extension module, a bytecode file, a zip) gives None."""
    from importlib.machinery import (BYTECODE_SUFFIXES, EXTENSION_SUFFIXES, SOURCE_SUFFIXES, ExtensionFileLoader,
                                     FileFinder, SourceFileLoader, SourcelessFileLoader)
    parts = str(mod).split(".")
    if not mod or not all(p.isidentifier() for p in parts):
        return None
    details = ((ExtensionFileLoader, EXTENSION_SUFFIXES), (SourceFileLoader, SOURCE_SUFFIXES),
               (SourcelessFileLoader, BYTECODE_SUFFIXES))
    search = list(sys.path)
    spec = None
    for i in range(len(parts)):
        name = ".".join(parts[:i + 1])
        spec, portions = None, []
        for entry in search:
            if not isinstance(entry, str) or not os.path.isdir(entry or "."):
                continue
            found = FileFinder(entry, *details).find_spec(name)
            if found is None:
                continue
            if found.loader is None:                 # a namespace portion: keep looking, as PathFinder does
                portions += list(found.submodule_search_locations or [])
                continue
            spec = found
            break
        if i < len(parts) - 1:
            if spec is not None:
                if spec.submodule_search_locations is None:
                    return None                      # a plain module has no submodules
                search = list(spec.submodule_search_locations)
            elif portions:
                search = portions
            else:
                return None
    if spec is None or not isinstance(spec.loader, SourceFileLoader) or not spec.origin:
        return None
    f = Path(spec.origin)
    return f if f.is_file() else None


def _entry_code(e: dict, code_dir: Path) -> dict:
    """Copy a python entry's registered module file into code_dir; the copy's sha256 must equal code_sha256.

    The module file is found without an import (module_file), so no code of the entry runs in the kit process."""
    import hashlib
    r = e["runner"]
    mod, _, fn = str(r.get("callable", "")).partition(":")
    f0 = module_file(mod) if mod else None
    if f0 is None:
        raise SpecError(f"entry {e['name']!r}: python runner {r.get('callable')!r}: module source file not found")
    data = f0.read_bytes()
    if hashlib.sha256(data).hexdigest() != r.get("code_sha256"):
        raise SpecError(f"entry {e['name']!r}: the module file changed after registration (code_sha256)")
    code_dir.mkdir(parents=True)
    f = code_dir / "entry_module.py"
    f.write_bytes(data)
    _chmod_tree(code_dir, 0o444, 0o555)
    return {"kind": "python", "module": mod, "function": fn, "params": dict(r.get("params") or {}),
            "code_file": str(f)}


def container_argv(image: str, run_dir: Path, code_dir: Path | None) -> list[str]:
    """The docker command for one run: /inputs read-only, /outputs the only writable mount, no network."""
    argv = ["docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", "65534:65534", "--cpus", CONTAINER_CPUS,
            "--memory", CONTAINER_MEMORY, "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--workdir", "/inputs",
            "--env", "HOME=/tmp", "--env", "TMPDIR=/outputs",
            "-v", f"{run_dir / 'inputs'}:/inputs:ro", "-v", f"{run_dir / 'outputs'}:/outputs:rw",
            "-v", f"{run_dir / 'job'}:/job:ro", "-v", f"{KIT_ROOT / 'scoreboard'}:/kit/scoreboard:ro",
            "--entrypoint", CONTAINER_PYTHON]
    if code_dir is not None:
        argv += ["-v", f"{code_dir}:/code:ro"]
    return argv + [image, "-I", "-B", "/kit/scoreboard/evalkit/sandbox_child.py", "/job/job.json"]


def run_entry(e: dict, inputs_dir: str | Path, frozen_rows: dict, *, input_type: str, manifest_sha256: str = "",
              mode: str = "process", timeout_s: float = DEFAULT_TIMEOUT_S, image: str | None = None,
              tmp_root: str | Path | None = None, allow_audit_only: bool = False):
    """Run one entry once in a new sandboxed process and return its values (a numpy array).

    Raises SandboxRefused when the entry tried to leave the sandbox; SpecError for an entry error, a timeout or a
    bad result. The /inputs/ copy is checked against `frozen_rows` (the input manifest's table) before the run."""
    import numpy as np
    lay = layers(mode, allow_audit_only=allow_audit_only)
    r = e["runner"]
    if r.get("kind") not in ("builtin", "python"):
        raise SpecError(f"entry {e['name']!r}: runner kind must be builtin or python")
    if r.get("kind") == "python":
        if not r.get("code_sha256"):
            raise SpecError(f"entry {e['name']!r}: python runner needs code_sha256 (callable_sha256 of its module)")
        if not str(r.get("callable", "")).partition(":")[2]:
            raise SpecError(f"entry {e['name']!r}: python runner needs 'package.module:function'")
    top = Path(tempfile.mkdtemp(prefix="evalkit-sbx-", dir=tmp_root)).resolve()
    try:
        run = top / "run"
        shutil.copytree(inputs_dir, run / "inputs", symlinks=False)
        if _package.folder_rows(run / "inputs") != frozen_rows:
            raise SpecError("sandbox: the /inputs/ copy differs from the frozen table")
        _chmod_tree(run / "inputs", 0o444, 0o555)
        (run / "outputs").mkdir()
        (run / "job").mkdir()
        code_dir = top / "code"
        runner = (_entry_code(e, code_dir) if r["kind"] == "python"
                  else {"kind": "builtin", "name": r.get("name"), "params": dict(r.get("params") or {})})
        if mode == "container":
            paths = {"inputs": "/inputs", "outputs": "/outputs", "job": "/job", "code": "/code", "kit": "/kit"}
            if runner["kind"] == "python":
                runner["code_file"] = "/code/entry_module.py"
        else:
            paths = {"inputs": str(run / "inputs"), "outputs": str(run / "outputs"), "job": str(run / "job"),
                     "code": str(code_dir), "kit": str(KIT_ROOT)}
        # the child adds its own interpreter's install folders in every mode (review 1, S2), so none are named here
        read_roots = [paths["inputs"], paths["outputs"], paths["job"]] + \
                     ([paths["code"]] if runner["kind"] == "python" else [])
        job = {"input_type": input_type, "manifest_sha256": manifest_sha256, "runner": runner,
               "inputs": paths["inputs"], "outputs": paths["outputs"], "kit_root": paths["kit"],
               "code_dir": paths["code"] if runner["kind"] == "python" else None,
               "code_file": runner.get("code_file"), "read_roots": read_roots}
        jp = run / "job" / "job.json"
        jp.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        _chmod_tree(run / "job", 0o444, 0o555)
        if mode == "container":
            if not image:
                raise SpecError("sandbox: container mode needs the kit image name")
            argv = container_argv(image, run, code_dir if runner["kind"] == "python" else None)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        else:
            argv = [sys.executable, "-I", "-B", str(CHILD), str(jp)]
            if lay.kernel_layer == "macos sandbox-exec":
                py = _python_roots()
                py = (py + [py[-1]] * 4)[:4]
                d = {"OUT": str(run / "outputs"), "HOME": _real(Path.home()), "KIT": _real(KIT_ROOT),
                     "RUN": str(run), "CODE": str(code_dir), "KITCODE": _real(KIT_ROOT / "scoreboard"),
                     **{f"PY{k}": v for k, v in enumerate(py)}}
                argv = ["/usr/bin/sandbox-exec", "-p", sbpl_profile()] + \
                       [x for k, v in d.items() for x in ("-D", f"{k}={v}")] + argv
            env = {"PATH": "/usr/bin:/bin", "HOME": str(run), "TMPDIR": str(run / "outputs"), "LC_ALL": "C",
                   "LANG": "C"}
        try:
            cp = subprocess.run(argv, cwd=str(run / "inputs"), env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=timeout_s, close_fds=True)
        except subprocess.TimeoutExpired:
            raise SpecError(f"entry {e['name']!r}: the run passed its time limit ({timeout_s:g} s)") from None
        err = cp.stderr.decode("utf-8", "replace")
        if cp.returncode == VIOLATION_EXIT:
            line = next((ln for ln in reversed(err.splitlines()) if ln.startswith("SANDBOX REFUSED ")), "")
            try:
                d = json.loads(line[len("SANDBOX REFUSED "):])
            except ValueError:
                d = {"kind": "refused (no detail)", "event": "?", "path": None}
            raise SandboxRefused(e["name"], d.get("kind", "?"), d.get("event", "?"), d.get("path"))
        if cp.returncode != 0:
            line = next((ln for ln in reversed(err.splitlines()) if ln.startswith("ENTRY ERROR ")), "")
            if not line and lay.kernel_layer == "macos sandbox-exec" and "Operation not permitted" in err:
                raise SandboxRefused(e["name"], "blocked by the kernel layer", "sandbox-exec", None)
            raise SpecError(f"entry {e['name']!r}: the sandboxed run failed (exit {cp.returncode}): "
                            f"{(line or err.strip()[-500:])}")
        res = run / "outputs" / RESULT
        if res.is_symlink() or not res.is_file():
            raise SpecError(f"entry {e['name']!r}: the sandboxed run wrote no result")
        out = json.loads(res.read_text(encoding="utf-8"))
        return np.asarray(out["values"], dtype=float).reshape(out["shape"])
    finally:
        _rmtree(top)
