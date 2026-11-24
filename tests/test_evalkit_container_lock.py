"""evalkit container lock (eval M4, D1): every third-party module the evalkit code imports is in
container/requirements-score.lock, with every dependency it needs on Linux.

Why: wave 4 (docs/EVAL-WAVE4-2026-11-23.md section 8) found that the image from container/Dockerfile stopped
at gate G-k because pyproj, imported by graderkit/geo.py, was missing from the lock. The Dockerfile installs the
lock with pip --no-deps, so a dependency of a locked package (pyproj needs certifi) must be in the lock too.
The test reads the source with ast, so a lazy import inside a function counts.

Scope (container-v31 review 1, nit 3): the scan starts at all of scoreboard/evalkit, and it follows local
imports (scoreboard.* and relative imports such as "from ...scorer import auc") to the files they load, with
the __init__.py of every package on the way. Dependency markers are evaluated for the image (Linux, aarch64
and x86_64), not for this machine. No em dashes."""
import ast
import importlib.metadata as md
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "container" / "requirements-score.lock"
EVALKIT = ROOT / "scoreboard" / "evalkit"
LOCAL_TOP = "scoreboard"
# The image runs Linux on both build targets. A dependency counts if its marker holds on either one.
LINUX_ENVS = tuple({"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
                    "platform_machine": m, "extra": ""} for m in ("aarch64", "x86_64"))


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _lock_pins() -> dict:
    pins = {}
    for line in LOCK.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, ver = line.partition("==")
        assert sep, f"lock line is not name==version: {line!r}"
        pins[_norm(name)] = ver.strip()
    return pins


def _module_files(dotted: str) -> list:
    """Files Python runs to import a local dotted module: each package __init__.py on the path, then the
    module itself. Empty if the name is not a local module (for example a function name)."""
    parts = dotted.split(".")
    files = []
    for i in range(1, len(parts) + 1):
        base = ROOT.joinpath(*parts[:i])
        if (base / "__init__.py").is_file():
            files.append(base / "__init__.py")
        elif i == len(parts) and base.with_suffix(".py").is_file():
            files.append(base.with_suffix(".py"))
        else:
            return files if i == len(parts) else []
    return files


def _package_of(path: Path) -> list:
    rel = path.relative_to(ROOT).with_suffix("").parts
    return list(rel[:-1])  # for __init__.py this drops "__init__" and keeps the package itself


def _local_targets(node: ast.ImportFrom, path: Path) -> list:
    """Dotted local names an ImportFrom can load: the module, and each imported name as a submodule."""
    if node.level:
        pkg = _package_of(path)
        base = pkg[: len(pkg) - (node.level - 1)]
        mod = ".".join(base + ([node.module] if node.module else []))
    else:
        mod = node.module
    return [mod] + [f"{mod}.{a.name}" for a in node.names if a.name != "*"]


def _third_party_imports(start: Path) -> tuple:
    """Top-level module name -> the files that import it, over start/**.py and every local file they load.
    Also returns the set of local files scanned."""
    found = {}
    queue = sorted(start.rglob("*.py"))
    seen = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
                local = [n for n in names if n.split(".", 1)[0] == LOCAL_TOP]
            elif isinstance(node, ast.ImportFrom):
                if node.level or (node.module or "").split(".", 1)[0] == LOCAL_TOP:
                    local = _local_targets(node, path)
                    names = []
                else:
                    names = [node.module] if node.module else []
                    local = []
            else:
                continue
            for dotted in local:
                queue.extend(_module_files(dotted))
            for n in names:
                top = n.split(".", 1)[0]
                if top in sys.stdlib_module_names or top == LOCAL_TOP or top == "__future__":
                    continue
                found.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return found, seen


def _requires(dist: str) -> list:
    """Names of the run-time dependencies of an installed distribution (no extras) whose marker holds on a
    Linux build target of the image."""
    from packaging.requirements import Requirement

    out = []
    for raw in md.requires(dist) or []:
        req = Requirement(raw)
        if req.marker is not None and not any(req.marker.evaluate(env) for env in LINUX_ENVS):
            continue
        out.append(_norm(req.name))
    return out


def test_lock_is_name_equals_version_and_sorted():
    pins = _lock_pins()
    names = [_norm(line.partition("==")[0]) for line in LOCK.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert names == sorted(names), "the lock keeps pip freeze order (sorted, case-insensitive)"
    assert len(pins) == len(names), "a package is pinned twice"


def test_lock_pins_pyproj_and_certifi_like_the_kit():
    pins = _lock_pins()
    kit = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        name, sep, ver = line.strip().partition("==")
        if sep:
            kit[_norm(name)] = ver
    for name in ("pyproj", "certifi"):
        assert pins.get(name) == kit.get(name), f"{name}: lock {pins.get(name)} vs kit {kit.get(name)}"


def test_scan_follows_local_imports_out_of_evalkit():
    _, files = _third_party_imports(EVALKIT)
    rel = {str(p.relative_to(ROOT)) for p in files}
    assert "scoreboard/evalkit/graderkit/geo.py" in rel
    assert "scoreboard/scorer.py" in rel, "relative import 'from ...scorer import' was not followed"
    assert "scoreboard/__init__.py" in rel


def test_linux_markers_are_used():
    from packaging.requirements import Requirement

    req = Requirement('fake; sys_platform == "linux" and platform_machine == "aarch64"')
    assert any(req.marker.evaluate(env) for env in LINUX_ENVS)
    assert not any(Requirement('fake; sys_platform == "darwin"').marker.evaluate(env) for env in LINUX_ENVS)


def test_every_evalkit_import_is_in_the_lock_with_its_dependencies():
    pins = _lock_pins()
    imports, _ = _third_party_imports(EVALKIT)
    assert "pyproj" in imports and "numpy" in imports, "the ast scan lost the known imports"
    owners = md.packages_distributions()
    missing = []
    todo = []
    for mod, files in sorted(imports.items()):
        dists = owners.get(mod)
        if not dists:
            pytest.fail(f"module {mod} (imported by {sorted(files)}) is not installed here; cannot map it")
        for d in dists:
            todo.append((_norm(d), f"{mod} in {sorted(files)}"))
    seen = set()
    while todo:
        dist, why = todo.pop()
        if dist in seen:
            continue
        seen.add(dist)
        if dist not in pins:
            missing.append(f"{dist} (needed by {why})")
            continue
        try:
            deps = _requires(dist)
        except md.PackageNotFoundError:
            pytest.fail(f"{dist} is in the lock but not installed here; cannot read its dependencies")
        for dep in deps:
            todo.append((dep, f"{dist}"))
    assert not missing, "container/requirements-score.lock lacks: " + "; ".join(sorted(missing))
