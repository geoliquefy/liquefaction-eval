"""Planted forward entries for the B4 sandbox tests (ruling 59a). Synthetic only. No em dashes.

Each function is a python runner: fn(inputs, **params) -> one value per site. The bad ones catch their own error and
return honest values, so a test shows that the sandbox refuses the attempt even when the entry hides the failure.
"""
import os

import numpy as np


def honest(inp):
    """Reads /inputs/ (its working folder) and writes a scratch file to /outputs/ (TMPDIR): both allowed."""
    with open("event.json", encoding="utf-8") as f:
        assert f.read().startswith("{")
    with open(os.path.join(os.environ["TMPDIR"], "scratch.txt"), "w", encoding="utf-8") as f:
        f.write("ok")
    return -np.hypot(inp.x_m, inp.y_m)


def reads_label(inp, path):
    """Tries to read a label file outside /inputs/."""
    try:
        with open(path, encoding="utf-8") as f:
            f.read()
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def writes_outside(inp, path):
    """Tries to write a file outside /outputs/."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("leak")
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def opens_socket(inp):
    """Tries to open a network socket."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(("127.0.0.1", 9))
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def starts_process(inp, path):
    """Tries to read a file through a new process."""
    import subprocess
    try:
        subprocess.run(["/bin/cat", path], capture_output=True)
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def lists_outside(inp, path):
    """Tries to list a folder outside /inputs/ (names of label files are a leak too)."""
    try:
        os.listdir(path)
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def imports_stdlib(inp):
    """Honest: imports a standard library module the child has not loaded yet (review 1, S2: container roots)."""
    import sys
    assert "colorsys" not in sys.modules
    import colorsys
    assert colorsys.rgb_to_hsv(1.0, 0.0, 0.0)[0] == 0.0
    return -np.hypot(inp.x_m, inp.y_m)


def dir_fd_read(inp, path):
    """Review 1, S1: opens an allowed folder as a handle, then reads a file outside through dir_fd."""
    import sys
    try:
        fd = os.open(sys.prefix, os.O_RDONLY)
        rel = os.path.relpath(path, os.path.realpath(sys.prefix))
        g = os.open(rel, os.O_RDONLY, dir_fd=fd)
        os.read(g, 100)
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def dotdot_read(inp, path):
    """Review 1, S1: a relative path with '..' parts (the shape a dir_fd read needs)."""
    try:
        with open(os.path.relpath(path), encoding="utf-8") as f:
            f.read()
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)


def sqlite_outside(inp, path):
    """Review 1, S1: sqlite3 opens its file in C, with no open audit event; tries to write a database outside."""
    try:
        import sqlite3
        con = sqlite3.connect(path)
        con.execute("create table t (x)")
        con.commit()
        con.close()
    except Exception:
        pass
    return -np.hypot(inp.x_m, inp.y_m)
