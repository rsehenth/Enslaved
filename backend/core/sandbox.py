"""Subprocess resource-limit runner (Windows Job Object memory cap + timeout)."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _job_for(proc: subprocess.Popen, memory_limit: int):
    """Creates a Job Object with memory limit and kill-on-close, assigning `proc` to it. None outside Windows/on failure."""
    if os.name != "nt":
        return None  # shortcut: no memory cap outside Windows; use resource.setrlimit if ever run there
    import ctypes
    from ctypes import wintypes

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic), ("IoInfo", ctypes.c_ulonglong * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _Extended()
    # JOB_MEMORY (0x200) | DIE_ON_UNHANDLED_EXCEPTION (0x400) | KILL_ON_JOB_CLOSE (0x2000)
    info.BasicLimitInformation.LimitFlags = 0x200 | 0x400 | 0x2000
    info.JobMemoryLimit = memory_limit
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) and k32.AssignProcessToJobObject(
        job, wintypes.HANDLE(int(proc._handle))  # type: ignore[attr-defined]
    )
    if not ok:
        k32.CloseHandle(job)
        return None
    return (k32, job)


def _close_job(job) -> None:
    """Closes the job: KILL_ON_JOB_CLOSE terminates the child and all living descendants."""
    if job is not None:
        k32, handle = job
        k32.CloseHandle(handle)


def run_limited(cmd: list[str], *, timeout: float, memory_limit: int, env: dict | None = None, cwd: str | None = None):
    """Runs `cmd` with memory cap (Windows) and timeout; returns (returncode, stdout, isolated, timed_out).

    The child waits for 1 byte on stdin before starting so it is already inside the job when it works.
    """
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env, cwd=cwd,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    job = _job_for(proc, memory_limit)
    isolated = job is not None
    if not isolated and os.name == "nt":
        logger.warning("Isolated execution without memory limit: failed to create/assign Job Object.")
    timed_out = False
    try:
        try:
            out, _ = proc.communicate(b"g", timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _close_job(job)
            job = None
            proc.kill()
            out, _ = proc.communicate()
    finally:
        _close_job(job)
    return proc.returncode, out, isolated, timed_out
