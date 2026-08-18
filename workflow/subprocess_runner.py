"""Generic bounded subprocess execution for Controller-dispatched external executors.

R28 exposed a defect where a dispatched external executor (the acquisition ``augment-atoms``
subprocess) could hang indefinitely with no wall-time budget, no process-group ownership, and no
heartbeat -- so the Controller had no way to bound it, detect progress, or clean it up. This
module is the single, generic mechanism both dispatch paths use to fix that:

- ``workflow.controller.RunController.run_stage`` (the generic command-list stage path)
- ``adapters.acquisition.run_augment_atoms`` (the in-process pydantic_ai acquisition path)

Every launched process gets its OWN process group (``start_new_session=True``) so a timeout kill
can terminate exactly the tree this module started and nothing else sharing the host.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass
class BoundedRunResult:
    returncode: Optional[int]
    timed_out: bool
    elapsed_s: float
    pid: Optional[int]


def run_bounded(command: Sequence[str], *, cwd, env, stdout, stderr=subprocess.STDOUT,
                timeout_s: Optional[float] = None,
                on_start: Optional[Callable[[int], None]] = None,
                heartbeat_cb: Optional[Callable[[], None]] = None,
                heartbeat_interval_s: float = 5.0,
                poll_interval_s: float = 0.2,
                grace_s: float = 10.0) -> BoundedRunResult:
    """Launch ``command`` in its own process group and bound it by wall-clock time.

    ``on_start(pid)`` fires immediately after the process is spawned (before the first poll), so a
    caller can durably record the pid before any heartbeat can possibly fire for it.
    ``heartbeat_cb()`` fires at most every ``heartbeat_interval_s`` while the process is still
    running. If ``timeout_s`` elapses, ONLY this process's own process group is sent SIGTERM, given
    ``grace_s`` to exit, then SIGKILL if still alive -- never any other process on the host.
    """
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                               start_new_session=True)
    if on_start is not None:
        on_start(process.pid)
    last_heartbeat = start
    timed_out = False
    returncode = None
    try:
        while True:
            try:
                returncode = process.wait(timeout=poll_interval_s)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - start
                if timeout_s is not None and elapsed >= timeout_s:
                    timed_out = True
                    _kill_process_group(process, grace_s=grace_s)
                    returncode = process.wait()
                    break
                now = time.monotonic()
                if heartbeat_cb is not None and (now - last_heartbeat) >= heartbeat_interval_s:
                    heartbeat_cb()
                    last_heartbeat = now
    finally:
        elapsed_s = time.monotonic() - start
    return BoundedRunResult(returncode=returncode, timed_out=timed_out, elapsed_s=elapsed_s,
                            pid=process.pid)


def _kill_process_group(process: "subprocess.Popen", *, grace_s: float) -> None:
    """Terminate ONLY the process group this module started -- never a process outside it."""
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
