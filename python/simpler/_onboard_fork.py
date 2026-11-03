# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Process lifetime admission for device bootstrap and forked hierarchies."""

import os
import threading
from contextlib import contextmanager


class _ProcessDeviceState:
    def __init__(self):
        self._lock = threading.RLock()
        self._first_pid = None
        self._fork_depth = 0

    def after_fork(self):
        self._lock = threading.RLock()
        self._fork_depth = 0

    @contextmanager
    def device_init(self, onboard):
        if not onboard:
            yield
            return
        with self._lock:
            if self._fork_depth:
                raise RuntimeError("Onboard initialization cannot run during hierarchy startup")
            if self._first_pid is not None and self._first_pid != os.getpid():
                raise RuntimeError("Onboard initialization inherited device state; use a fresh Python process")
            # Failed bootstrap and successful close both leave process-global CANN state.
            self._first_pid = os.getpid()
            yield

    @contextmanager
    def hierarchy(self, timeout=None):
        lock = self._lock
        acquired = lock.acquire() if timeout is None else lock.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError("hierarchical startup: startup deadline exceeded waiting for process admission")
        try:
            if self._first_pid is not None:
                raise RuntimeError(
                    "Worker hierarchy startup after onboard initialization is unsupported in this process, "
                    "even after close(); start the hierarchy in a fresh Python process"
                )
            owner_pid = os.getpid()
            self._fork_depth += 1
            try:
                yield
            finally:
                if os.getpid() == owner_pid:
                    self._fork_depth -= 1

        finally:
            lock.release()


process_device_state = _ProcessDeviceState()
os.register_at_fork(after_in_child=process_device_state.after_fork)
