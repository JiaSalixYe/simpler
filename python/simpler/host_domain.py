# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Borrowed CPU mappings of native-owned Host communication windows."""

from __future__ import annotations

import os
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory


class HostDomainMapping(SharedMemory):
    """Borrow an L2-owned POSIX window through the standard SharedMemory API.

    L3 only closes its mapping. Creation and unlink belong to the native L2
    domain, so Python's resource tracker must not claim the backing.
    """

    def __init__(self, name: str, size: int):
        if not name or len(name) > 31 or not all(c.isascii() and (c.isalnum() or c in "_-") for c in name):
            raise ValueError("invalid Host domain shared memory name")
        try:
            super().__init__(name=name, create=False, track=False)
        except TypeError:  # Python < 3.13 has no per-instance tracking switch.
            super().__init__(name=name, create=False)
            resource_tracker.unregister(self._name, "shared_memory")
        if size <= 0 or self.size < size:
            self.close()
            raise ValueError("Host domain shared memory is smaller than the mapped window")

    def unlink(self) -> None:
        raise RuntimeError("Host domain shared memory belongs to L2; use domain.release()")

    def close(self) -> None:
        try:
            super().close()
        except BufferError:
            # Outstanding exports keep the mapping alive until their last
            # reference disappears. Stop exposing it through this handle.
            self._buf = None
            self._mmap = None
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
