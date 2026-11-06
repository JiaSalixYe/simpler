# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Borrowed CPU mappings of L3-owned Host communication windows."""

from __future__ import annotations

import os
from multiprocessing.shared_memory import SharedMemory


class HostDomainMapping(SharedMemory):
    """Borrow a POSIX window from its L3 owner through the SharedMemory API."""

    def __init__(self, owner: SharedMemory, size: int):
        if size <= 0 or owner.size < size:
            raise ValueError("Host domain shared memory is smaller than the mapped window")
        try:
            super().__init__(name=owner.name, create=False, track=False)
        except TypeError:  # Python < 3.13 has no per-instance tracking switch.
            # Registration is idempotent in the L3 owner's tracker. Leave
            # the name tracked until the owner unlinks it.
            super().__init__(name=owner.name, create=False)

    def unlink(self) -> None:
        raise RuntimeError("Host domain shared memory belongs to the domain; use domain.release()")

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
