# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HostRegister backend and URMA transport skeleton.

HostRegister owns the local CPU/device mapping. URMA is a separate transport
backend: an absent/unsupported transport must fail explicitly, never fall back
to HostRegister or an A3 Fabric implementation.
"""

from __future__ import annotations

import os

from .buffer import Buffer, create_host_shared_buffer
from .comm_endpoints import AdapterProfile


class HostTransportUnsupported(RuntimeError):
    """The A5 transport backend is not qualified or not compiled in this build."""


class A5HostRegisterEntry:
    """The HostRegister backend paired with its per-view mapping operations."""

    profile = AdapterProfile.HOST_SVM_MAP
    attachment_profile = AdapterProfile.HOST_SVM_MAP

    def __init__(self, chip_worker):
        self.chip = chip_worker
        self.backend = self
        self.execution_environment = getattr(chip_worker, "_execution_environment", None)
        self._pid = os.getpid()
        self.shared: Buffer | None = None

    def _check_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("A5 HostRegister mapping belongs to another process")

    def allocate(self, spec, identity, device, rank_count) -> Buffer:
        self._check_process()
        if not getattr(self.chip, "host_register_supported", lambda: False)():
            raise RuntimeError("A5 HostRegister is not supported by this runtime")
        self.shared = create_host_shared_buffer(
            int(spec.nbytes),
            owner_instance_id=bytes(identity.owner_instance_id),
            buffer_id=int(identity.buffer_id),
            owner_worker_path="a5-host-register",
            generation=int(identity.generation),
        )
        return self.shared

    def release(self):
        self._check_process()
        if self.shared is not None:
            self.shared.close()
            self.shared = None

    def attach(self, cw, host_base: int, nbytes: int) -> int:
        self._check_process()
        return int(cw.host_register_map(int(host_base), int(nbytes)))

    def detach(self, cw, host_base: int) -> None:
        self._check_process()
        cw.host_register_unmap(int(host_base))


class SharedHostBacking:
    """A5 simulator has no HostRegister capability; it is never a silent fallback."""

    profile = AdapterProfile.HOST_SHM_MAP
    attachment_profile = AdapterProfile.HOST_SHM_MAP

    def __init__(self, chip_worker):
        raise RuntimeError("A5 HostRegister is not implemented for the a5 simulator")


class A5UrmaBackend:
    """Transport skeleton only; no local mapping or data path is implemented."""

    def __init__(self, chip_worker):
        self.chip = chip_worker

    def capability(self) -> bool:
        probe = getattr(self.chip, "host_urma_supported", None)
        return bool(probe()) if probe is not None else False

    def register(self, *args, **kwargs):
        raise HostTransportUnsupported("A5 URMA transport is not qualified or not compiled")

    def post(self, *args, **kwargs):
        raise HostTransportUnsupported("A5 URMA transport is not qualified or not compiled")

    def wait(self, *args, **kwargs):
        raise HostTransportUnsupported("A5 URMA transport is not qualified or not compiled")


_HOST_BACKENDS = {"a5": A5HostRegisterEntry, "a5sim": SharedHostBacking}


def host_backing_type(platform):
    try:
        return _HOST_BACKENDS[platform]
    except KeyError:
        raise ValueError(f"HostRegister backing is not implemented for {platform!r}") from None
