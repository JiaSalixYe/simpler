# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host allocation backend; resource identity and lifetime live in host_domain.

Local exports reuse the native global descriptor v2 envelope, not rank membership.
Its version/profile rules are checked by Python and the platform import backend;
changing that envelope requires a paired Python/native migration.
"""

import ctypes
import os
from multiprocessing.shared_memory import SharedMemory

from _task_interface import _HostDomainMemory

from .buffer import create_host_shared_buffer
from .comm_endpoints import AdapterProfile
from .global_comm_domain import (
    GLOBAL_DOMAIN_PROFILE_A3_FABRIC,
    GLOBAL_DOMAIN_PROFILE_IDS,
    GLOBAL_DOMAIN_PROFILE_SIM,
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainDescriptor,
)


class SharedHostBacking:
    profile = AdapterProfile.HOST_SHM_MAP
    attachment_profile = AdapterProfile.HOST_SHM_MAP

    def __init__(self, *, host_runtime_path=None):
        self.shared = None
        self.imported = None
        self.closing = False

    @staticmethod
    def process_identity():
        return 0

    def authorize(self, identities):
        pass

    def allocate(self, nbytes, identity, device, rank_count):
        page_size = os.sysconf("SC_PAGE_SIZE")
        mapping_size = (nbytes + page_size - 1) // page_size * page_size
        self.shared = create_host_shared_buffer(
            mapping_size,
            bytes(identity.owner_instance_id),
            identity.buffer_id,
            generation=identity.generation,
        )
        return self.shared.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM],
            0,
            rank_count,
            mapping_size,
            self.shared.body,
        )

    @property
    def base(self):
        if self.closing:
            raise ValueError("Host backing is closing")
        if self.imported is not None:
            view = self.imported.buf
            assert view is not None
            return ctypes.addressof(ctypes.c_char.from_buffer(view))
        if self.shared is None:
            raise ValueError("Host backing is not allocated")
        return self.shared.base

    def import_shared(self, device, exported):
        if self.closing or self.shared is not None or self.imported is not None:
            raise ValueError("Host import is not empty")
        if exported.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_SIM]:
            raise ValueError("Host import profile does not match simulation")
        self.imported = SharedMemory(name=exported.handle.decode("utf-8"))
        if self.imported.size < exported.mapping_size:
            raise ValueError("Host import is shorter than the exported extent")

    def import_device(self, device, exported):
        self.import_shared(device, exported)

    def buffer(self, nbytes):
        _ = self.base
        shared = self.imported
        if shared is None:
            assert self.shared is not None
            shared = self.shared.shm
        assert shared is not None
        if nbytes < 0 or nbytes > shared.size:
            raise ValueError("Host buffer view exceeds the backing")
        view = shared.buf
        assert view is not None
        return view[:nbytes]

    def release(self):
        self.closing = True
        if self.imported is not None:
            self.imported.close()
            self.imported = None
        if self.shared is not None:
            self.shared.close()


class A3HostBacking:
    profile = AdapterProfile.HOST_VMM_MAP
    attachment_profile = AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT

    def __init__(self, *, host_runtime_path=None):
        if host_runtime_path is None:
            raise ValueError("Host backing requires an initialized platform runtime")
        self.closing = False
        self.native = _HostDomainMemory(str(host_runtime_path))
        self.native.require_mapping_support()

    def allocate(self, nbytes, identity, device, rank_count):
        size, token = self.native.allocate(device, 0, nbytes)
        return self.native.base, GlobalDomainDescriptor(
            GLOBAL_DOMAIN_VERSION,
            GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC],
            0,
            rank_count,
            size,
            token,
        )

    @property
    def base(self):
        if self.closing:
            raise ValueError("Host backing is closing")
        return self.native.base

    def import_shared(self, device, exported):
        if self.closing:
            raise ValueError("Host backing is closing")
        if exported.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC]:
            raise ValueError("Host import profile does not match A3 Fabric")
        self.native.import_shared(device, exported.handle, exported.mapping_size)

    def import_device(self, device, exported):
        if self.closing:
            raise ValueError("Host backing is closing")
        if exported.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[GLOBAL_DOMAIN_PROFILE_A3_FABRIC]:
            raise ValueError("Host import profile does not match A3 Fabric")
        self.native.import_shared(device, exported.handle, exported.mapping_size, device_view=True)

    def buffer(self, nbytes):
        _ = self.base
        return memoryview(self.native.buffer(nbytes))

    def process_identity(self):
        return self.native.process_identity()

    def authorize(self, identities):
        self.native.authorize(identities)

    def release(self):
        self.closing = True
        self.native.release()


_HOST_BACKENDS = {"a2a3sim": SharedHostBacking, "a2a3": A3HostBacking}


def host_backing_type(platform):
    """Select before any allocation. Unsupported platforms fail without side effects."""
    try:
        return _HOST_BACKENDS[platform]
    except KeyError:
        raise ValueError(f"Host domain backing is not implemented for {platform!r}") from None


def host_attachment_profile(descriptor: GlobalDomainDescriptor, platform: str) -> AdapterProfile:
    expected = GLOBAL_DOMAIN_PROFILE_SIM if platform == "a2a3sim" else GLOBAL_DOMAIN_PROFILE_A3_FABRIC
    backend = host_backing_type(platform)
    if descriptor.profile_id != GLOBAL_DOMAIN_PROFILE_IDS[expected]:
        raise ValueError("Host attachment descriptor does not match the platform backend")
    return backend.attachment_profile
