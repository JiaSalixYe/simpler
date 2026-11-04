# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Native Host imports retain CPU mappings through exported views and close retry."""

import os
from types import SimpleNamespace

import pytest
from _task_interface import DataType
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    MappedArg,
    mint_owner_instance_id,
    wrap_device_malloc,
)
from simpler.comm_endpoints import DEVICE_AICPU, HOST_CPU, AdapterProfile
from simpler.host_domain_backend import SharedHostBacking


def test_native_cpu_import_exported_view_blocks_release_and_closes_admission():
    identity = CanonicalIdentity(mint_owner_instance_id(), 1, 1)
    owner = SharedHostBacking()
    base, exported = owner.allocate(64, identity, 0, 1)
    buffer = Buffer(identity, AddressSpace.HOST, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64, base=base)
    buffer._host_backing = owner
    imported = SharedHostBacking()
    registry = ImportRegistry(ImportContext(HOST_CPU, platform="a2a3sim", runtime="host_build_graph"))
    view = None
    derived = None
    try:
        registry.import_native_host(buffer.to_descriptor(), imported, device=0, exported=exported)
        registry.import_native_host(buffer.to_descriptor(), imported, device=0, exported=exported)
        tensor = buffer.tensor((16,), DataType.FLOAT32)
        resolved = registry.resolve_tensor(tensor)
        native = buffer.buffer
        native[:4] = b"data"
        native.release()
        view = MappedArg(resolved, 0, (16,), (1,), DataType.FLOAT32).buffer
        derived = view[:4]
        view.release()
        view = None
        assert bytes(derived) == b"data"
        with pytest.raises(BufferError):
            registry.unregister(identity)
        with pytest.raises(ValueError, match="not live"):
            registry.resolve_tensor(tensor)
        with pytest.raises(ValueError, match="closing"):
            registry.materialize(buffer.to_descriptor())
        derived.release()
        derived = None
        registry.unregister(identity)
        with pytest.raises(KeyError):
            registry.require(identity)
    finally:
        if derived is not None:
            derived.release()
        if view is not None:
            view.release()
        registry.close()
        imported.release()
        owner.release()


def test_native_cpu_import_read_grant_is_preserved():
    identity = CanonicalIdentity(mint_owner_instance_id(), 2, 1)
    owner = SharedHostBacking()
    base, exported = owner.allocate(64, identity, 0, 1)
    buffer = Buffer(identity, AddressSpace.HOST, AccessMode.READ, BackendKind.REMOTE_SIDECAR, 64, base=base)
    buffer._host_backing = owner
    imported = SharedHostBacking()
    registry = ImportRegistry(ImportContext(HOST_CPU, platform="a2a3sim", runtime="host_build_graph"))
    try:
        view = buffer.buffer
        assert view.readonly
        view.release()
        imported.import_shared(0, exported)
        registry.register_native_host(buffer.to_descriptor(), imported)
        tensor = buffer.tensor((16,), DataType.FLOAT32)
        with pytest.raises(ValueError, match="grant"):
            registry.require_mapping_access(tensor, AccessMode.WRITE)
        buffer.close()
        with pytest.raises(ValueError, match="live"):
            _ = buffer.buffer
    finally:
        registry.close()
        imported.release()
        owner.release()


def test_confirmed_device_mapping_keeps_restricted_grant_and_context():
    owner = mint_owner_instance_id()
    buffer = wrap_device_malloc(4096, 64, owner, 3)
    environment = (os.getpid(), 1, 1, 0, False)
    current: list[tuple[int, int, int, int, bool]] = [environment]
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            device_owner_instance_id=owner,
            execution_environment=environment,
            current_environment=lambda: current[0],
        )
    )
    registry.register_attachment(
        buffer.to_descriptor(),
        buffer.base,
        mapping_ref=1,
        access=AccessMode.READ,
        profile=AdapterProfile.DEVICE_LOCAL,
        live=lambda: True,
    )
    mapped = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1)
    registry.require_mapping_access(mapped, AccessMode.READ)
    with pytest.raises(ValueError, match="grant"):
        registry.require_mapping_access(mapped, AccessMode.WRITE)
    current[0] = (os.getpid(), 1, 2, 0, False)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(mapped)
    registry.close()


@pytest.mark.parametrize("failure_stage", ["import", "register"])
def test_native_import_and_cleanup_failure_retains_closed_cleanup_owner(monkeypatch, failure_stage):
    identity = CanonicalIdentity(mint_owner_instance_id(), 4, 1)
    buffer = Buffer(identity, AddressSpace.HOST, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64)
    registry = ImportRegistry(ImportContext(HOST_CPU, platform="a2a3", runtime="host_build_graph"))
    failure = RuntimeError("injected " + failure_stage + " failure")

    class Backing:
        profile = AdapterProfile.HOST_VMM_MAP
        base = 4096
        closing = False
        releases = 0

        def import_shared(self, device, exported):
            if failure_stage == "import":
                raise failure

        def release(self):
            self.closing = True
            self.releases += 1
            if self.releases == 1:
                raise RuntimeError("injected cleanup failure")

    backing = Backing()
    if failure_stage == "register":

        def refuse(*_args):
            raise failure

        monkeypatch.setattr(registry, "register_native_host", refuse)
    with pytest.raises(RuntimeError) as caught:
        registry.import_native_host(
            buffer.to_descriptor(), backing, device=0, exported=SimpleNamespace(mapping_size=64)
        )
    assert caught.value is failure
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "cleanup failure" in str(caught.value.__cause__)
    assert registry.require(identity).host_backing is backing
    assert backing.releases == 1
    with pytest.raises(ValueError):
        registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32))
    with pytest.raises(ValueError, match="closing"):
        registry.materialize(buffer.to_descriptor())
    registry.unregister(identity)
    assert backing.releases == 2
    with pytest.raises(KeyError):
        registry.require(identity)


def test_device_host_grants_share_import_and_last_unmap_preserves_cpu_import():
    identity = CanonicalIdentity(mint_owner_instance_id(), 5, 1)
    owner = SharedHostBacking()
    base, exported = owner.allocate(64, identity, 0, 1)
    buffer = Buffer(identity, AddressSpace.HOST, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64, base=base)
    environment = (os.getpid(), 1, 1, 0, False)
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )
    cpu, device, unused = SharedHostBacking(), SharedHostBacking(), SharedHostBacking()
    source_live = [True, True]
    try:
        registry.import_native_host(buffer.to_descriptor(), cpu, device=0, exported=exported)
        registry.import_device_host(
            buffer.to_descriptor(),
            device,
            device=0,
            exported=exported,
            mapping_ref=1,
            access=AccessMode.READ,
            live=lambda: source_live[0],
        )
        registry.import_device_host(
            buffer.to_descriptor(),
            unused,
            device=0,
            exported=exported,
            mapping_ref=2,
            access=AccessMode.WRITE,
            live=lambda: source_live[1],
        )
        assert unused.imported is None
        native = buffer.tensor((16,), DataType.FLOAT32)
        first, second = native._with_mapping_ref(1), native._with_mapping_ref(2)
        assert registry.resolve_tensor(first).host_backing is registry.resolve_tensor(second).host_backing is device
        with pytest.raises(ValueError, match="grant"):
            registry.require_mapping_access(first, AccessMode.WRITE)
        source_live[0] = False
        with pytest.raises(ValueError, match="not live"):
            registry.resolve_tensor(first)
        registry.unmap_attachment(identity, 1)
        assert not device.closing
        registry.require_mapping_access(second, AccessMode.WRITE)
        source_live[1] = False
        with pytest.raises(ValueError, match="not live"):
            registry.resolve_tensor(second)
        registry.unmap_attachment(identity, 2)
        assert device.closing and device.imported is None
        assert registry.resolve_tensor(native).host_backing is cpu and not cpu.closing
        with pytest.raises(ValueError):
            registry.resolve_tensor(first)
        registry.unregister(identity)
        assert cpu.closing and cpu.imported is None
    finally:
        registry.close()
        owner.release()


def test_device_import_failure_retains_cleanup_owner_until_unmap_retry():
    identity = CanonicalIdentity(mint_owner_instance_id(), 6, 1)
    buffer = Buffer(identity, AddressSpace.HOST, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64)
    environment = (os.getpid(), 1, 1, 0, False)
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a2a3",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )
    failure = RuntimeError("injected device import failure")

    class Backing:
        attachment_profile = AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT
        closing = False
        releases = 0

        def import_device(self, device, exported):
            raise failure

        def release(self):
            self.closing = True
            self.releases += 1
            if self.releases == 1:
                raise RuntimeError("injected device cleanup failure")

    backing = Backing()
    with pytest.raises(RuntimeError) as caught:
        registry.import_device_host(
            buffer.to_descriptor(),
            backing,
            device=0,
            exported=SimpleNamespace(mapping_size=64),
            mapping_ref=1,
            access=AccessMode.READ,
        )
    assert caught.value is failure and backing.releases == 1
    assert "cleanup failure" in str(failure.__cause__)
    with pytest.raises(ValueError):
        registry.resolve_tensor(buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(1))
    registry.unmap_attachment(identity, 1)
    assert backing.releases == 2
    registry.close()
