# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cross-node mappings reuse restricted records and the current endpoint context."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    mint_owner_instance_id,
)
from simpler.comm_domain import CommDomainRef
from simpler.comm_endpoints import DEVICE_AICORE, HOST_CPU, AdapterProfile
from simpler.global_comm_domain import GlobalDomainPhase
from simpler.host_domain import HostDomainView, HostMappingRecords
from simpler.task_interface import CommDomainHandle, DataType, GlobalDomainMember
from simpler.worker import Worker


def source_buffer(space=AddressSpace.HOST):
    return Buffer(
        CanonicalIdentity(mint_owner_instance_id(), 1, 1), space, AccessMode.READWRITE, BackendKind.REMOTE_SIDECAR, 64
    )


@pytest.mark.parametrize("space", [AddressSpace.HOST, AddressSpace.DEVICE])
def test_global_records_request_only_selected_access_and_revoke(space):
    buffer = source_buffer(space)
    environment = (os.getpid(), 1, 1, 0, False)
    chip = SimpleNamespace(_validate_execution_context=lambda: None, _execution_environment=environment)
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICORE,
            platform="a2a3sim",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: chip._execution_environment,
        )
    )
    detached = []
    view = HostDomainView(
        buffer.to_descriptor(),
        None,
        0,
        environment,
        lambda: True,
        attach=lambda: 4096,
        detach=lambda: detached.append(1),
        profile=AdapterProfile.HOST_SHM_MAP,
    )
    records = HostMappingRecords(
        buffer,
        (7,),
        view_scope=lambda *_: nullcontext(),
        view_control=lambda _resource, _worker, ref, access, unmap: view.apply(
            chip, registry, 0, 1, ref, access, unmap=unmap, platform="a2a3sim"
        ),
        reference_scope=lambda _worker, count: 2 + 4 * count,
    )
    tensor = buffer.tensor((16,), DataType.FLOAT32)
    mapped = records.map_view(tensor, 7, access=AccessMode.READ)
    assert mapped.buffer == tensor.buffer and mapped.mapping_ref == 2
    records.validate(mapped, 7)
    registry.require_mapping_access(mapped, AccessMode.READ)
    with pytest.raises(ValueError, match="grant"):
        registry.require_mapping_access(mapped, AccessMode.WRITE)
    with pytest.raises(ValueError, match="no granted attachment"):
        registry.resolve_tensor(tensor._with_mapping_ref(6))
    with pytest.raises(ValueError, match="target worker"):
        records.validate(mapped, 3)
    records.unmap_view(mapped)
    assert detached == [1]
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(mapped)
    again = records.map_view(tensor, 7, access=AccessMode.WRITE)
    assert again.mapping_ref == 6
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(mapped)
    chip._execution_environment = (os.getpid(), 1, 2, 0, False)
    with pytest.raises(ValueError, match="not live"):
        registry.resolve_tensor(again)


def test_global_reference_namespaces_are_disjoint_and_exhaust_without_reuse():
    buffer = source_buffer()
    calls = []
    records = [
        HostMappingRecords(
            buffer,
            (0,),
            view_scope=lambda *_: nullcontext(),
            view_control=lambda *args, **kw: calls.append((args, kw)),
            reference_scope=lambda _worker, count, rank=rank: rank + 1 + 64 * count,
        )
        for rank in (0, 63)
    ]
    tensor = buffer.tensor((16,), DataType.FLOAT32)
    refs = []
    for item, count in zip(records, (4, 3)):
        for _ in range(count):
            mapped = item.map_view(tensor, 0, access=AccessMode.READ)
            refs.append(mapped.mapping_ref)
            item.unmap_view(mapped)
        before = len(calls)
        with pytest.raises(RuntimeError, match="exhausted"):
            item.map_view(tensor, 0)
        assert len(calls) == before
    assert len(refs) == len(set(refs))


def test_global_unmap_failure_keeps_record_closed_and_retries_cleanup():
    buffer = source_buffer()
    failure = [False]

    def control(*_args, unmap, **_kw):
        if unmap and failure[0]:
            raise RuntimeError("detach failed")

    records = HostMappingRecords(buffer, (0,), view_scope=lambda *_: nullcontext(), view_control=control)
    tensor = buffer.tensor((16,), DataType.FLOAT32)
    mapped = records.map_view(tensor, 0)
    failure[0] = True
    with pytest.raises(RuntimeError, match="detach failed"):
        records.unmap_view(mapped)
    with pytest.raises(ValueError, match="live explicit mapping"):
        records.validate(mapped, 0)
    with pytest.raises(RuntimeError, match="unconfirmed"):
        records.map_view(tensor, 0)
    failure[0] = False
    records.unmap_view(mapped)
    assert records.map_view(tensor, 0).mapping_ref != mapped.mapping_ref


@pytest.mark.parametrize("profile", [AdapterProfile.DEVICE_FABRIC_V2_PEER_IMPORT, AdapterProfile.HOST_SHM_MAP])
def test_native_device_attachment_requires_target_context(profile):
    buffer = source_buffer(AddressSpace.DEVICE)
    with pytest.raises(ValueError, match="device endpoint"):
        ImportRegistry(ImportContext(HOST_CPU)).register_attachment(
            buffer.to_descriptor(), 4096, mapping_ref=0, profile=profile, live=lambda: True
        )
    with pytest.raises(ValueError, match="execution environment"):
        ImportRegistry(ImportContext(DEVICE_AICORE)).register_attachment(
            buffer.to_descriptor(), 4096, mapping_ref=0, profile=profile, live=lambda: True
        )


@pytest.mark.parametrize("writing", [False, True])
def test_borrowed_copy_checks_owner_before_native_routing(monkeypatch, writing):
    worker = Worker(level=3, num_sub_workers=0)
    owner = mint_owner_instance_id()
    state = SimpleNamespace(
        phase=GlobalDomainPhase.COMMIT,
        view=object(),
        command=SimpleNamespace(owner_instance_id=owner, generation=1),
    )
    monkeypatch.setattr(worker, "_global_node_domains", {1: state})
    calls = []

    def copy(command, *, copy_to_device):
        calls.append((command, copy_to_device))
        return b"data"

    monkeypatch.setattr(worker, "_copy_global_domain_node", copy)

    def invoke(identity):
        handle = CommDomainHandle(
            name="test",
            _release_fn=None,
            domain_id=1,
            generation=1,
            members=(GlobalDomainMember(0, 0, 0, 0),),
            ref=CommDomainRef(identity, 1, 1),
        )
        if writing:
            worker._copy_to_global_domain(handle, 0, b"data", 0)
        else:
            worker._copy_from_global_domain(handle, 0, 4, 0)

    with pytest.raises(KeyError, match="no live grant"):
        invoke(mint_owner_instance_id())
    assert not calls
    invoke(owner)
    assert len(calls) == 1 and calls[0][1] is writing
