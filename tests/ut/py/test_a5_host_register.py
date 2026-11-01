# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 HostRegister mapping and failure-path tests (no device required)."""

from __future__ import annotations

import contextlib
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
from simpler.buffer import (
    AccessMode,
    AddressSpace,
    CanonicalIdentity,
    ImportContext,
    ImportRegistry,
    Tensor,
    create_host_shared_buffer,
    mint_owner_instance_id,
)
from simpler.comm_domain import HostWindowSpec
from simpler.comm_endpoints import DEVICE_AICPU, AdapterProfile
from simpler.host_domain import HostDomainResource
from simpler.host_domain_backend import A5HostRegisterEntry, A5UrmaBackend, HostTransportUnsupported
from simpler.task_interface import (
    ArgDirection,
    CallConfig,
    DataType,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
    _materialize_chip_task_args,
)
from simpler.worker import Worker, _Lifecycle


class FakeChip:
    def __init__(self):
        self.environment = (os.getpid(), 1, 1, 0, False)
        self._execution_environment = self.environment
        self.unmapped: list[int] = []
        self.map_calls: list[tuple[int, int]] = []
        self.map_error: BaseException | None = None
        self.unmap_error: BaseException | None = None
        self.map_address = 0x8000

    def _validate_execution_context(self):
        return None

    def host_register_supported(self):
        return True

    def host_register_map(self, host, nbytes):
        self.map_calls.append((int(host), int(nbytes)))
        if self.map_error is not None:
            raise self.map_error
        return self.map_address

    def host_register_unmap(self, host):
        if self.unmap_error is not None:
            raise self.unmap_error
        self.unmapped.append(int(host))


class FakeWorker:
    def __init__(self, chip, registry):
        self.chip = chip
        self.registry = registry

    @contextlib.contextmanager
    def scope(self, *_args, **_kwargs):
        yield

    def control(self, resource, worker, reference, access, *, unmap):
        resource._direct_view.apply(self.chip, self.registry, reference, access, unmap=unmap)


def _host_buffer(resource):
    buffer = resource.buffer
    assert buffer is not None
    return buffer


def _resource():
    chip = FakeChip()
    environment = chip.environment
    registry = ImportRegistry(
        ImportContext(
            DEVICE_AICPU,
            platform="a5",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )
    )
    identity = CanonicalIdentity(mint_owner_instance_id(), 1, 1)
    mapping = A5HostRegisterEntry(chip)
    resource = HostDomainResource(
        HostWindowSpec(128),
        identity,
        (0,),
        mapping=mapping,
        view_scope=None,
        view_control=None,
    )
    worker = FakeWorker(chip, registry)
    resource._view_scope = worker.scope
    resource._view_control = worker.control
    resource.allocate()
    return resource, chip, registry, worker


def test_a5_hostregister_maps_one_view_and_unmaps_last():
    resource, chip, _registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)
        mapped = resource.map_view(native, 0, access=AccessMode.READ)
        assert mapped.mapping_ref and mapped.buffer == native.buffer
        with pytest.raises(RuntimeError, match="Worker.release_buffer"):
            _host_buffer(resource).close()
        assert mapped.buffer.address_space == AddressSpace.HOST
        assert chip.map_calls and chip.map_calls[0][0] == _host_buffer(resource).base
        args = TaskArgs()
        args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
        target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.IN)
        materialized = _materialize_chip_task_args(args, _registry, target)
        assert materialized.tensor(0).data == 0x8000
        assert materialized.tensor(0).address_space == AddressSpace.DEVICE
        with pytest.raises(ValueError, match="explicit mapped view"):
            resource.validate(native, 0)
        resource.unmap_view(mapped)
        assert chip.unmapped == [int(_host_buffer(resource).base)]
        with pytest.raises(ValueError, match="not live|no issued mapping"):
            _registry.resolve_tensor(mapped)
        second = resource.map_view(native, 0, access=AccessMode.READ)
        assert second.mapping_ref != mapped.mapping_ref
        resource.unmap_view(second)
        resource.close()
    finally:
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_a5_hostregister_unmap_failure_enters_closing_and_retries():
    resource, chip, _registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)
        mapped = resource.map_view(native, 0, access=AccessMode.READ)
        chip.unmap_error = RuntimeError("injected unmap failure")
        with pytest.raises(RuntimeError, match="injected unmap"):
            resource.unmap_view(mapped)
        with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
            resource.map_view(native, 0, access=AccessMode.READ)
        chip.unmap_error = None
        resource.unmap_view(mapped)
        remapped = resource.map_view(native, 0, access=AccessMode.READ)
        assert remapped.mapping_ref > mapped.mapping_ref
        resource.unmap_view(remapped)
        resource.close()
    finally:
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_a5_hostregister_map_failure_has_no_grant_and_cleans_up():
    resource, chip, registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)
        chip.map_error = RuntimeError("injected map failure")
        with pytest.raises(RuntimeError, match="injected map"):
            resource.map_view(native, 0, access=AccessMode.READ)
        with pytest.raises(ValueError, match="no granted attachment"):
            registry.resolve_tensor(native._with_mapping_ref(1))
        chip.map_error = None
        mapped = resource.map_view(native, 0, access=AccessMode.READ)
        resource.unmap_view(mapped)
        resource.close()
    finally:
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_a5_hostregister_unsupported_is_explicit():
    class UnsupportedChip(FakeChip):
        def host_register_supported(self):
            return False

    mapping = A5HostRegisterEntry(UnsupportedChip())
    with pytest.raises(RuntimeError, match="HostRegister is not supported"):
        mapping.allocate(HostWindowSpec(128), SimpleNamespace(owner_instance_id=b"o", buffer_id=1, generation=1))


def test_a5_urma_skeleton_never_falls_back():
    chip = FakeChip()
    transport = A5UrmaBackend(chip)
    assert transport.capability() is False
    with pytest.raises(HostTransportUnsupported):
        transport.register(0, 1)
    with pytest.raises(HostTransportUnsupported):
        transport.post()
    with pytest.raises(HostTransportUnsupported):
        transport.wait()


def test_worker_host_window_lifecycle_uses_host_register():
    chip = FakeChip()
    worker = Worker(
        level=2,
        platform="a5",
        runtime="tensormap_and_ringbuffer",
        device_id=0,
    )
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    try:
        buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
        assert buffer.address_space == AddressSpace.HOST
        native = buffer.tensor((32,), DataType.FLOAT32)
        mapped = worker.map_view(native, access=AccessMode.READ)
        args = TaskArgs()
        args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
        worker._l2_materialize_local.target = SimpleNamespace(sig_count=1, sig=lambda _index: ArgDirection.IN)
        try:
            materialized = worker._materialize_l2_args(args)
        finally:
            worker._l2_materialize_local.target = None
        assert materialized.tensor(0).data == 0x8000
        with pytest.raises(RuntimeError, match="unmap every mapped view"):
            worker.release_buffer(buffer)
        worker.unmap_view(mapped)
        worker.release_buffer(buffer)
        assert not worker._direct_host_domains
    finally:
        worker._chip_run_touched_identities.clear()
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


class TestMappedTensorContract:
    def _context(self):
        environment = (os.getpid(), 1, 1, 0, False)
        return ImportContext(
            DEVICE_AICPU,
            platform="a5",
            runtime="tensormap_and_ringbuffer",
            execution_environment=environment,
            current_environment=lambda: environment,
        )

    def test_native_and_mapped_views_share_one_identity(self):
        backing = create_host_shared_buffer(128, mint_owner_instance_id(), 7)
        try:
            native = Tensor(backing, shapes=(8,), dtype=DataType.FLOAT32, byte_offset=16)
            mapped = native._with_mapping_ref(1)
            assert mapped.buffer == native.buffer
            assert mapped.mapping_ref == 1 and native.mapping_ref == 0
            with pytest.raises(ValueError, match="immutable"):
                mapped._with_mapping_ref(2)
            with pytest.raises(ValueError, match="nonzero"):
                native._with_mapping_ref(0)
            with pytest.raises(ValueError, match="transfer NONE"):
                TaskArgs().add_tensor(mapped, TensorArgType.INPUT)
        finally:
            backing.close()

    def test_mapping_requires_live_confirmed_attachment(self):
        backing = create_host_shared_buffer(64, mint_owner_instance_id(), 8)
        registry = ImportRegistry(self._context())
        live = [True]
        try:
            native = backing.tensor((16,), DataType.FLOAT32)
            mapped = native._with_mapping_ref(7)
            with pytest.raises(ValueError, match="no granted attachment"):
                registry.resolve_tensor(mapped)
            registry.register_attachment(
                backing.to_descriptor(),
                0x9000,
                mapping_ref=7,
                profile=AdapterProfile.HOST_SVM_MAP,
                live=lambda: live[0],
            )
            assert registry.resolve_tensor(mapped).base == 0x9000
            forged = replace(backing, nbytes=128).tensor((16,), DataType.FLOAT32)._with_mapping_ref(7)
            with pytest.raises(ValueError, match="descriptor differs"):
                registry.resolve_tensor(forged)
            live[0] = False
            with pytest.raises(ValueError, match="not live"):
                registry.resolve_tensor(mapped)
        finally:
            registry.close()
            backing.close()


def test_worker_refuses_missing_hostregister_without_fallback():
    class UnsupportedChip(FakeChip):
        def host_register_supported(self):
            return False

    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = UnsupportedChip()
    worker._lifecycle = _Lifecycle.READY
    try:
        with pytest.raises(RuntimeError, match="HostRegister is not supported"):
            worker.create_buffer(128, host_window=HostWindowSpec(128))
        assert not worker._direct_host_domains
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_release_refuses_inflight_host_use_then_retries():
    chip = FakeChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    try:
        buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
        worker._chip_run_touched_identities[1] = {buffer.identity}
        with pytest.raises(RuntimeError, match="in-flight L2 run"):
            worker.release_buffer(buffer)
        worker._chip_run_touched_identities.clear()
        worker.release_buffer(buffer)
        assert not worker._direct_host_domains
    finally:
        worker._chip_run_touched_identities.clear()
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_refuses_hostregister_on_a5sim_without_fallback():
    worker = Worker(level=2, platform="a5sim", runtime="host_build_graph", device_id=0)
    worker._lifecycle = _Lifecycle.READY
    try:
        with pytest.raises(ValueError, match="requires the a5 platform"):
            worker.create_buffer(128, host_window=HostWindowSpec(128))
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_a5_hostregister_rejects_cross_process_mapping():
    resource, _chip, _registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)
        resource.mapping._pid = -1
        with pytest.raises(RuntimeError, match="another process"):
            resource.map_view(native, 0, access=AccessMode.READ)
        resource.mapping._pid = os.getpid()
        resource.close()
    finally:
        resource.mapping._pid = os.getpid()
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_worker_l2_submit_accepts_mapped_host_argument():
    class SubmitChip(FakeChip):
        def __init__(self):
            super().__init__()
            self.submitted = []

        def _submit_chip_run_direct(self, callable_id, args, cfg):
            self.submitted.append(args)
            return SimpleNamespace()

    chip = SubmitChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    worker._callable_registry[3] = SimpleNamespace(
        sig_count=1,
        sig=lambda _index: ArgDirection.IN,
    )
    try:
        buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
        mapped = worker.map_view(buffer.tensor((32,), DataType.FLOAT32), access=AccessMode.READ)
        args = TaskArgs()
        args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
        worker._submit_l2_locked(3, args, CallConfig())
        assert len(chip.submitted) == 1
        assert chip.submitted[0].tensor(0).data == 0x8000
        worker._chip_runs.clear()
        worker._chip_run_touched_identities.clear()
        worker.unmap_view(mapped)
        worker.release_buffer(buffer)
    finally:
        worker._chip_run_touched_identities.clear()
        worker._chip_runs.clear()
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_refuses_numa_placement_for_hostregister():
    chip = FakeChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    try:
        with pytest.raises(ValueError, match="numa_id"):
            worker.create_buffer(128, host_window=HostWindowSpec(128, numa_id=1))
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_close_reclaims_mapped_host_domain():
    chip = FakeChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    try:
        buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
        mapped = worker.map_view(buffer.tensor((32,), DataType.FLOAT32), access=AccessMode.READ)
        base = buffer.base
        worker._release_all_direct_host_domains()
        assert not worker._direct_host_domains
        assert chip.unmapped == [int(base)]
        assert buffer.closed
        assert mapped.mapping_ref
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_refuses_borrowed_context_for_hostregister():
    class BorrowedChip(FakeChip):
        def __init__(self):
            super().__init__()
            self.environment = (os.getpid(), 1, 1, 0, True)
            self._execution_environment = self.environment

    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = BorrowedChip()
    worker._lifecycle = _Lifecycle.READY
    try:
        with pytest.raises(RuntimeError, match="borrowed context"):
            worker.create_buffer(128, host_window=HostWindowSpec(128))
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_worker_close_reclaims_mapped_host_domain_through_teardown():
    class CloseChip(FakeChip):
        def workspace_report(self):
            return ("disabled", None)

        def finalize(self):
            return None

    chip = CloseChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    buffer = worker.create_buffer(128, host_window=HostWindowSpec(128))
    mapped = worker.map_view(buffer.tensor((32,), DataType.FLOAT32), access=AccessMode.READ)
    try:
        worker.close()
        assert chip.unmapped == [int(buffer.base)]
        assert not worker._direct_host_domains
        assert buffer.closed
        assert mapped.mapping_ref
        assert not worker._cleanup_journal
    finally:
        worker._close_chip_import_registry()


def test_failed_second_map_does_not_invalidate_live_sibling(monkeypatch):
    resource, chip, registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)
        first = resource.map_view(native, 0, access=AccessMode.READ)
        original = registry.register_attachment

        def fail_second(*args, **kwargs):
            if kwargs.get("mapping_ref") == 2:
                raise ValueError("injected registration failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(registry, "register_attachment", fail_second)
        with pytest.raises(ValueError, match="injected registration failure"):
            resource.map_view(native, 0, access=AccessMode.READ)
        assert registry.resolve_tensor(first).base == 0x8000
        assert chip.unmapped == []
        resource.unmap_view(first)
        assert chip.unmapped == [int(_host_buffer(resource).base)]
        resource.close()
    finally:
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_host_buffer_direct_close_is_refused_even_when_unmapped():
    resource, _chip, _registry, _worker = _resource()
    try:
        with pytest.raises(RuntimeError, match="Worker.release_buffer"):
            _host_buffer(resource).close()
        resource.close()
    finally:
        if not resource.closed:
            resource.close()


def test_mapping_ref_is_bound_to_the_issued_view():
    resource, chip, _registry, _worker = _resource()
    try:
        buffer = _host_buffer(resource)
        native = buffer.tensor((32,), DataType.FLOAT32)
        mapped = resource.map_view(native, 0, access=AccessMode.READ)
        forged = buffer.tensor((16,), DataType.FLOAT32)._with_mapping_ref(mapped.mapping_ref)
        with pytest.raises(ValueError, match="live explicit mapping|not the view"):
            resource.validate(forged, 0)
        with pytest.raises(ValueError, match="not the view"):
            resource.unmap_view(forged)
        resource.unmap_view(mapped)
        assert chip.unmapped == [int(buffer.base)]
        resource.close()
    finally:
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()


def test_worker_exposes_urma_capability_without_fallback():
    chip = FakeChip()
    worker = Worker(level=2, platform="a5", runtime="tensormap_and_ringbuffer", device_id=0)
    worker._chip_worker = chip
    worker._lifecycle = _Lifecycle.READY
    try:
        assert worker.host_transport_capability() is False
        chip.host_urma_supported = lambda: True
        assert worker.host_transport_capability() is True
    finally:
        worker._close_chip_import_registry()
        worker._lifecycle = _Lifecycle.CLOSED


def test_failed_map_with_failed_cleanup_is_retried_by_close(monkeypatch):
    resource, chip, registry, _worker = _resource()
    try:
        native = _host_buffer(resource).tensor((32,), DataType.FLOAT32)

        def fail_registration(*_args, **_kwargs):
            raise ValueError("injected registration failure")

        monkeypatch.setattr(registry, "register_attachment", fail_registration)
        chip.unmap_error = RuntimeError("injected cleanup failure")
        with pytest.raises(ValueError, match="injected registration failure"):
            resource.map_view(native, 0, access=AccessMode.READ)
        assert resource._mapping_targets
        chip.unmap_error = None
        resource.close()
        assert not resource._mapping_targets
        assert chip.unmapped == [int(_host_buffer(resource).base)]
    finally:
        chip.unmap_error = None
        if not resource.closed:
            resource._mapping_targets.clear()
            resource.close()
