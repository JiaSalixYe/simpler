# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Package-internal Host ownership and explicit per-chip mapping implementation.

The supported user surface is Buffer/Tensor plus Worker/CommDomainHandle methods;
this module's resource and grant-registration helpers are not public APIs.
"""

from __future__ import annotations

import ctypes
import struct
from collections.abc import Callable
from dataclasses import dataclass

from _task_interface import _decode_buffer_descriptor_wire, _encode_buffer_descriptor_wire

from .buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    BufferDescriptor,
    CanonicalIdentity,
    Tensor,
)
from .comm_domain import HostWindowSpec
from .global_comm_domain import GlobalDomainDescriptor
from .host_domain_backend import host_attachment_profile, host_backing_type


@dataclass
class HostDomainView:
    """Package-internal physical import; grants belong only to ImportRegistry."""

    descriptor: BufferDescriptor
    exported: GlobalDomainDescriptor
    base: int
    execution_environment: tuple
    owner_live: Callable[[], bool]
    closing: bool = False
    attach: Callable | None = None
    detach: Callable | None = None

    def apply(self, cw, registry, comm, allocation, reference, access, *, unmap, platform):
        cw._validate_execution_context()
        if cw._execution_environment != self.execution_environment:
            self.closing = True
            raise ValueError("Host mapping target context changed")
        if unmap:
            registry.close_attachment(self.descriptor.identity, reference)
            if registry.has_active_attachments(self.descriptor.identity):
                return
            self.closing = True
            if self.detach is None:
                cw.comm_detach_domain_host(comm, allocation)
            else:
                self.detach()
            self.base = 0
            self.closing = False
            return
        if self.closing:
            raise RuntimeError("Host mapping cleanup is unconfirmed")
        if not self.base:
            self.base = (
                cw.comm_attach_domain_host(comm, allocation, self.exported.encode())
                if self.attach is None
                else self.attach()
            )
        try:
            registry.register_attachment(
                self.descriptor,
                self.base,
                mapping_ref=reference,
                access=access,
                profile=host_attachment_profile(self.exported, platform),
                live=lambda: self.owner_live() and not self.closing and bool(self.base),
            )
        except BaseException:
            # A failed registration can follow a committed native import.
            self.closing = True
            raise


class HostDomainResource:
    """Owner retained until all member views are confirmed gone.

    Issued references never repeat within the backing lifetime. Target routes are
    not grants: only the target registry authorizes a selected access.
    """

    def __init__(
        self,
        spec: HostWindowSpec,
        identity,
        workers,
        platform,
        device,
        *,
        view_scope,
        view_control,
        host_runtime_path=None,
        buffer=None,
        reference_namespace=(1, 1),
    ):
        self.spec = spec
        self.identity = identity
        self.workers = tuple(workers)
        self.device = device
        self.backend = host_backing_type(platform)(host_runtime_path=host_runtime_path) if buffer is None else None
        self.buffer = buffer
        self._descriptor = None if buffer is None else buffer.to_descriptor()
        self.export = None
        self.published = False
        self.views_released = False
        self.closed = False
        self._view_scope = view_scope
        self._view_control = view_control
        self._direct_view = None
        reference_start, reference_step = reference_namespace
        if not 1 <= reference_start <= 255 or not 1 <= reference_step <= 64:
            raise ValueError("invalid Host mapping issuer namespace")
        self._next_reference = reference_start
        self._reference_step = reference_step
        self._mapping_targets = {}
        self._revoked_refs = set()

    def allocate(self):
        if self.backend is None:
            raise ValueError("a borrowed Host source cannot allocate its owner")
        base, self.export = self.backend.allocate(self.spec, self.identity, self.device, len(self.workers))
        self.buffer = Buffer(
            identity=self.identity,
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.REMOTE_SIDECAR,
            nbytes=self.spec.nbytes,
            base=base,
        )
        self._descriptor = self.buffer.to_descriptor()

    def request(self) -> bytes:
        if self.closed or self.buffer is None or self.buffer.closed or self.export is None:
            raise RuntimeError("Host backing is not live")
        return self.export.encode() + _encode_buffer_descriptor_wire(self._descriptor)

    def _check_source(self, tensor):
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host mapping requires the live domain backing")
        if tensor.buffer != self._descriptor:
            raise ValueError("Host Tensor differs from its source registration")

    def map_view(self, tensor: Tensor, worker: int, *, access=AccessMode.READWRITE) -> Tensor:
        with self._view_scope(self, worker, "map_view"):
            self._check_source(tensor)
            if tensor.mapping_ref:
                raise ValueError("map_view requires the native Tensor view")
            access = AccessMode(access)
            if worker not in self.workers:
                raise ValueError("Host mapping requires a domain member")
            # An unacknowledged map has no caller-visible Tensor. Retry its
            # cleanup through this public operation before issuing another ref.
            for ref, route in tuple(self._mapping_targets.items()):
                if route == (worker, False):
                    self._view_control(self, worker, ref, 0, unmap=True)
                    self._mapping_targets.pop(ref)
            if any(self._mapping_targets.get(ref) == (worker, True) for ref in self._revoked_refs):
                raise RuntimeError("Host mapping cleanup is unconfirmed")
            if self._next_reference > 255:
                raise RuntimeError("Host mapping references exhausted; allocate a new backing")
            reference = self._next_reference
            self._next_reference += self._reference_step
            mapped = tensor._with_mapping_ref(reference)
            self._mapping_targets[reference] = (worker, False)
            self._revoked_refs.add(reference)
            # Publish no authority until the target confirms its restricted record.
            try:
                self._view_control(self, worker, reference, access, unmap=False)
                self._revoked_refs.remove(reference)
                self._mapping_targets[reference] = (worker, True)
                return mapped
            except BaseException:
                self._revoked_refs.add(reference)
                self._mapping_targets[reference] = (worker, False)
                raise

    def unmap_view(self, tensor: Tensor):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        route = self._mapping_targets.get(reference)
        if route is None:
            if reference in self._revoked_refs:
                return
            raise ValueError("Host Tensor has no issued mapping")
        worker, _acknowledged = route
        with self._view_scope(self, worker, "unmap_view"):
            self._check_source(tensor)
            self._revoked_refs.add(reference)
            self._view_control(self, worker, reference, 0, unmap=True)
            self._mapping_targets.pop(reference, None)

    def validate(self, tensor, worker, *, native=False):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        if worker not in self.workers or (
            not (native and reference == 0)
            and (reference in self._revoked_refs or self._mapping_targets.get(reference) != (worker, True))
        ):
            raise ValueError("Host Tensor requires a live explicit mapping for the target worker")

    def copy(self, offset, value, *, writing):
        size = len(value) if writing else value
        if offset < 0 or size < 0 or offset > self.spec.nbytes or size > self.spec.nbytes - offset:
            raise ValueError("Host domain copy exceeds the backing range")
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host domain backing is not live")
        base = self.buffer.base
        if not base:
            raise ValueError("Host copy requires the owning process's native view")
        if writing:
            ctypes.memmove(base + offset, value, size)
            return None
        return ctypes.string_at(base + offset, size)

    def close(self):
        self.closed = True
        if self.buffer is not None:
            self.buffer.closed = True
        if self.published and not self.views_released:
            raise RuntimeError("Host domain owner retained: member view release is unconfirmed")
        if self.backend is not None:
            self.backend.release()


_GLOBAL_HOST_VIEW = struct.Struct("<4sQQQBBBB")
_GLOBAL_HOST_MAGIC = b"HGV1"


def _global_host_buffer(owner_instance_id, window):
    return Buffer(
        identity=CanonicalIdentity(owner_instance_id, window.buffer_id, 1),
        address_space=AddressSpace.HOST,
        access=AccessMode.READWRITE,
        backend_kind=BackendKind.REMOTE_SIDECAR,
        nbytes=window.spec.nbytes,
        base=0,
    )


def _global_host_request(domain, generation, buffer_id, reference=0, access=0, operation=0, resource=None):
    payload = _GLOBAL_HOST_VIEW.pack(
        _GLOBAL_HOST_MAGIC, domain, generation, buffer_id, reference, int(access), operation, 0
    )
    if resource is not None:
        payload += resource.export.encode() + _encode_buffer_descriptor_wire(resource.buffer.to_descriptor())
    return payload


def _apply_global_host_request(cw, registry, store, sources, payload, *, platform):  # noqa: PLR0912 -- validate, bind, map and revoke share one private control boundary
    if len(payload) < _GLOBAL_HOST_VIEW.size:
        raise ValueError("truncated global Host view request")
    magic, domain, generation, buffer_id, reference, access, operation, reserved = _GLOBAL_HOST_VIEW.unpack_from(
        payload
    )
    if magic != _GLOBAL_HOST_MAGIC or reserved or operation not in (0, 1, 2, 3):
        raise ValueError("invalid global Host view request")
    cw._validate_execution_context()
    entry = store.domains.get(domain)
    if (
        entry is None
        and operation == 3
        and len(payload) == _GLOBAL_HOST_VIEW.size
        and not (reference or access or buffer_id)
    ):
        return
    if entry is None or entry.generation != generation:
        raise ValueError("global Host view requires the matching domain generation")
    if operation == 3:
        if len(payload) != _GLOBAL_HOST_VIEW.size or reference or access or buffer_id:
            raise ValueError("invalid global Host revoke request")
        entry.host_closing = True
        for key, source in tuple(sources.items()):
            if not isinstance(key, tuple) or key[0] != domain:
                continue
            source.closing = True
            registry.unregister(source.descriptor.identity)
            if source.base:
                source.detach()
                source.base = 0
            sources.pop(key)
        return
    if entry.host_closing:
        raise ValueError("global Host imports are closing")
    key = (domain, buffer_id)
    if operation == 0:
        from .global_comm_domain import GLOBAL_DOMAIN_DESCRIPTOR_BYTES  # noqa: PLC0415

        tail = bytes(payload[_GLOBAL_HOST_VIEW.size :])
        if len(tail) <= GLOBAL_DOMAIN_DESCRIPTOR_BYTES or reference or access or not buffer_id:
            raise ValueError("invalid global Host binding size")
        exported = GlobalDomainDescriptor.decode(tail[:GLOBAL_DOMAIN_DESCRIPTOR_BYTES])
        descriptor = _decode_buffer_descriptor_wire(tail[GLOBAL_DOMAIN_DESCRIPTOR_BYTES:])
        if (
            descriptor.address_space != AddressSpace.HOST
            or descriptor.backend_kind != BackendKind.REMOTE_SIDECAR
            or descriptor.identity.buffer_id != buffer_id
            or descriptor.nbytes > exported.mapping_size
        ):
            raise ValueError("invalid global Host backing descriptor")
        host_attachment_profile(exported, platform)
        previous = sources.get(key)
        if previous is not None:
            if previous.descriptor != descriptor or previous.exported != exported:
                raise ValueError("global Host binding conflicts with its source")
            return
        sources[key] = HostDomainView(
            descriptor,
            exported,
            0,
            cw._execution_environment,
            lambda: store.domains.get(domain) is entry and not entry.host_closing,
            attach=lambda: cw.comm_global_domain_import_host(
                domain, exported.profile_id, exported.handle, exported.mapping_size
            ),
            detach=lambda: cw.comm_global_domain_detach_host(domain, exported.handle),
        )
        return
    if len(payload) != _GLOBAL_HOST_VIEW.size or not reference or (operation == 2 and access):
        raise ValueError("invalid global Host mapping operation")
    source = sources.get(key)
    if source is None or not entry.device_ctx:
        raise ValueError("global Host view requires a bound imported backing")
    if operation == 1:
        access = AccessMode(access)
    source.apply(cw, registry, 0, domain, reference, access, unmap=operation == 2, platform=platform)
