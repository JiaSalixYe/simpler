# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Package-internal Host ownership and explicit per-chip mapping implementation.

The supported user surface is Buffer/Tensor plus Worker methods; this module's
resource and grant-registration helpers are not public APIs.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass

from .buffer import AccessMode, Buffer, Tensor
from .comm_domain import HostWindowSpec


class _BufferCloseGuard:
    """Refuse direct Buffer.close/unlink while a device mapping still borrows it."""

    def __init__(self, resource):
        self.resource = resource

    def validate(self, identity, _descriptor) -> None:
        if identity != self.resource.identity:
            raise ValueError("Host Buffer descriptor identity differs from its resource")

    def close(self, _identity) -> None:
        if self.resource._mapping_targets:
            raise RuntimeError("Host Buffer is mapped; unmap every view before closing it")


@dataclass
class HostDomainView:
    """Package-internal physical import; grants belong only to ImportRegistry."""

    descriptor: object
    host_base: int
    mapped_base: int
    execution_environment: tuple | None
    owner_live: Callable[[], bool]
    mapping: object
    closing: bool = False

    def apply(self, cw, registry, reference, access, *, unmap):
        cw._validate_execution_context()
        if cw._execution_environment != self.execution_environment:
            self.closing = True
            raise ValueError("Host mapping target context changed")
        if unmap:
            registry.close_attachment(self.descriptor.identity, reference)
            if registry.has_active_attachments(self.descriptor.identity):
                return
            self.closing = True
            if self.mapped_base:
                self.mapping.detach(cw, self.host_base)
            self.mapped_base = 0
            self.closing = False
            return
        if self.closing:
            raise RuntimeError("Host mapping cleanup is unconfirmed")
        if not self.mapped_base:
            self.mapped_base = self.mapping.attach(cw, self.host_base, self.descriptor.nbytes)
        try:
            registry.register_attachment(
                self.descriptor,
                self.mapped_base,
                mapping_ref=reference,
                access=access,
                profile=self.mapping.profile,
                live=lambda: self.owner_live() and not self.closing and bool(self.mapped_base),
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
        mapping,
        view_scope,
        view_control,
    ):
        self.spec = spec
        self.identity = identity
        self.workers = tuple(workers)
        self.device = device
        self.platform = platform
        self.mapping = mapping
        self.buffer: Buffer | None = None
        self._descriptor = None
        self.published = False
        self.views_released = False
        self.closed = False
        self._view_scope = view_scope
        self._view_control = view_control
        self._direct_view: HostDomainView | None = None
        self._next_reference = 1
        self._mapping_targets: dict[int, tuple[int, bool]] = {}
        self._revoked_refs: set[int] = set()

    def allocate(self):
        buffer = self.mapping.backend.allocate(self.spec, self.identity, self.device, len(self.workers))
        self.buffer = buffer
        buffer._source_registration = (_BufferCloseGuard(self), self.identity)
        self._descriptor = buffer.to_descriptor()
        self.views_released = not self._mapping_targets
        self._direct_view = HostDomainView(
            self._descriptor,
            int(buffer.base),
            0,
            self.mapping.execution_environment,
            lambda: not self.closed and not (self.buffer is not None and self.buffer.closed),
            self.mapping,
        )

    def _check_source(self, tensor):
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host mapping requires the live Host backing")
        if tensor.buffer != self._descriptor:
            raise ValueError("Host Tensor differs from its source registration")

    def map_view(self, tensor: Tensor, worker: int, *, access=AccessMode.READWRITE) -> Tensor:
        with self._view_scope(self, worker, "map_view"):
            self._check_source(tensor)
            if tensor.mapping_ref:
                raise ValueError("map_view requires the native Tensor view")
            access = AccessMode(access)
            if worker not in self.workers:
                raise ValueError("Host mapping requires a live endpoint")
            for ref, route in tuple(self._mapping_targets.items()):
                if route == (worker, False):
                    self._view_control(self, worker, ref, 0, unmap=True)
                    self._mapping_targets.pop(ref)
            if any(self._mapping_targets.get(ref) == (worker, True) for ref in self._revoked_refs):
                raise RuntimeError("Host mapping cleanup is unconfirmed")
            if self._next_reference > 255:
                raise RuntimeError("Host mapping references exhausted; allocate a new backing")
            reference = self._next_reference
            self._next_reference += 1
            mapped = tensor._with_mapping_ref(reference)
            self._mapping_targets[reference] = (worker, False)
            self._revoked_refs.add(reference)
            try:
                self._view_control(self, worker, reference, access, unmap=False)
                self._revoked_refs.remove(reference)
                self._mapping_targets[reference] = (worker, True)
                return mapped
            except BaseException:
                try:
                    self._view_control(self, worker, reference, 0, unmap=True)
                except BaseException:
                    self._revoked_refs.add(reference)
                    self._mapping_targets[reference] = (worker, False)
                else:
                    self._mapping_targets.pop(reference, None)
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
            if not self._mapping_targets:
                self.views_released = True

    def validate(self, tensor, worker):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        if not reference:
            raise ValueError("A5 Host arguments require an explicit mapped view")
        if worker not in self.workers or (
            reference in self._revoked_refs or self._mapping_targets.get(reference) != (worker, True)
        ):
            raise ValueError("Host Tensor requires a live explicit mapping for the target endpoint")

    def copy(self, offset, value, *, writing):
        size = len(value) if writing else value
        if offset < 0 or size < 0 or offset > self.spec.nbytes or size > self.spec.nbytes - offset:
            raise ValueError("Host copy exceeds the backing range")
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host domain backing is not live")
        base = self.buffer.base
        if writing:
            ctypes.memmove(base + offset, value, size)
            return None
        return ctypes.string_at(base + offset, size)

    def close(self):
        if self.closed:
            return
        if self._mapping_targets:
            raise RuntimeError("Host domain owner retained: unmap every mapped view first")
        if self.buffer is not None:
            self.buffer.closed = True
        if self.published and not self.views_released:
            raise RuntimeError("Host domain owner retained: view release is unconfirmed")
        self.mapping.backend.release()
        self.closed = True
