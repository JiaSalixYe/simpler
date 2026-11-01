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

from collections.abc import Callable
from dataclasses import dataclass

from .buffer import AccessMode, Buffer, Tensor
from .comm_domain import HostWindowSpec


class _BufferCloseGuard:
    """Refuse direct Buffer.close; the Worker owns Host backing release."""

    def __init__(self, resource):
        self.resource = resource

    def validate(self, identity, _descriptor) -> None:
        if identity != self.resource.identity:
            raise ValueError("Host Buffer descriptor identity differs from its resource")

    def close(self, _identity) -> None:
        raise RuntimeError("Host Buffer is owned by its Worker; use Worker.release_buffer")


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
        registry.register_attachment(
            self.descriptor,
            self.mapped_base,
            mapping_ref=reference,
            access=access,
            profile=self.mapping.profile,
            live=lambda: self.owner_live() and not self.closing and bool(self.mapped_base),
        )


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
        *,
        mapping,
        view_scope,
        view_control,
    ):
        self.spec = spec
        self.identity = identity
        self.workers = tuple(workers)
        self.mapping = mapping
        self.buffer: Buffer | None = None
        self._descriptor = None
        self.closed = False
        self._view_scope = view_scope
        self._view_control = view_control
        self._direct_view: HostDomainView | None = None
        self._next_reference = 1
        self._mapping_targets: dict[int, tuple[int, bool]] = {}
        self._mapping_views: dict[int, tuple] = {}
        self._revoked_refs: set[int] = set()

    @staticmethod
    def _view_key(tensor: Tensor) -> tuple:
        return (tuple(tensor.shapes), tuple(tensor.strides), int(tensor.dtype), int(tensor.byte_offset))

    def allocate(self):
        buffer = self.mapping.backend.allocate(self.spec, self.identity)
        self.buffer = buffer
        buffer._source_registration = (_BufferCloseGuard(self), self.identity)
        self._descriptor = buffer.to_descriptor()
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
                    self._mapping_views.pop(ref, None)
            if any(self._mapping_targets.get(ref) == (worker, True) for ref in self._revoked_refs):
                raise RuntimeError("Host mapping cleanup is unconfirmed")
            if self._next_reference > 255:
                raise RuntimeError("Host mapping references exhausted; allocate a new backing")
            reference = self._next_reference
            self._next_reference += 1
            mapped = tensor._with_mapping_ref(reference)
            self._mapping_targets[reference] = (worker, False)
            self._mapping_views[reference] = self._view_key(tensor)
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
                    self._mapping_views.pop(reference, None)
                raise

    def unmap_view(self, tensor: Tensor):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        if self._mapping_views.get(reference) != self._view_key(tensor):
            raise ValueError("Host Tensor is not the view that was mapped")
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
            self._mapping_views.pop(reference, None)

    def validate(self, tensor, worker):
        self._check_source(tensor)
        reference = tensor.mapping_ref
        if not reference:
            raise ValueError("A5 Host arguments require an explicit mapped view")
        if (
            worker not in self.workers
            or self._mapping_views.get(reference) != self._view_key(tensor)
            or reference in self._revoked_refs
            or self._mapping_targets.get(reference) != (worker, True)
        ):
            raise ValueError("Host Tensor requires a live explicit mapping for the target endpoint")

    def _retry_unconfirmed_views(self) -> None:
        for reference, (worker, acknowledged) in tuple(self._mapping_targets.items()):
            if acknowledged:
                continue
            self._view_control(self, worker, reference, 0, unmap=True)
            self._mapping_targets.pop(reference, None)
            self._mapping_views.pop(reference, None)

    def revoke_all(self) -> None:
        for reference, (worker, _acknowledged) in tuple(self._mapping_targets.items()):
            self._view_control(self, worker, reference, 0, unmap=True)
            self._mapping_targets.pop(reference, None)
            self._mapping_views.pop(reference, None)

    def close(self, *, force: bool = False):
        if self.closed:
            return
        if force:
            self.revoke_all()
        else:
            self._retry_unconfirmed_views()
            if self._mapping_targets:
                raise RuntimeError("Host domain owner retained: unmap every mapped view first")
        if self.buffer is not None:
            self.buffer.closed = True
            self.buffer._source_registration = None
        self.mapping.backend.release()
        self.closed = True
