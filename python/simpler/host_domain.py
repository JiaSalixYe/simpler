# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Package-internal Host ownership and explicit per-chip mapping implementation.

The supported user surface is Buffer/Tensor plus Worker methods;
this module's resource and grant-registration helpers are not public APIs.
"""

from __future__ import annotations

from _task_interface import _encode_buffer_descriptor_wire

from .buffer import (
    AccessMode,
    AddressSpace,
    BackendKind,
    Buffer,
    Tensor,
)
from .host_domain_backend import host_backing_type


class MappingResource:
    """Routing for issued references; the target ImportRegistry owns grants."""

    def __init__(self, buffer, workers, view_scope, view_control, *, identity=None):
        self.buffer = buffer
        self.identity = buffer.identity if buffer is not None else identity
        self._descriptor = buffer.to_descriptor() if buffer is not None else None
        self.workers = tuple(workers)
        self.closed = False
        self._view_scope = view_scope
        self._view_control = view_control
        self._next_reference = 1
        self._mapping_targets = {}
        self._revoked_refs = set()

    def _check_source(self, tensor):
        if self.closed or self.buffer is None or self.buffer.closed:
            raise ValueError("Host mapping requires the live source backing")
        if tensor.buffer != self._descriptor:
            raise ValueError("Host Tensor differs from its source registration")

    def map_view(self, tensor: Tensor, worker: int, *, access=AccessMode.READWRITE) -> Tensor:
        with self._view_scope(self, worker, "map_view"):
            self._check_source(tensor)
            if tensor.mapping_ref:
                raise ValueError("map_view requires the native Tensor view")
            access = AccessMode(access)
            if worker not in self.workers:
                raise ValueError("Host mapping requires a local device endpoint")
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
            self._next_reference += 1
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


class HostDomainResource(MappingResource):
    """Worker-owned backing retained until endpoint mappings retire.

    Issued references never repeat within the backing lifetime. Target routes are
    not grants: only the target registry authorizes a selected access.
    """

    def __init__(
        self,
        nbytes: int,
        identity,
        workers,
        platform,
        device,
        *,
        view_scope,
        view_control,
        host_runtime_path=None,
    ):
        super().__init__(None, workers, view_scope, view_control, identity=identity)
        self.nbytes = nbytes
        self.device = device
        self.backend = host_backing_type(platform)(host_runtime_path=host_runtime_path)
        self.export = None
        self._cpu_targets = set()

    def allocate(self):
        base, self.export = self.backend.allocate(self.nbytes, self.identity, self.device, max(1, len(self.workers)))
        self.buffer = Buffer(
            identity=self.identity,
            address_space=AddressSpace.HOST,
            access=AccessMode.READWRITE,
            backend_kind=BackendKind.REMOTE_SIDECAR,
            nbytes=self.nbytes,
            base=base,
        )
        self.buffer._host_backing = self.backend
        self._descriptor = self.buffer.to_descriptor()

    def request(self) -> bytes:
        if self.closed or self.buffer is None or self.buffer.closed or self.export is None:
            raise RuntimeError("Host backing is not live")
        return self.export.encode() + _encode_buffer_descriptor_wire(self._descriptor)

    def close(self):
        if self._mapping_targets or self._cpu_targets:
            raise RuntimeError("Host owner retained: mappings remain")
        self.closed = True
        if self.buffer is not None:
            self.buffer.closed = True
        self.backend.release()
