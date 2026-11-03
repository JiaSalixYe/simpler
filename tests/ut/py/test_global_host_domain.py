# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler.comm_domain import HostWindowSpec
from simpler.global_comm_domain import (
    GLOBAL_DOMAIN_VERSION,
    GlobalDomainHostWindow,
    GlobalDomainPhase,
    GlobalDomainReleaseCommand,
    HostExportDescriptor,
    decode_domain_command,
    encode_domain_command,
)
from simpler.task_interface import CommDomainHandle
from simpler.worker import Worker, _GlobalNodeDomainState

from tests.ut.py.test_global_comm_domain import _host_command


@pytest.fixture
def host_authorization():
    template = _host_command(GlobalDomainPhase.IMPORT)
    descriptors, windows = template.descriptors, template.host_windows
    command = replace(template, phase=GlobalDomainPhase.AUTHORIZE, descriptors=(), target_pids=(101, 202))
    prepared = replace(
        command,
        phase=GlobalDomainPhase.PREPARE_EXPORT,
        descriptors=(),
        target_pids=(),
        host_windows=tuple(replace(w, export=None) for w in windows),
    )
    exported = windows[0].export
    assert exported is not None
    resource = SimpleNamespace(
        backend=SimpleNamespace(authorize=Mock()),
        export=replace(descriptors[0], handle=exported.handle),
        buffer=None,
        closed=False,
        close=Mock(),
    )
    state = _GlobalNodeDomainState(
        command=prepared,
        node_worker_id=0,
        host_resources={0: resource},
        descriptors={descriptor.domain_rank: descriptor for descriptor in descriptors},
    )
    worker = SimpleNamespace(
        level=3,
        _worker=Mock(),
        _global_node_domains={command.domain_id: state},
        _global_domain_command_identity=Worker._global_domain_command_identity,
        _require_global_host_authorization=Worker._require_global_host_authorization,
        _child_prov_lock=threading.RLock(),
        _drop_domain_allocs=Mock(),
        _global_domain_provenance_id=Worker._global_domain_provenance_id,
        _cleanup_journal=Mock(),
        _py_control_timeout_s=1,
    )
    return worker, state, command, resource


def test_authorization_replay_is_idempotent(host_authorization):
    worker, _state, command, resource = host_authorization
    Worker._authorize_global_host_node(worker, command)
    Worker._authorize_global_host_node(worker, command)
    resource.backend.authorize.assert_called_once_with(command.target_pids)


def test_authorization_rejects_substitution_of_the_local_owner_export(host_authorization):
    worker, _state, command, resource = host_authorization
    wrong = replace(command.host_windows[0], export=HostExportDescriptor(1, 4096, b"/different-owner"))
    with pytest.raises(ValueError, match="export"):
        Worker._authorize_global_host_node(worker, replace(command, host_windows=(wrong, command.host_windows[1])))
    resource.backend.authorize.assert_not_called()


@pytest.mark.parametrize("change", ["pids", "export"])
def test_authorization_replay_cannot_change_its_authority(host_authorization, change):
    worker, _state, command, resource = host_authorization
    Worker._authorize_global_host_node(worker, command)
    if change == "pids":
        changed = replace(command, target_pids=(101, 303))
    else:
        remote = replace(command.host_windows[1], export=HostExportDescriptor(1, 4096, b"/changed-remote"))
        changed = replace(command, host_windows=(command.host_windows[0], remote))
    with pytest.raises(ValueError, match="authorization"):
        Worker._authorize_global_host_node(worker, decode_domain_command(encode_domain_command(changed)))
    resource.backend.authorize.assert_called_once()


@pytest.mark.parametrize("operation", ["import", "commit"])
def test_import_commit_reject_export_substitution_before_native_work(host_authorization, operation):
    worker, state, command, _resource = host_authorization
    Worker._authorize_global_host_node(worker, command)
    state.command = replace(
        command, phase=GlobalDomainPhase.IMPORT, target_pids=(), descriptors=tuple(state.descriptors.values())
    )
    state.phase = GlobalDomainPhase.IMPORT
    state.view = SimpleNamespace(_committed=False)
    wrong = replace(command.host_windows[1], export=HostExportDescriptor(1, 4096, b"/replaced-after-auth"))
    incoming = replace(state.command, host_windows=(command.host_windows[0], wrong))
    with pytest.raises(ValueError, match="authorization"):
        if operation == "import":
            Worker._import_global_domain_node(worker, incoming, 0)
        else:
            Worker._commit_global_domain_node(worker, replace(incoming, phase=GlobalDomainPhase.COMMIT))
    assert not state.view._committed
    worker._worker.control_payload.assert_not_called()


def test_failed_authorization_attempt_cannot_replace_a_pid(host_authorization):
    worker, _state, command, resource = host_authorization
    resource.backend.authorize.side_effect = RuntimeError("unconfirmed whitelist update")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        Worker._authorize_global_host_node(worker, command)
    with pytest.raises(ValueError, match="authorization"):
        changed = replace(command, target_pids=(101, 303))
        Worker._authorize_global_host_node(worker, decode_domain_command(encode_domain_command(changed)))
    resource.backend.authorize.assert_called_once()


def test_owner_allocation_failure_still_releases_prepared_chip_views(host_authorization):
    worker, state, command, resource = host_authorization
    state.host_resources.clear()
    state.prepared_domain_ranks = {0}
    resource.allocate = Mock(side_effect=RuntimeError("Host allocation failed"))
    worker._global_host_resource = Mock(return_value=resource)
    with pytest.raises(RuntimeError, match="allocation failed"):
        Worker._prepare_global_host_owner(worker, state, 0)
    assert resource.buffer is None
    Worker._release_global_domain_node(worker, GlobalDomainReleaseCommand(command.domain_id, command.generation))
    assert state.native_released and resource.closed
    worker._worker.control_payload.assert_called_once()
    worker._cleanup_journal.add_once.assert_called_once()


def test_abort_fanout_encodes_host_phases_without_rank_metadata(host_authorization):
    worker, state, command, _resource = host_authorization
    command = replace(command, descriptors=tuple(state.descriptors.values()))
    worker._global_domain_control = Mock()
    worker._global_host_phase = lambda cmd, nodes, phase: Worker._global_host_phase(worker, cmd, nodes, phase)
    Worker._abort_global_host_domain(worker, command, (0, 1))
    replies = [decode_domain_command(call.args[2]) for call in worker._global_domain_control.call_args_list]
    assert [reply.phase for reply in replies] == [
        GlobalDomainPhase.ABORT,
        GlobalDomainPhase.ABORT,
        GlobalDomainPhase.RELEASE_HOST,
        GlobalDomainPhase.RELEASE_HOST,
    ]
    assert all(not reply.descriptors and not reply.attachments and not reply.target_pids for reply in replies)
    assert all(reply.host_windows == command.host_windows for reply in replies)


def test_close_journal_retains_owner_until_views_and_domain_release_confirmed(host_authorization):
    from simpler.host_domain import HostDomainResource  # noqa: PLC0415
    from simpler.worker import CleanupJournal  # noqa: PLC0415

    worker, state, command, resource = host_authorization
    resource.published, resource.views_released = True, False
    resource.backend.release = Mock()
    resource.close = lambda: HostDomainResource.close(resource)
    worker._cleanup_journal = CleanupJournal()
    worker._cleanup_journal.add_once("host-domain", f"global-{command.domain_id}", resource.close)
    assert isinstance(worker._cleanup_journal.drive_kinds({"host-domain"}), RuntimeError)
    resource.backend.release.assert_not_called()
    assert not worker._cleanup_journal.empty
    with pytest.raises(RuntimeError, match="confirmed"):
        Worker._release_global_host_node(worker, command)
    resource.backend.release.assert_not_called()
    state.native_released = True
    Worker._release_global_host_node(worker, command)
    resource.backend.release.assert_called_once()
    assert worker._cleanup_journal.empty and command.domain_id not in worker._global_node_domains


def test_receivers_cannot_reinterpret_each_others_mapping_reference():
    from contextlib import nullcontext  # noqa: PLC0415

    from simpler.buffer import AccessMode, DataType  # noqa: PLC0415
    from simpler.host_domain import HostDomainResource, _global_host_buffer  # noqa: PLC0415

    spec = HostWindowSpec(128)
    source = _global_host_buffer(b"owner123", GlobalDomainHostWindow(0, spec, 1))
    resources = [
        HostDomainResource(
            spec,
            source.identity,
            (0,),
            "a2a3sim",
            0,
            buffer=source,
            reference_namespace=(node + 1, 2),
            view_scope=lambda *_args: nullcontext(),
            view_control=Mock(),
        )
        for node in range(2)
    ]
    native = source.tensor((16,), DataType.FLOAT32)
    views = [r.map_view(native, 0, access=AccessMode.READ) for r in resources]
    assert views[0].mapping_ref != views[1].mapping_ref
    for receiver, resource in enumerate(resources):
        resource.validate(views[receiver], 0)
        with pytest.raises(ValueError, match="live explicit mapping"):
            resource.validate(views[1 - receiver], 0)


def test_first_map_binds_source_and_failed_release_revokes_it(host_authorization, monkeypatch):
    from simpler import worker as worker_module  # noqa: PLC0415
    from simpler.buffer import AccessMode  # noqa: PLC0415
    from simpler.global_comm_domain import LOCAL_DOMAIN_MAGIC, LOCAL_RELEASE_REQUEST  # noqa: PLC0415
    from simpler.host_domain import (  # noqa: PLC0415
        HostDomainResource,
        _apply_global_host_request,
        _global_host_buffer,
    )

    worker, state, command, resource = host_authorization
    state.phase = GlobalDomainPhase.COMMIT
    resource.buffer = _global_host_buffer(command.owner_instance_id, command.host_windows[0])
    resource._descriptor = resource.buffer.to_descriptor()
    resource.identity = resource.buffer.identity
    resource.request = lambda: HostDomainResource.request(resource)
    cw, registry, sources = Mock(), Mock(), {}
    cw.comm_global_domain_import_host.return_value = 8192
    entry = worker_module._L2GlobalDomain(
        command.domain_id, 1, 0, 2, state.descriptors[0], 8192, 4096, 4096, device_ctx=1
    )
    store = worker_module._L2GlobalDomainStore({command.domain_id: entry})

    def control(_kind, _target, _operation, payload, _timeout):
        _apply_global_host_request(cw, registry, store, sources, payload, platform="a2a3sim")

    worker._worker.control_payload.side_effect = control

    def apply(reference, *, unmap=False):
        Worker._control_global_host_view(
            worker, state, resource, 0, reference, 0 if unmap else AccessMode.READ, unmap=unmap
        )

    apply(1, unmap=True)  # A failed first request may never have created a source.
    cw.comm_global_domain_detach_host.assert_not_called()
    apply(2)
    cw.comm_global_domain_import_host.assert_called_once()
    assert registry.register_attachment.call_args.kwargs["access"] == AccessMode.READ
    resource.export = replace(resource.export, handle=b"/substituted")
    with pytest.raises(ValueError, match="conflicts"):
        apply(3)
    assert registry.register_attachment.call_count == 1
    live = registry.register_attachment.call_args.kwargs["live"]
    assert live()
    request = LOCAL_RELEASE_REQUEST.pack(LOCAL_DOMAIN_MAGIC, GLOBAL_DOMAIN_VERSION, command.domain_id, 1)
    monkeypatch.setattr(
        worker_module, "_open_global_domain_payload", lambda _: (Mock(), memoryview(request), len(request))
    )
    cw.comm_global_domain_release.side_effect = RuntimeError("unconfirmed Host import cleanup")
    with pytest.raises(RuntimeError, match="unconfirmed"):
        worker_module._handle_ctrl_global_domain_release(
            cw, memoryview(request), store, import_registry=registry, host_sources=sources
        )
    assert not live() and entry.host_closing and store.domains[command.domain_id] is entry
    registry.unregister.assert_called_once_with(resource.buffer.identity)
    assert not sources


def test_missing_node_release_ack_does_not_start_owner_release(host_authorization):
    worker, state, command, _resource = host_authorization
    worker._mpi_group_for_involved_nodes = lambda _nodes: None
    worker._global_domain_control = Mock(side_effect=[b"", RuntimeError("node 1 has not acknowledged")])
    worker._global_host_phase = Mock()
    worker._live_global_domains = {}
    handle = CommDomainHandle(
        name=command.name,
        domain_id=command.domain_id,
        generation=command.generation,
        members=command.members,
        _release_fn=None,
    )
    handle._host_command = command
    with pytest.raises(RuntimeError, match="not acknowledged"):
        Worker._release_global_domain_claimed(worker, handle)
    worker._global_host_phase.assert_not_called()
