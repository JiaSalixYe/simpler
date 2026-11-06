# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from simpler import worker as worker_module
from simpler.comm_domain import CommDomainRef
from simpler.orchestrator import Orchestrator, _callback_run
from simpler.task_interface import CallConfig, CcuFunction, ChipCallable, CommDomainHandle, TaskArgs
from simpler.worker import Worker


@pytest.fixture
def setup(monkeypatch):
    worker = Worker(level=3, platform="a5", runtime="tensormap_and_ringbuffer", device_ids=[0, 1])
    callable_ = ChipCallable.build(
        signature=[],
        func_name="orchestrate",
        binary=b"orch",
        children=[],
        ccu_functions=[CcuFunction.build(binary=b"module", entry="build_copy")],
    )
    handle = worker.register(callable_)
    ref = CommDomainRef(worker._owner_instance_id, 0, 1)
    domain = CommDomainHandle(name="comm", workers=(1, 0), allocation_id=0, _release_fn=None, ref=ref)
    domain._config = SimpleNamespace(workspace_engines=("ccu",))
    control = Mock()

    def get_domain(reference):
        if reference != domain.ref:
            raise KeyError("communication domain reference has no live grant on this node")
        return domain

    monkeypatch.setattr(worker, "_get_domain", get_domain)
    monkeypatch.setattr(worker, "_control_admission", lambda _: nullcontext())
    monkeypatch.setattr(worker, "_worker", control)
    monkeypatch.setattr(worker, "_require_local_next_level_target", lambda *_, **__: None)
    with _callback_run(1, worker):
        yield Orchestrator(Mock(), worker), control, handle, domain


def _prepare_reply(worker_id, command, request):
    allocation_id, binding, digest = worker_module._CCU_CONTROL_HEADER.unpack_from(request)
    assert binding == 0
    if command == worker_module._CTRL_CCU_PREPARE:
        return worker_module._CCU_CONTROL_HEADER.pack(allocation_id, worker_id + 1, digest)
    assert command == worker_module._CTRL_CCU_ROLLBACK
    return request


def test_ccu_group_prepares_once_and_rebinds_a_new_domain_generation(setup):
    orch, control, handle, domain = setup
    control.control_payload.side_effect = lambda _, rank, cmd, request, timeout: _prepare_reply(rank, cmd, request)
    old_ref = domain.ref
    cfg = CallConfig()
    cfg.aicpu_thread_num = 4

    orch.prepare_ccu(handle, old_ref)
    for _ in range(2):
        orch.submit_next_level_group(handle, [TaskArgs(), TaskArgs()], cfg, workers=[0, 1], ccu_domain=old_ref)
    assert control.control_payload.call_count == 2
    sent_cfg = orch._o.submit_next_level_group.call_args.args[4]
    assert sent_cfg is not cfg and sent_cfg.aicpu_thread_num == 4

    domain._ccu_prepared.clear()
    domain.allocation_id = 1
    domain._ref = CommDomainRef(old_ref.owner_instance_id, 1, 2)
    with pytest.raises(KeyError, match="no live grant"):
        orch.prepare_ccu(handle, old_ref)
    orch.prepare_ccu(handle, domain.ref)
    assert control.control_payload.call_count == 4
    reply = control.control_payload.call_args.args[3]
    assert worker_module._CCU_CONTROL_HEADER.unpack_from(reply)[0] == 1


def test_ccu_group_validates_before_mailbox_control(setup):
    orch, control, handle, domain = setup
    with pytest.raises(ValueError, match="require ccu_domain"):
        orch.submit_next_level_group(handle, [TaskArgs(), TaskArgs()], workers=[0, 1])
    with pytest.raises(ValueError, match="exactly the two domain workers"):
        orch.submit_next_level_group(handle, [TaskArgs()], workers=[0], ccu_domain=domain.ref)
    with pytest.raises(ValueError, match="require ccu_domain"):
        orch.submit_next_level(handle, TaskArgs(), worker=0)
    domain._config = SimpleNamespace(workspace_engines=("urma",))
    with pytest.raises(ValueError, match="workspace_engines"):
        orch.prepare_ccu(handle, domain.ref)
    control.control_payload.assert_not_called()
    orch._o.submit_next_level_group.assert_not_called()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_ccu_partial_prepare_joins_all_ranks_then_rolls_back(setup, rollback_fails):
    orch, control, handle, domain = setup
    events = []

    def issue(_, rank, command, request, timeout):
        events.append((command, rank))
        if command == worker_module._CTRL_CCU_PREPARE and rank == 1:
            raise RuntimeError("rank 1 prepare failed")
        if command == worker_module._CTRL_CCU_ROLLBACK and rank == 0 and rollback_fails:
            raise RuntimeError("rank 0 rollback failed")
        return _prepare_reply(rank, command, request)

    control.control_payload.side_effect = issue
    with pytest.raises(RuntimeError, match="rollback was unconfirmed" if rollback_fails else "rank 1 prepare failed"):
        orch.prepare_ccu(handle, domain.ref)
    assert not domain._ccu_prepared
    assert sorted(events[:2]) == [(worker_module._CTRL_CCU_PREPARE, 0), (worker_module._CTRL_CCU_PREPARE, 1)]
    assert sorted(events[2:]) == [(worker_module._CTRL_CCU_ROLLBACK, 0), (worker_module._CTRL_CCU_ROLLBACK, 1)]
    assert orch._worker is not None
    assert bool(orch._worker._ordered_cleanup_error) == rollback_fails
