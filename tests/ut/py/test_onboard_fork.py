# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device history survives close/failure and never crosses a fork as fresh state."""

import os
import subprocess
import sys
import textwrap
import threading

import pytest
from simpler._onboard_fork import _ProcessDeviceState


def test_failed_bootstrap_still_refuses_hierarchy():
    state = _ProcessDeviceState()
    with pytest.raises(ValueError):
        with state.device_init(True):
            raise ValueError("native bootstrap failed")
    with pytest.raises(RuntimeError, match="fresh Python process"):
        with state.hierarchy():
            pytest.fail("must refuse before hierarchy preparation")
    with state.device_init(True):
        pass


def test_simulation_and_fresh_hierarchies_do_not_latch_device_history():
    state = _ProcessDeviceState()
    with state.device_init(False):
        pass
    for _ in range(2):
        with state.hierarchy():
            with state.hierarchy():
                with state.device_init(False):
                    pass


def test_reentrant_onboard_bootstrap_during_fork_epoch_is_refused():
    state = _ProcessDeviceState()
    with state.hierarchy():
        with pytest.raises(RuntimeError, match="during hierarchy startup"):
            with state.device_init(True):
                pytest.fail("reentrant bootstrap must not contaminate the parent")
    with state.device_init(True):
        pass


def test_onboard_init_waits_for_other_workers_fork_epoch():
    state = _ProcessDeviceState()
    attempted, entered = threading.Event(), threading.Event()

    def initialize():
        attempted.set()
        with state.device_init(True):
            entered.set()

    with state.hierarchy():
        thread = threading.Thread(target=initialize)
        thread.start()
        assert attempted.wait(2)
        assert not entered.wait(0.05)
    thread.join(2)
    assert not thread.is_alive() and entered.is_set()
    with pytest.raises(RuntimeError, match="after onboard initialization"):
        with state.hierarchy():
            pass


def test_child_reset_retains_history_and_discards_inherited_epoch(monkeypatch):
    state = _ProcessDeviceState()
    with state.device_init(True):
        pass
    parent_pid = os.getpid()
    state._fork_depth = 1
    state.after_fork()
    assert state._fork_depth == 0
    monkeypatch.setattr(os, "getpid", lambda: parent_pid + 1)
    with pytest.raises(RuntimeError, match="inherited device state"):
        with state.device_init(True):
            pass
    with pytest.raises(RuntimeError, match="fresh Python process"):
        with state.hierarchy():
            pass


def test_worker_refuses_before_preparing_any_hierarchy(monkeypatch):
    pytest.importorskip("_task_interface")
    from simpler import worker as worker_module  # noqa: PLC0415 -- optional native integration

    state = _ProcessDeviceState()
    with state.device_init(True):
        pass
    monkeypatch.setattr(worker_module, "process_device_state", state)
    worker = worker_module.Worker(level=3, num_sub_workers=0, device_ids=[0])
    monkeypatch.setattr(worker, "_init_hierarchical", lambda: pytest.fail("must refuse before preparation"))
    with pytest.raises(RuntimeError, match="even after close"):
        worker.init()
    worker.close()


@pytest.mark.parametrize("sim_path", [None, "", "sim_context.so"])
def test_bootstrap_tracks_native_platform_even_without_dispatcher(monkeypatch, sim_path):
    pytest.importorskip("_task_interface")
    from types import SimpleNamespace  # noqa: PLC0415

    from simpler import task_interface as task_module  # noqa: PLC0415

    state = _ProcessDeviceState()
    monkeypatch.setattr(task_module, "process_device_state", state)
    monkeypatch.setattr(task_module, "_initialize_host_log", lambda _level: None)

    def failed_init(*_args, **_kwargs):
        raise ValueError("native failed after touching device state")

    bins = SimpleNamespace(host_path="host.so", aicpu_path="cpu.so", aicore_path="core.o", sim_context_path=sim_path)
    with pytest.raises(ValueError, match="native failed"):
        task_module._initialize_chip_runtime(SimpleNamespace(init=failed_init), 0, bins)  # pyright: ignore[reportArgumentType]
    if sim_path:
        with state.hierarchy():
            pass
    else:
        with pytest.raises(RuntimeError, match="fresh Python process"):
            with state.hierarchy():
                pass


def test_real_fork_resets_epoch_but_preserves_parent_history():
    script = textwrap.dedent("""
        import os
        import signal
        from simpler._onboard_fork import process_device_state as state
        with state.hierarchy():
            pid = os.fork()
            if pid == 0:
                signal.alarm(3)
                try:
                    with state.device_init(True):
                        pass
                except BaseException:
                    os._exit(2)
                os._exit(0)
            assert os.waitpid(pid, 0)[1] == 0
        with state.device_init(True):
            pass
        pid = os.fork()
        if pid == 0:
            signal.alarm(3)
            try:
                with state.device_init(True):
                    pass
            except RuntimeError:
                os._exit(0)
            os._exit(3)
        assert os.waitpid(pid, 0)[1] == 0
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_hierarchy_admission_respects_startup_timeout():
    state = _ProcessDeviceState()
    attempted, finished = threading.Event(), threading.Event()
    errors = []

    def start():
        attempted.set()
        try:
            with state.hierarchy(timeout=0.02):
                errors.append("unexpected admission")
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            finished.set()

    with state.device_init(True):
        thread = threading.Thread(target=start)
        thread.start()
        assert attempted.wait(2)
        assert finished.wait(2)
    thread.join(2)
    assert errors == ["hierarchical startup: startup deadline exceeded waiting for process admission"]
