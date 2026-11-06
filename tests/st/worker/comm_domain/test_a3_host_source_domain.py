# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A3 retained Host windows mapped for both local and peer MTE."""

import struct

import pytest
from simpler.task_interface import AddressSpace, CommBufferSpec, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from examples.workers.l4.global_tload_mixed_l3.main import COUNT, _build_tload_callable

_RUNTIME = "tensormap_and_ringbuffer"


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(1)
def test_a3_device_window_baseline(st_platform, st_device_ids):
    nbytes = COUNT * 4
    worker = Worker(level=3, platform=st_platform, runtime=_RUNTIME, device_ids=[st_device_ids[0]], num_sub_workers=0)
    chip = worker.register(_build_tload_callable(st_platform, _RUNTIME))
    try:
        worker.init()
        scratch = worker.create_buffer(nbytes)

        def run(orch, _args, cfg):
            domain = orch.allocate_domain(
                name="device",
                members=[0],
                window_size=2 * nbytes,
                buffers=[CommBufferSpec(name, "float32", COUNT, nbytes) for name in ("table", "out")],
                retain_after_run=True,
            )
            orch.copy_to_domain(domain, 0, struct.pack(f"<{COUNT}f", *([2.0] * COUNT)), buffer="table")
            args = TaskArgs()
            buffers = domain.member(0).buffers
            args.add_tensor(buffers["table"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.INPUT)
            args.add_tensor(buffers["out"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            args.add_scalar(1)
            args.add_scalar(domain.device_context(0))
            orch.submit_next_level(chip, args, cfg, worker=0)

        worker.run(run)
        worker.copy_from(scratch, worker.live_domains["device"].member(0).buffers["out"])
        assert struct.unpack_from(f"<{COUNT}f", scratch.shm.buf) == (2.0,) * COUNT
    finally:
        worker.close()


@pytest.mark.platforms(["a2a3"])
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize("depth", [1, 2], ids=["serial", "staged"])
def test_host_source_domain_maps_peer_windows(st_platform, st_device_ids, depth):
    nbytes = COUNT * 4
    # The input offset exercises local Tensor carving and peer window addressing.
    prefix = 64
    worker = Worker(
        level=3,
        platform=st_platform,
        runtime=_RUNTIME,
        device_ids=list(st_device_ids[:2]),
        num_sub_workers=0,
        pipeline_depth=depth,
        pending_run_depth=depth,
        launch_depth=1,
    )
    chip = worker.register(_build_tload_callable(st_platform, _RUNTIME))
    request = dict(
        name="host",
        members=[1, 0],
        window_size=prefix + 2 * nbytes,
        buffers=[
            CommBufferSpec("guard", "uint8", prefix, prefix),
            CommBufferSpec("table", "float32", COUNT, nbytes),
            CommBufferSpec("out", "float32", COUNT, nbytes),
        ],
        memory_source=AddressSpace.HOST,
        retain_after_run=True,
    )
    state = {}
    try:
        worker.init()
        scratch = worker.create_buffer(nbytes)

        def prepare(orch, _args, _cfg):
            state["host"] = orch.get_or_allocate_domain(**request)
            state["device"] = orch.get_or_allocate_domain(
                name="device",
                members=[1, 0],
                window_size=nbytes,
                buffers=[CommBufferSpec("out", "float32", COUNT, nbytes)],
                retain_after_run=True,
            )
            for rank in range(2):
                data = struct.pack(f"<{COUNT}f", *([float(rank + 1)] * COUNT))
                orch.copy_to_domain(state["host"], rank, data, buffer="table")
                assert orch.copy_from_domain(state["host"], rank, nbytes, buffer="table") == data

        worker.run(prepare)
        host, device = state["host"], state["device"]
        assert host.memory_source == AddressSpace.HOST and device.memory_source == AddressSpace.DEVICE
        window = host.member(0).buffers["table"].shm
        with pytest.raises(ValueError, match="Host domain windows use CPU access"):
            worker.copy_to(host.member(0).buffers["table"], scratch)
        with pytest.raises(ValueError, match="Host domain windows use CPU access"):
            worker.copy_from(scratch, host.member(0).buffers["table"])

        def run_gather(orch, _args, cfg):
            assert orch.get_or_allocate_domain(**request) is host
            for rank in range(2):
                member = host.member(rank)
                output = member.buffers["out"] if rank == 0 else device.member(rank).buffers["out"]
                args = TaskArgs()
                args.add_tensor(member.buffers["table"].tensor((COUNT,), DataType.FLOAT32), TensorArgType.INPUT)
                args.add_tensor(output.tensor((COUNT,), DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
                args.add_scalar(2)
                args.add_scalar(host.device_context(member.local_worker_id))
                orch.submit_next_level(chip, args, cfg, worker=member.local_worker_id)

        offset, _ = host.buffer_range("out")
        for value in (1.0, 5.0):
            with window.buf[prefix : prefix + nbytes] as table:
                table[:] = struct.pack(f"<{COUNT}f", *([value] * COUNT))
            pending = worker.submit(run_gather)
            following = worker.submit(run_gather) if depth == 2 else None
            pending.wait()
            if following is not None:
                following.wait()
            assert struct.unpack_from(f"<{COUNT}f", window.buf, offset) == (value + 2.0,) * COUNT
            assert bytes(window.buf[:prefix]) == bytes(prefix)
            worker.copy_from(scratch, device.member(1).buffers["out"])
            assert struct.unpack_from(f"<{COUNT}f", scratch.shm.buf) == (value + 2.0,) * COUNT
        with window.buf[offset : offset + nbytes] as exported:
            host.release()
            assert host.freed and window.buf is None
            assert struct.unpack_from(f"<{COUNT}f", exported) == (7.0,) * COUNT
        worker.copy_from(scratch, device.member(1).buffers["out"])
        assert struct.unpack_from(f"<{COUNT}f", scratch.shm.buf) == (7.0,) * COUNT
        device.release()
    finally:
        worker.close()
