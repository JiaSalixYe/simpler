# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise two CCU IDs in one A5 TMR graph and reuse the domain across runs."""

from __future__ import annotations

import struct
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from simpler.buffer import Buffer
from simpler.task_interface import (
    ArgDirection,
    CcuFunction,
    ChipCallable,
    CommBufferSpec,
    DataType,
    TaskArgs,
    TensorArgType,
)
from simpler.worker import Worker

from simpler_setup.scene_test import compile_chip_callable_spec

_HERE = Path(__file__).resolve().parent
_RUNTIME = "tensormap_and_ringbuffer"
_ELEMENTS = 16
_SLICE_BYTES = 256
_VIEW_OFFSET = _ELEMENTS * 4
_PACK_SLICE = struct.Struct(f"<{_SLICE_BYTES // 4}f")
_SENTINEL = -93.0


def _build_callable(platform: str, build_dir: Path) -> ChipCallable:
    subprocess.run(["cmake", "-S", str(_HERE), "-B", str(build_dir)], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "ccu_collectives_builder"], check=True)
    module = (build_dir / "libccu_collectives_builder.so").read_bytes()
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(_HERE / "kernels/orchestration/ccu_collectives_orch.cpp"),
                "function_name": "pair_orch",
                "config_name": "pair_config",
                "signature": [ArgDirection.IN, ArgDirection.OUT, ArgDirection.OUT, ArgDirection.SCALAR],
            },
            "incores": [
                {
                    "source": str(_HERE / "kernels/aiv/ccu_collectives_submit.cpp"),
                    "core_type": "aiv",
                    "func_id": 0,
                    "signature": [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR, ArgDirection.SCALAR],
                }
            ],
            "ccu_functions": [
                CcuFunction.build(binary=module, entry=name) for name in ("build_allreduce", "build_allgather")
            ],
        },
        platform,
        _RUNTIME,
        "ccu-pair-allreduce-allgather",
    )


def _slice_bytes(values: Sequence[float]) -> bytes:
    guards = [_SENTINEL] * (_SLICE_BYTES // 4)
    guards[_ELEMENTS : _ELEMENTS + len(values)] = values
    return _PACK_SLICE.pack(*guards)


def _host_view(buffer: Buffer) -> memoryview:
    shm = buffer.shm
    assert shm is not None
    view = shm.buf
    assert view is not None
    return view


@pytest.mark.platforms(["a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(_RUNTIME)
@pytest.mark.manual
def test_ccu_allreduce_allgather(st_platform, st_device_ids, tmp_path):
    program = _build_callable(st_platform, tmp_path / "builder")
    assert [fn.entry for fn in program.ccu_functions] == ["build_allreduce", "build_allgather"]
    ranks = [0, 1]
    with Worker(level=3, platform=st_platform, runtime=_RUNTIME, device_ids=list(st_device_ids)) as worker:
        handle = worker.register(program)
        worker.init()
        staged = [worker.create_buffer(_SLICE_BYTES) for _ in ranks]
        initial_output = worker.create_buffer(_SLICE_BYTES)
        readback = worker.create_buffer(_SLICE_BYTES)
        _host_view(initial_output)[:] = _slice_bytes([])
        domains = []

        for iteration in range(2):
            inputs = [[float((rank + 1) * (i + 1) + iteration * 50) for i in range(_ELEMENTS)] for rank in ranks]
            for rank in ranks:
                _host_view(staged[rank])[:] = _slice_bytes(inputs[rank])

            def graph(orch, _args, _config):
                domain = orch.get_or_allocate_domain(
                    name="ccu_collectives",
                    members=ranks,
                    window_size=3 * _SLICE_BYTES,
                    buffers=[
                        CommBufferSpec(name, "float32", _SLICE_BYTES // 4, _SLICE_BYTES)
                        for name in ("src", "sum", "gather")
                    ],
                    retain_after_run=True,
                    workspace_engines=("ccu",),
                )
                domains.append(domain)
                orch.prepare_ccu(handle, domain.ref)
                args_list = []
                for rank in ranks:
                    buffers = domain[rank].buffers
                    args = TaskArgs()
                    for name, count, direction, initial in (
                        ("src", _ELEMENTS, TensorArgType.INPUT, staged[rank]),
                        ("sum", _ELEMENTS, TensorArgType.OUTPUT, initial_output),
                        ("gather", 2 * _ELEMENTS, TensorArgType.OUTPUT, initial_output),
                    ):
                        orch.copy_to(buffers[name], initial)
                        view = buffers[name].tensor((count,), DataType.FLOAT32, byte_offset=_VIEW_OFFSET)
                        args.add_tensor(view, direction)
                    args.add_scalar(domain.device_context(rank, workspace_engines=("ccu",)))
                    args_list.append(args)
                orch.submit_next_level_group(handle, args_list, workers=ranks, ccu_domain=domain.ref)

            worker.run(graph)
            domain = domains[-1]
            assert domain is domains[0]
            expected_sum = [sum(values) for values in zip(*inputs)]
            expected_gather = inputs[0] + inputs[1]
            for rank in ranks:
                for name, values in (("src", inputs[rank]), ("sum", expected_sum), ("gather", expected_gather)):
                    worker.copy_from(readback, domain[rank].buffers[name])
                    actual = bytes(_host_view(readback))
                    assert actual == _slice_bytes(values), (
                        f"iteration={iteration} rank={rank} buffer={name} "
                        f"actual={_PACK_SLICE.unpack(actual)} expected={values}"
                    )

        domain.release()
        assert domain.freed
        for buffer in [*staged, initial_output, readback]:
            worker.release_buffer(buffer)
