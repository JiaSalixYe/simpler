# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Build and submit the A5 CCU broadcast through the Simpler callable path."""

from pathlib import Path

from simpler.task_interface import ArgDirection, CcuFunction, CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup.scene_test import compile_chip_callable_spec

HERE = Path(__file__).resolve().parent
RUNTIME = "tensormap_and_ringbuffer"
ELEMENTS = 16
BYTES = ELEMENTS * 4


def build_broadcast(module: Path):
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.SCALAR, ArgDirection.SCALAR]
    return compile_chip_callable_spec(
        {
            "orchestration": {
                "source": str(HERE / "kernels/orchestration/ccu_callable_orch.cpp"),
                "function_name": "ccu_callable_orchestration",
                "config_name": "ccu_callable_config",
                "signature": signature,
            },
            "incores": [
                {
                    "func_id": 0,
                    "source": str(HERE / "kernels/aiv/ccu_callable_submit.cpp"),
                    "core_type": "aiv",
                    "signature": signature,
                }
            ],
            "ccu_functions": [CcuFunction.build(binary=module.read_bytes(), entry="build_broadcast")],
        },
        "a5",
        RUNTIME,
        "a5-ccu-callable-broadcast",
    )


def submit_broadcast(orch, handle, workers, *, produce):
    """Use a retained domain and let the caller submit a rank-zero producer."""
    domain = orch.get_or_allocate_domain(
        name="ccu_broadcast",
        members=workers,
        window_size=2 * BYTES,
        buffers=[CommBufferSpec(name, "float32", ELEMENTS, BYTES) for name in ("src", "dst")],
        retain_after_run=True,
        workspace_engines=("ccu",),
    )
    orch.prepare_ccu(handle, domain.ref)
    produce(orch, workers[0], domain[workers[0]].buffers["src"].tensor(shapes=(ELEMENTS,), dtype=DataType.FLOAT32))
    args_list = []
    for worker in workers:
        buffers = domain[worker].buffers
        args = TaskArgs()
        args.add_tensor(buffers["src"].tensor(shapes=(ELEMENTS,), dtype=DataType.FLOAT32), TensorArgType.INPUT)
        args.add_tensor(buffers["dst"].tensor(shapes=(ELEMENTS,), dtype=DataType.FLOAT32), TensorArgType.OUTPUT)
        args.add_scalar(domain.device_context(worker, workspace_engines=("ccu",)))
        args.add_scalar(ELEMENTS)
        args_list.append(args)
    completion = orch.submit_next_level_group(handle, args_list, workers=workers, ccu_domain=domain.ref)
    return domain, completion
