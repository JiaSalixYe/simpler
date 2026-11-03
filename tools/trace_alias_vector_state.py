# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Capture A3 alias vector state without changing vector mode; run under task-submit."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples/workers/l2/framework_eager"
_WORDS = 64
_UNPROVEN = []
_FIELDS: tuple[str, ...] = (
    "entry_ctrl",
    "pre_add_ctrl",
    "post_add_issue_ctrl",
    "src_rows",
    "src_cols",
    "dst_rows",
    "dst_cols",
    "src_gm",
    "dst_gm",
    "stage",
    "src_stride0",
    "dst_stride0",
    "core_id",
    "tile_rows",
    "tile_cols",
    "expected_repeats",
    "src_ub",
    "target_ub",
    "result_ub",
    "post_vector_sync_ctrl",
    "post_all_sync_ctrl",
    "source_tensor_addr",
    "target_tensor_addr",
    "source_stride1",
    "target_stride1",
    "source_offset",
    "target_offset",
    "source_buffer",
    "target_buffer",
    "source_ndims",
    "target_ndims",
    "trace_device_addr",
    "producer_source_buffer",
    "producer_target_buffer",
    "producer_source_offset",
    "producer_target_offset",
    "producer_src_rows",
    "producer_src_cols",
    "producer_dst_rows",
    "producer_dst_cols",
    "producer_src_stride0",
    "producer_src_stride1",
    "producer_dst_stride0",
    "producer_dst_stride1",
    "producer_source_ndims",
    "producer_target_ndims",
    "producer_tensor_count",
    "producer_scalar_count",
    "producer_trace_device",
    "producer_marker",
)
_PUBLISH = (
    " ".join(
        f"dcci(const_cast<__gm__ uint64_t *>(trace) + {offset}, SINGLE_CACHE_LINE, CACHELINE_OUT);"
        for offset in range(0, _WORDS, 8)
    )
    + " dsb(DSB_DDR);"
)


def _publish(stage):
    return (
        _PUBLISH
        + f" trace[9] = {stage}; "
        + "dcci(const_cast<__gm__ uint64_t *>(trace) + 8, SINGLE_CACHE_LINE, CACHELINE_OUT); dsb(DSB_DDR);"
    )


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old!r}")
    return source.replace(old, new, 1)


def generate_sources(output):
    source = (_EXAMPLE / "kernels/aiv/alias_inout.cpp").read_text()
    source = _replace_once(
        source,
        'extern "C" __aicore__',
        """struct AliasVectorRecord { volatile uint64_t values[64]; };

template <typename TileData>
__tf__ PTO_INTERNAL void record_tile_address(
    typename TileData::TileDType __in__ tile, __gm__ volatile uint64_t *out) {
    *out = reinterpret_cast<uint64_t>((__ubuf__ float *)__cce_get_tile_ptr(tile));
}

extern "C" __aicore__""",
    )
    entry = """    auto *record = reinterpret_cast<__gm__ AliasVectorRecord *>(args[2]);
    auto &trace = record->values;
    trace[0] = get_ctrl();
    trace[3] = source_tensor->shapes[0]; trace[4] = source_tensor->shapes[1];
    trace[5] = target_tensor->shapes[0]; trace[6] = target_tensor->shapes[1];
    trace[7] = reinterpret_cast<uint64_t>(source); trace[8] = reinterpret_cast<uint64_t>(target);
    trace[10] = source_tensor->strides[0]; trace[11] = target_tensor->strides[0];
    trace[12] = get_coreid();
    trace[21] = args[0]; trace[22] = args[1];
    trace[23] = source_tensor->strides[1]; trace[24] = target_tensor->strides[1];
    trace[25] = source_tensor->start_offset; trace[26] = target_tensor->start_offset;
    trace[27] = source_tensor->buffer.addr; trace[28] = target_tensor->buffer.addr;
    trace[29] = source_tensor->ndims; trace[30] = target_tensor->ndims; trace[31] = args[2];
    trace[32] = args[3]; trace[33] = args[4]; trace[34] = args[5]; trace[35] = args[6];
    for (int i = 0; i < 6; ++i) {
        trace[36 + i * 2] = static_cast<uint32_t>(args[7 + i]);
        trace[37 + i * 2] = static_cast<uint64_t>(args[7 + i]) >> 32;
    }
    trace[48] = args[13]; trace[49] = args[14];
"""
    source = _replace_once(source, "    using ShapeType", entry + "    " + _publish(1) + "\n    using ShapeType")
    add = "    pto::TADD(result, source_tile, target_tile);"
    before = """    trace[1] = get_ctrl();
    trace[13] = result.GetValidRow(); trace[14] = result.GetValidCol();
    trace[15] = (result.GetValidRow() * result.GetValidCol()) / 64;
    record_tile_address<Data>(source_tile.data(), trace + 16);
    record_tile_address<Data>(target_tile.data(), trace + 17);
    record_tile_address<Data>(result.data(), trace + 18);
"""
    source = _replace_once(
        source, add, before + "    " + _publish(2) + "\n" + add + "\n    trace[2] = get_ctrl(); " + _publish(3)
    )
    sync = "    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);"
    source = _replace_once(
        source,
        sync,
        sync
        + "\n    set_flag(PIPE_V, PIPE_S, EVENT_ID1); wait_flag(PIPE_V, PIPE_S, EVENT_ID1);\n"
        + "    trace[19] = get_ctrl(); "
        + _publish(4),
    )
    source = _replace_once(
        source, "    pipe_sync();", "    pipe_sync();\n" + "    trace[20] = get_ctrl(); " + _publish(5)
    )
    kernel = output / "alias_vector_state.cpp"
    kernel.write_text(source)
    orchestration = (_EXAMPLE / "kernels/alias_inout_orchestration.cpp").read_text()
    orchestration = _replace_once(orchestration, "expected_arg_count = 2", "expected_arg_count = 3")
    producer = """    const auto &source = args.tensor(0).ref();
    const auto &target = args.tensor(1).ref();
    const auto pack = [](uint32_t low, uint32_t high) {
        return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
    };
    const uint64_t counts = pack(task.tensor_count(), task.scalar_count() + 12);
    task.add_scalar(
        source.buffer.addr, target.buffer.addr, source.start_offset, target.start_offset,
        pack(source.shapes[0], source.shapes[1]), pack(target.shapes[0], target.shapes[1]),
        pack(source.strides[0], source.strides[1]), pack(target.strides[0], target.strides[1]),
        pack(source.ndims, target.ndims), counts, args.scalar<uint64_t>(0), uint64_t(0xa3a3f83d));
"""
    orchestration = _replace_once(
        orchestration,
        "    rt_submit_aiv_task(0, task);",
        "    task.add_scalar(args.scalar<uint64_t>(0));\n" + producer + "    rt_submit_aiv_task(0, task);",
    )
    orch = output / "alias_vector_state_orchestration.cpp"
    orch.write_text(orchestration)
    return kernel, orch


def _host_trace(calls):
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclrtMallocHost.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    acl.aclrtHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    acl.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    acl.aclrtFreeHost.argtypes = [ctypes.c_void_p]
    raw, device = ctypes.c_void_p(), ctypes.c_void_p()
    size = ((calls * _WORDS * 8 + 4095) // 4096) * 4096
    rc = acl.aclrtMallocHost(ctypes.byref(raw), size + 4095)
    if rc != 0 or not raw.value:
        raise RuntimeError(f"aclrtMallocHost: {rc}")
    aligned = (raw.value + 4095) & ~4095
    ctypes.memset(aligned, 0, size)
    rc = acl.aclrtHostRegister(ctypes.c_void_p(aligned), size, 0, ctypes.byref(device))
    if rc != 0 or not device.value:
        acl.aclrtFreeHost(raw)
        raise RuntimeError(f"aclrtHostRegister: {rc}")
    records = (ctypes.c_uint64 * (calls * _WORDS)).from_address(aligned)
    return acl, raw, aligned, device.value, records


def run(options):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.task_interface import ArgDirection, TaskArgs  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    kernel, orch = generate_sources(output)
    spec = importlib.util.spec_from_file_location("alias_state_consumer", _EXAMPLE / "alias_inout.py")
    assert spec is not None and spec.loader is not None
    consumer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(consumer)
    signature = [ArgDirection.IN, ArgDirection.INOUT, ArgDirection.SCALAR]

    def compile_probe(runtime, platform="a2a3"):
        return compile_chip_callable_spec(
            {
                "orchestration": {
                    "source": str(orch),
                    "function_name": "framework_alias_inout",
                    "signature": signature,
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": str(kernel),
                        "core_type": "aiv",
                        "signature": signature + [ArgDirection.SCALAR] * 12,
                    },
                ],
            },
            platform,
            runtime,
            "alias-vector-state-" + runtime,
        )

    consumer._compile = compile_probe
    if options.compile_only:
        compile_probe(options.runtime)
        return
    torch.npu.set_device(options.device)
    acl, raw, aligned, device, records = _host_trace(options.calls * 2)
    original = Worker.submit
    submitted = 0

    def submit(worker, handle, args, *positional, **keywords):
        nonlocal submitted
        if submitted >= options.calls * 2:
            raise RuntimeError("trace record capacity exceeded")
        copied = TaskArgs()
        for index in range(args.tensor_count()):
            copied.add_tensor(args.tensor(index), args.tag(index), transfer=args.transfer(index))
        copied.add_scalar(device + submitted * _WORDS * 8)
        submitted += 1
        return original(worker, handle, copied, *positional, **keywords)

    completed = False
    _UNPROVEN.append((acl, raw, aligned, records, consumer))
    Worker.submit = submit
    try:
        consumer.run(options.runtime, options.device, options.bridge.resolve(), options.calls, platform="a2a3")
        completed = True
    finally:
        if completed:
            Worker.submit = original
        rows = [
            dict(zip(_FIELDS, records[index * _WORDS : (index + 1) * _WORDS])) for index in range(options.calls * 2)
        ]
        result = {
            "completion_confirmed": completed,
            "snapshot_consistent": completed,
            "runtime": options.runtime,
            "device": options.device,
            "kernel_source_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
            "submitted": submitted,
            "rows": rows,
        }
        (output / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
        print("ALIAS_VECTOR_STATE " + json.dumps(result), flush=True)
        # Unknown completion retains storage that may still be reachable by the device.
        if completed:
            rc = acl.aclrtHostUnregister(ctypes.c_void_p(aligned))
            if rc != 0:
                raise RuntimeError(f"aclrtHostUnregister: {rc}")
            rc = acl.aclrtFreeHost(raw)
            if rc != 0:
                raise RuntimeError(f"aclrtFreeHost: {rc}")
            _UNPROVEN.pop()
    assert submitted == options.calls * 2
    for index, row in enumerate(rows):
        assert row["stage"] == 5, row
        assert [row[key] for key in ("src_rows", "src_cols", "dst_rows", "dst_cols")] == [64] * 4, row
        assert [row[key] for key in ("src_stride0", "dst_stride0")] == [96, 96], row
        assert [row[key] for key in ("tile_rows", "tile_cols", "expected_repeats")] == [64] * 3, row
        assert [row[key] for key in ("src_ub", "target_ub", "result_ub")] == [0, 0x4000, 0x8000], row
        assert row["dst_gm"] - row["src_gm"] == (0 if index < options.calls else 32), row
        assert row["producer_marker"] == 0xA3A3F83D, row
        assert [row[key] for key in ("producer_tensor_count", "producer_scalar_count")] == [2, 13], row
        for name in (
            "source_buffer",
            "target_buffer",
            "source_offset",
            "target_offset",
            "src_rows",
            "src_cols",
            "dst_rows",
            "dst_cols",
            "src_stride0",
            "dst_stride0",
            "source_ndims",
            "target_ndims",
        ):
            assert row["producer_" + name] == row[name], (name, row)
        assert row["producer_src_stride1"] == row["source_stride1"] == 1, row
        assert row["producer_dst_stride1"] == row["target_stride1"] == 1, row
        assert row["producer_trace_device"] == row["trace_device_addr"] == device + index * _WORDS * 8, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["host_build_graph", "tensormap_and_ringbuffer"], required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true")
    options = parser.parse_args()
    if options.calls < 2:
        parser.error("--calls must be at least 2")
    run(options)


if __name__ == "__main__":
    main()
