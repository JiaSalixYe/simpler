# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import ctypes
import struct

import pytest
from simpler.callable_identity import build_chip_callable_descriptor, compute_callable_hashid
from simpler.task_interface import ArgDirection, CcuFunction, ChipCallable, CoreCallable


def _build(*functions):
    core = CoreCallable.build(signature=[ArgDirection.IN], binary=b"aiv")
    return ChipCallable.build(
        signature=[ArgDirection.IN],
        func_name="orchestrate",
        binary=b"orch",
        children=[(7, core)],
        ccu_functions=functions,
    )


def _wire(callable_):
    return ctypes.string_at(callable_.buffer_ptr(), callable_.buffer_size())


def _hashid(callable_):
    return compute_callable_hashid(build_chip_callable_descriptor(target=callable_, platform="a5"))


def test_ccu_callable_round_trip_preserves_device_image_and_deduplicates_module():
    module = b"test CCU module\x00\xff"
    copy = CcuFunction.build(binary=module, entry="build_copy")
    reduce = CcuFunction.build(binary=module, entry="build_reduce")
    legacy = _build()
    callable_ = _build(copy, reduce)
    wire = _wire(callable_)

    assert legacy.ccu_function_count == 0
    assert ChipCallable.from_bytes(_wire(legacy)).ccu_functions == ()
    assert callable_.ccu_function_count == 2
    assert wire.startswith(_wire(legacy))
    assert wire[len(_wire(legacy)) :].count(module) == 1
    restored = ChipCallable.from_bytes(wire)
    assert _wire(restored) == wire
    assert restored.child_func_id(0) == 7
    assert restored.ccu_function_count == 2
    assert [(fn.binary, fn.entry) for fn in restored.ccu_functions] == [
        (module, "build_copy"),
        (module, "build_reduce"),
    ]
    assert _hashid(restored) == _hashid(callable_)
    assert _hashid(legacy) != _hashid(callable_)
    assert _hashid(callable_) != _hashid(_build(reduce, copy))
    assert _hashid(callable_) != _hashid(_build(CcuFunction.build(binary=b"other module", entry="build_copy"), reduce))


def test_ccu_callable_rejects_corrupt_serialized_sections():
    module = b"module"
    wire = _wire(_build(CcuFunction.build(binary=module, entry="build_copy")))
    section_start = len(_wire(_build()))
    invalid = [wire[:-1], wire + b"extra"]
    for offset, value in (
        (0, 0),  # magic
        (4, 2),  # version
        (8, 0),  # module count
        (12, 17),  # function count
        (16, len(wire)),  # module bytes
        (20 + len(module), 1),  # invalid module index
    ):
        raw = bytearray(wire)
        struct.pack_into("<I", raw, section_start + offset, value)
        invalid.append(bytes(raw))

    for raw in invalid:
        with pytest.raises(ValueError, match="invalid serialized image"):
            ChipCallable.from_bytes(raw)


def test_ccu_callable_rejects_invalid_host_entry():
    with pytest.raises(ValueError, match="invalid module binary size"):
        CcuFunction.build(binary=b"", entry="build_copy")
    with pytest.raises(ValueError, match="entry symbol"):
        CcuFunction.build(binary=b"module", entry="build\0copy")
    entry = CcuFunction.build(binary=b"module", entry="build_copy")
    with pytest.raises(ValueError, match="too many CCU functions"):
        _build(*[entry] * 17)
