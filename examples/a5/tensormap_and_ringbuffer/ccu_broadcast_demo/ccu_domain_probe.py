# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the CCU deferred completion witness on real Simpler L3 domain windows."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import struct
import tempfile
import time
from contextlib import closing
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from unittest.mock import patch

from ccu_driver import ALLOCATION_BYTES, RUNTIME, NativeSession, build_callable, build_helper, validate_request

PAYLOAD_FLOATS = 64
SRC_OFFSET = 32 * 4
DST_OFFSET = (1024 + 2 * 32) * 4


def _publish(path, data):
    pending = path.with_name(path.name + ".tmp")
    pending.write_bytes(data)
    pending.replace(path)


def _read_peer(path):
    deadline = time.monotonic() + 30
    while True:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for CCU domain probe record {path.name}") from None
            time.sleep(0.01)


def _shm_bytes(name):
    with closing(SharedMemory(name=name)) as shm:
        buf = shm.buf
        assert buf is not None
        return bytes(buf)


def _require_domain_tensors(domain, worker_id, *views):
    # The current framework validates the destination worker, while this probe
    # additionally validates the selected domain. Addresses can be recycled.
    identities = {buffer.identity for buffer in domain[worker_id].buffers.values()}
    if any(view.buffer.identity not in identities for view in views):
        raise ValueError("CCU probe requires views from the selected domain and rank")


class _DomainProbe:
    """Fork-inherited test adapter around the actual domain controls."""

    def __init__(self, library, rendezvous):
        self.library = library
        self.rendezvous = rendezvous
        self.worker_module = importlib.import_module("simpler.worker")
        self.allocate_domain = self.worker_module._handle_ctrl_alloc_domain
        self.release_domain = self.worker_module._handle_ctrl_release_domain
        self.finalize_chip = self.worker_module._finalize_chip_worker_and_publish
        self.sessions = {}

    def _request(self, buf):
        wm = self.worker_module
        return wm._DOMAIN_REQ_HEADER.unpack_from(_shm_bytes(wm._read_shm_name(buf, wm._OFF_ARGS)))

    def allocate(self, cw, buf, *, base_comm_handle):
        allocation, ranks, rank, window_size, buffers = self._request(buf)
        self.allocate_domain(cw, buf, base_comm_handle=base_comm_handle)
        if buffers != 2 or window_size != 2 * ALLOCATION_BYTES:
            return
        if ranks != 2:
            raise RuntimeError("CCU domain probe requires two ranks")

        wm = self.worker_module
        reply = _shm_bytes(wm._read_shm_name(buf, wm._OFF_ARGS + wm._CTRL_SHM_NAME_BYTES))
        committed, device_ctx, window_base, mapping_size, buffer_count = wm._DOMAIN_REPLY_HEADER.unpack_from(reply)
        src_base, dst_base = struct.unpack_from("<2Q", reply, wm._DOMAIN_REPLY_HEADER.size)
        if (
            not committed
            or device_ctx == 0
            or buffer_count != 2
            or src_base != window_base
            or dst_base != src_base + ALLOCATION_BYTES
            or mapping_size < window_size
        ):
            raise RuntimeError("Unexpected VMM domain geometry")

        native = NativeSession(self.library)
        self.sessions[allocation] = native, cw.device_id
        root_file = self.rendezvous / f"{allocation}.root"
        if rank == 0:
            _publish(root_file, native.root_info(cw.device_id))
        # CCU is owned by a private communicator for this allocation. The
        # payload comes from the real child-owned VMM windows allocated above.
        peer = native.open(
            cw.device_id, rank, _read_peer(root_file), PAYLOAD_FLOATS, allocation % 1000, True, (src_base, dst_base)
        )
        _publish(self.rendezvous / f"{allocation}.peer{rank}", json.dumps(peer).encode())
        peers = [json.loads(_read_peer(self.rendezvous / f"{allocation}.peer{i}")) for i in range(ranks)]
        mailbox = native.prepare_resident(peers, 0, service=False)
        _publish(
            self.rendezvous / f"{allocation}.ready{rank}",
            json.dumps((device_ctx, mailbox, peers[1 - rank][1], peer[0], peer[1])).encode(),
        )

    def release(self, cw, buf, *, base_comm_handle):
        allocation, _, rank, abort, _ = self._request(buf)
        session = self.sessions.get(allocation)
        if session is not None:
            native, device = session
            program = native.lib.ccu_demo_program_handle(native.handle)
            if not abort:
                # Verifies exact consumer values, both whole VMM slice guards
                # and peer data. Only an observed drain permits window release.
                native.check(native.lib.ccu_demo_verify_resident(native.handle, True))
            native.close()
            del self.sessions[allocation]
            if program:
                native.check(native.lib.ccu_demo_verify_program_retired(device, program))
            print(f"CCU_DOMAIN_RETIRED allocation={allocation} rank={rank}", flush=True)
        self.release_domain(cw, buf, base_comm_handle=base_comm_handle)

    def finalize(self, buf, cw):
        if self.sessions:
            # The stock VMM teardown has no knowledge of this private CCU
            # session. Do not explicitly free its windows after an unknown drain.
            print("CCU_DOMAIN_UNCONFIRMED: retaining private sessions until process exit", flush=True)
            os._exit(1)
        self.finalize_chip(buf, cw)

    def args(self, domain, worker_id):
        from simpler.task_interface import DataType, TaskArgs, TensorArgType  # noqa: PLC0415

        rank_context = domain[worker_id]
        rank = rank_context.domain_rank
        allocation = domain.allocation_id
        ctx, mailbox, peer_output, input_addr, output_addr = json.loads(
            _read_peer(self.rendezvous / f"{allocation}.ready{rank}")
        )
        src = rank_context.buffers["src"]
        dst = rank_context.buffers["dst"]
        if (
            ctx != domain.device_context(worker_id)
            or input_addr != src.base + SRC_OFFSET
            or output_addr != dst.base + DST_OFFSET
        ):
            raise RuntimeError("CCU probe binding does not match the current domain")
        src_view = src.tensor((PAYLOAD_FLOATS,), DataType.FLOAT32, byte_offset=SRC_OFFSET)
        dst_view = dst.tensor((PAYLOAD_FLOATS,), DataType.FLOAT32, byte_offset=DST_OFFSET)
        _require_domain_tensors(domain, worker_id, src_view, dst_view)
        args = TaskArgs()
        args.add_tensor(src_view, TensorArgType.INOUT)
        args.add_tensor(dst_view, TensorArgType.INOUT)
        for value in (mailbox, peer_output, 1, rank * 10000 + allocation % 1000 * 1000):
            args.add_scalar(value)
        return args


def _expect_rejected(action, exception, match):
    try:
        action()
    except exception as error:
        if match in str(error):
            return
        raise
    raise AssertionError(f"Expected CCU domain probe to reject {match!r}")


def run_domain_probe(devices):
    from simpler.task_interface import CommBufferSpec, DataType  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    validate_request(devices, [PAYLOAD_FLOATS], [0], 1, 120)
    library = build_helper()
    callable_obj = build_callable(resident=True)
    active = {
        "name": "ccu_payload",
        "members": [0, 1],
        "window_size": 2 * ALLOCATION_BYTES,
        "buffers": [CommBufferSpec(name, "opaque", ALLOCATION_BYTES, ALLOCATION_BYTES) for name in ("src", "dst")],
        "retain_after_run": True,
    }
    foreign = dict(active, name="ccu_foreign", window_size=ALLOCATION_BYTES, buffers=active["buffers"][:1])
    with tempfile.TemporaryDirectory(prefix="simpler-ccu-domain-") as directory:
        probe = _DomainProbe(library, Path(directory))
        wm = probe.worker_module
        with (
            patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": "CCU_SCHED"}),
            patch.object(wm, "_handle_ctrl_alloc_domain", probe.allocate),
            patch.object(wm, "_handle_ctrl_release_domain", probe.release),
            patch.object(wm, "_finalize_chip_worker_and_publish", probe.finalize),
            Worker(level=3, platform="a5", runtime=RUNTIME, device_ids=devices, num_sub_workers=0) as worker,
        ):
            handle = worker.register(callable_obj)
            worker.init()
            runs = []

            def orch_fn(orch, _args, config):
                unrelated = orch.get_or_allocate_domain(**foreign)
                domain = orch.get_or_allocate_domain(**active)
                ref = domain.ref
                if runs:
                    old, old_ref, old_view = runs[-1]
                    if ref == old_ref or not old.freed:
                        raise AssertionError("Same-name domain did not get a fresh identity")
                    _expect_rejected(lambda: old.device_context(0), RuntimeError, "already released")
                    _expect_rejected(
                        lambda: _require_domain_tensors(domain, 0, old_view), ValueError, "selected domain"
                    )
                wrong_view = unrelated[0].buffers["src"].tensor((PAYLOAD_FLOATS,), DataType.FLOAT32)
                _expect_rejected(lambda: _require_domain_tensors(domain, 0, wrong_view), ValueError, "selected domain")
                old_view = domain[0].buffers["src"].tensor((PAYLOAD_FLOATS,), DataType.FLOAT32, byte_offset=SRC_OFFSET)
                runs.append((domain, ref, old_view))
                orch.submit_next_level_group(handle, [probe.args(domain, i) for i in (0, 1)], config, workers=[0, 1])

            for _ in range(2):
                worker.run(orch_fn)
                domain = runs[-1][0]
                domain.release()
                if not domain.freed:
                    raise AssertionError("Domain was not freed after CCU retirement")
                print(f"CCU_DOMAIN_RUN_PASS allocation={domain.allocation_id} cross_domain=rejected", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--devices", required=True, help="Two physical A5 device IDs")
    run_domain_probe([int(device) for device in parser.parse_args().devices.split(",")])
