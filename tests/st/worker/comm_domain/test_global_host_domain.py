# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Manual two-node Host read/write using the existing L3 vector-add callable."""

import ctypes
import os
import socket
import struct
import subprocess
import sys
import time

import pytest
from simpler.buffer import AccessMode
from simpler.task_interface import (
    CommBufferSpec,
    CommDomainRef,
    DataType,
    HostWindowSpec,
    TaskArgs,
    TensorArgType,
    TensorTransfer,
)
from simpler.worker import RemoteCallable, RemoteWorkerSpec, Worker

from tests.st.worker.comm_domain.test_host_domain import RUNTIME, SIZE, N, build_chip_callable

_VIEWS = {}


def node_step(orch, args, cfg, chips=None):
    from simpler.remote_l3_session import get_inner_handle  # noqa: PLC0415

    views = _VIEWS
    digest = b"".join(int(args.scalar(index)).to_bytes(8, "little") for index in range(5, 9))
    node, phase = int(args.scalar(0)), int(args.scalar(1))
    ref = CommDomainRef(int(args.scalar(2)).to_bytes(8, "little"), int(args.scalar(3)), int(args.scalar(4)))
    domain = orch.get_domain(ref)
    rank, peer = 1 - node, 1 - node
    native = domain.host(node)
    assert native.base, "the source L3 must expose its actual CPU owner view"
    if phase == 0:
        data = struct.pack(f"<{N}f", *([3.0 * (node + 1)] * N))
        ctypes.memmove(native.base, data, SIZE)
        ctypes.memset(native.base + SIZE, 0, SIZE)
        return
    if phase == 3:
        original = struct.unpack(f"<{N}f", ctypes.string_at(native.base, SIZE))
        written = struct.unpack(f"<{N}f", ctypes.string_at(native.base + SIZE, SIZE))
        assert original == (3.0 * (node + 1),) * N
        assert written == (12.0 * (node + 1),) * N
        domain.unmap_view(views.pop(node))
        return

    def submit(a, b, out):
        task = TaskArgs()
        for tensor in (a, b, out):
            task.add_tensor(tensor, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
        return orch.submit_next_level(
            chips[node] if chips is not None else get_inner_handle(digest.hex()), task, cfg, worker=0
        )

    source = domain.host(peer).tensor((N,), DataType.FLOAT32, byte_offset=SIZE if phase == 2 else 0)
    with pytest.raises(ValueError, match="receiving L3"):
        domain.map_view(source, domain_rank=1 - rank, access=AccessMode.READ)
    output = domain.member(rank).buffers["output"].tensor((N,), DataType.FLOAT32)
    if phase == 1:
        mapped = domain.map_view(source, domain_rank=rank, access=AccessMode.READ)
        views[node] = mapped
        submit(mapped, mapped, output)
    else:
        old = views[node]
        domain.unmap_view(old)
        with pytest.raises(ValueError):
            submit(old, old, output)
        mapped = domain.map_view(source, domain_rank=rank, access=AccessMode.WRITE)
        assert mapped.mapping_ref != old.mapping_ref
        views[node] = mapped
        submit(output, output, mapped)


@pytest.fixture
def host_endpoints(request, tmp_path):
    if request.param == "local":
        return None
    external = os.environ.get("SIMPLER_HOST_ENDPOINTS")
    if external:
        endpoints = tuple(external.split(","))
        if len(endpoints) != 2 or not all(endpoints):
            raise ValueError("SIMPLER_HOST_ENDPOINTS requires two host:port endpoints")
        return endpoints
    processes, endpoints, logs = [], [], []

    def stop():
        for process in processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()

    request.addfinalizer(stop)
    for node in range(2):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        log_path = tmp_path / f"host-daemon-{node}.log"
        log = log_path.open("w")
        logs.append(log)
        process = subprocess.Popen(
            [sys.executable, "-m", "simpler.remote_l3_worker", "--host", "127.0.0.1", "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append(process)
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Host test daemon did not become ready: {log_path.read_text()}")
                time.sleep(0.01)
        endpoints.append(f"127.0.0.1:{port}")
    return tuple(endpoints)


@pytest.mark.manual
@pytest.mark.platforms(["a2a3sim", "a2a3"])
@pytest.mark.runtime(RUNTIME)
@pytest.mark.device_count(2)
@pytest.mark.parametrize("host_endpoints", ["local", "tcp"], indirect=True)
def test_global_host_peer_read_write_and_revoke(st_platform, st_device_ids, host_endpoints):
    worker = Worker(level=4, num_sub_workers=0)
    chips = []
    domains = []
    callable_obj = build_chip_callable(st_platform)
    profile = "sim" if st_platform.endswith("sim") else "a3-fabric-v1"
    for node in range(2):
        if host_endpoints is not None:
            assert (
                worker.add_remote_worker(
                    RemoteWorkerSpec(
                        endpoint=host_endpoints[node],
                        platform=st_platform,
                        runtime=RUNTIME,
                        device_ids=(int(st_device_ids[node]),),
                        comm_profile=profile,
                        global_device_ranks=(node,),
                    )
                )
                == node
            )
        else:
            child = Worker(
                level=3,
                platform=st_platform,
                runtime=RUNTIME,
                device_ids=[int(st_device_ids[node])],
                num_sub_workers=0,
                comm_profile=profile,
                global_device_ranks=[node],
            )
            chips.append(child.register(callable_obj))
            assert worker.add_worker(child) == node
    chip = worker.register(callable_obj)

    def local_step(orch, args, cfg):
        node_step(orch, args, cfg, chips)

    route = (
        worker.register(RemoteCallable(__name__ + ":node_step"), workers=[0, 1])
        if host_endpoints
        else worker.register(local_step)
    )

    request = dict(
        name="peer-host",
        members=[(1, 0), (0, 0)],
        window_size=SIZE,
        buffers=[CommBufferSpec("output", "float32", N, SIZE)],
        host_windows={0: HostWindowSpec(SIZE * 2), 1: HostWindowSpec(SIZE * 2)},
        retain_after_run=True,
    )
    try:
        worker.init()
        for phase in range(4):

            def forward(orch, _args, cfg, phase=phase):
                domain = orch.get_or_allocate_domain(**request)
                if domains:
                    assert domain is domains[0]
                else:
                    domains.append(domain)
                for node in range(2):
                    args = TaskArgs()
                    for scalar in (
                        node,
                        phase,
                        int.from_bytes(domain.ref.owner_instance_id, "little"),
                        domain.ref.domain_id,
                        domain.ref.generation,
                        *(int.from_bytes(chip.digest[start : start + 8], "little") for start in range(0, 32, 8)),
                    ):
                        args.add_scalar(scalar)
                    orch.submit_next_level(route, args, cfg, worker=node)

            worker.run(forward)
        domains[0].release()
        assert domains[0].freed
    finally:
        worker.close()
