# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pair orchestration inputs with AIV geometry and vector state on A3."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_PROBE = Path(__file__).resolve().parents[4] / "tools/trace_alias_vector_state.py"
pytestmark = [
    pytest.mark.requires_hardware,
    pytest.mark.platforms(["a2a3"]),
    pytest.mark.test_environment("torch_npu"),
]


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_framework_alias_vector_state(runtime, framework_module, tmp_path, request):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    # Forked children can recreate basetemp; retain evidence beside that tree.
    evidence = tmp_path.parent.with_name(tmp_path.parent.name + "-alias-vector-state")
    evidence.mkdir(exist_ok=True)
    case_path = Path(tempfile.mkdtemp(prefix=runtime + "-", dir=evidence))
    logs = case_path / "ascend"
    logs.mkdir()
    output = case_path / "vector-state"
    bridge = next(framework_module.glob("_simpler_framework_queue*.so"))
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(_PROBE),
            "--runtime",
            runtime,
            "--device",
            device,
            "--bridge",
            str(bridge),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={
            **os.environ,
            "TASK_QUEUE_ENABLE": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "MAX_JOBS": "2",
            "ASCEND_PROCESS_LOG_PATH": str(logs),
            "ASCEND_WORK_PATH": str(logs),
        },
    )
    (case_path / "consumer.log").write_text(result.stdout + result.stderr)
    print("ALIAS_VECTOR_STATE_DIR " + str(case_path), flush=True)
    print(result.stdout, end="", flush=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "trace.json").read_text())
    assert summary["completion_confirmed"] and summary["snapshot_consistent"]
    assert summary["runtime"] == runtime and summary["submitted"] == 6
    assert len(summary["rows"]) == 6
    assert all(row["stage"] == 5 and row["producer_marker"] == 0xA3A3F83D for row in summary["rows"])
