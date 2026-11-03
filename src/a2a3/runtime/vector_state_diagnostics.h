/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#pragma once

#include <cinttypes>

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/unified_log.h"
#include "common/memory_barrier.h"
#include "runtime.h"

// Cold-path snapshots can race a failed core; token fields retain that observation's identity.
inline void dump_vector_state_diagnostics(Runtime *runtime, const char *phase) {
    if (runtime == nullptr) return;
    cache_invalidate_range(runtime->dev.workers, sizeof(runtime->dev.workers));
    const uint64_t epoch = get_platform_run_result_epoch();
    if (epoch == 0) return;
    for (int index = 0; index < RUNTIME_MAX_WORKER; ++index) {
        const Handshake &worker = runtime->dev.workers[index];
        const uint64_t token_before = worker.diag_dispatch_token;
        if (worker.report_epoch != epoch || worker.core_type != CoreType::AIV || token_before == 0) continue;
        rmb();
        const uint64_t launch_ctrl = worker.diag_launch_ctrl;
        const uint64_t dispatch_ctrl = worker.diag_dispatch_ctrl;
        const uint64_t raw_core = worker.diag_coreid_raw;
        cache_invalidate_range(&worker, sizeof(worker));
        const uint64_t token_after = worker.diag_dispatch_token;
        LOG_WARN(
            "VECTOR_STATE_DIAG best_effort=1 token_encoding=task_id_plus_one phase=%s epoch=%" PRIu64
            " core=%d physical_core=%" PRIu64 " token_before=%" PRIu64 " token_after=%" PRIu64 " launch_ctrl=0x%" PRIx64
            " dispatch_ctrl=0x%" PRIx64,
            phase, epoch, index, raw_core, token_before, token_after, launch_ctrl, dispatch_ctrl
        );
    }
}
