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

#include "intrinsic.h"
#include "service_mailbox.h"

namespace simpler::ccu_service {

__aicore__ inline void SyncLine(volatile __gm__ uint64_t &value) {
    dcci(const_cast<__gm__ uint64_t *>(&value), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

// The caller serializes the device producers sharing this execution context.
__aicore__ inline bool TrySubmit(
    volatile __gm__ Mailbox &slot, uint64_t generation, uint64_t command, const uint64_t (&payload)[kPayloadWords]
) {
    if (generation == 0 || (command != kExecute && command != kStop)) return false;
    SyncLine(slot.command);
    SyncLine(slot.completed);
    SyncLine(slot.generation);
    // CCU can clear execute before storing completion. The last publication
    // prevents resubmitting the same generation during that interval.
    const uint64_t previous = generation - 1;
    if (slot.command != kIdle || slot.completed != previous || slot.generation != previous) return false;

    for (uint32_t i = 0; i < kPayloadWords; ++i) {
        slot.payload[i] = payload[i];
    }
    slot.generation = generation;
    SyncLine(slot.generation);
    slot.command = command;
    SyncLine(slot.command);
    return true;
}

__aicore__ inline bool Wait(volatile __gm__ Mailbox &slot, uint64_t generation, uint64_t timeout_ticks) {
    const uint64_t start = get_sys_cnt();
    do {
        SyncLine(slot.completed);
        const uint64_t completed = slot.completed;
        if (completed == generation) return true;
        if (completed > generation) return false;
    } while (get_sys_cnt() - start < timeout_ticks);
    return false;
}

}  // namespace simpler::ccu_service
