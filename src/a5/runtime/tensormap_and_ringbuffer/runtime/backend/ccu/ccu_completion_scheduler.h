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

#include "completion_token.h"
#include "platform_comm/ccu/call_protocol.h"

namespace pto2::ccu_backend {

inline CompletionPollResult poll_request(uint64_t address, uint64_t generation) {
    using namespace simpler::ccu;
    const CompletionPollResult invalid{CompletionPollState::FAILED, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID};
    if (address == 0 || address % alignof(Mailbox) != 0 || generation == 0 || generation == UINT64_MAX) {
        return invalid;
    }
    const auto *slot = reinterpret_cast<const Mailbox *>(address);
    const uint64_t completed = __atomic_load_n(&slot->completed, __ATOMIC_ACQUIRE);
    if (completed < generation) {
        return {CompletionPollState::PENDING, SIMPLER_ERROR_NONE};
    }
    if (completed != generation || __atomic_load_n(&slot->status, __ATOMIC_RELAXED) != kSuccess ||
        __atomic_load_n(&slot->command, __ATOMIC_RELAXED) != kIdle ||
        __atomic_load_n(&slot->generation, __ATOMIC_RELAXED) != generation ||
        __atomic_load_n(&slot->recycled, __ATOMIC_RELAXED) != generation - 1) {
        return invalid;
    }
    return {CompletionPollState::READY, SIMPLER_ERROR_NONE};
}

inline void retire_request(uint64_t address, uint64_t generation) {
    auto *slot = reinterpret_cast<simpler::ccu::Mailbox *>(address);
    __atomic_store_n(&slot->recycled, generation, __ATOMIC_RELEASE);
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#endif
}

}  // namespace pto2::ccu_backend
