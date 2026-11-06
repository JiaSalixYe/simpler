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

#include <stddef.h>
#include <stdint.h>

namespace simpler::ccu {

inline constexpr uint32_t kMaxFunctions = 16;
inline constexpr uint32_t kMaxTensors = 4;
inline constexpr uint32_t kMaxScalars = 4;
inline constexpr uint64_t kIdle = 0;
inline constexpr uint64_t kExecute = 1;
inline constexpr uint64_t kStop = 2;
inline constexpr uint64_t kSuccess = 0;

struct TensorRange {
    uint64_t offset;
    uint64_t bytes;
};

// Immutable device-visible view of the current callable's prepared domain binding.
struct Binding {
    uint64_t domain_ctx;
    uint64_t window_base;
    uint64_t window_size;
    uint64_t mailbox;
    uint32_t function_count;
    uint32_t reserved;
};

// Single in-flight request. Producer, CCU and completion owner communicate
// through separately aligned control lines. A generation is reusable only
// after the completion owner observes it and publishes recycled.
struct alignas(64) Mailbox {
    uint64_t generation;
    uint64_t function_id;
    uint64_t tensor_count;
    uint64_t scalar_count;
    TensorRange tensors[kMaxTensors];
    uint64_t scalars[kMaxScalars];

    alignas(64) uint64_t command;
    alignas(64) uint64_t completed;
    uint64_t status;
    alignas(64) uint64_t recycled;
};

static_assert(sizeof(TensorRange) == 16);
static_assert(sizeof(Binding) == 40);
static_assert(offsetof(Mailbox, tensors) == 32);
static_assert(offsetof(Mailbox, scalars) == 96);
static_assert(offsetof(Mailbox, command) == 128);
static_assert(offsetof(Mailbox, completed) == 192);
static_assert(offsetof(Mailbox, status) == 200);
static_assert(offsetof(Mailbox, recycled) == 256);
static_assert(sizeof(Mailbox) == 320);

}  // namespace simpler::ccu
