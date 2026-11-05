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

#include <stdint.h>

inline constexpr uint32_t kCcuContractRounds = 2;

struct alignas(64) CcuContractMailbox {
    uint64_t ready;
    uint64_t input;
    uint64_t output;
    uint64_t remote_output;
    uint64_t bytes;
    alignas(64) uint64_t release;
    uint64_t gate_saw_consumer;
    alignas(64) uint64_t done;
    alignas(64) uint64_t observed;
    uint64_t observed_value;
};

static_assert(sizeof(CcuContractMailbox) == 256);
