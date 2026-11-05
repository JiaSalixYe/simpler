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
#include "ccu_contract_device.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    const auto address = tensor_addr(args[0]);
    auto *output = reinterpret_cast<volatile __gm__ float *>(address);
    const auto count = static_cast<uint32_t>(args[1]);
    const auto first = static_cast<uint32_t>(args[2]);
    for (uint32_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(first + i);
    }
    for (uint32_t i = 0; i < count; i += 16) {
        mailbox_sync(reinterpret_cast<volatile __gm__ uint64_t *>(output + i));
    }
    pipe_barrier(PIPE_ALL);
}
