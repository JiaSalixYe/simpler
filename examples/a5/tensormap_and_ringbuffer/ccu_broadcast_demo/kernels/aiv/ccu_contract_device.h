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
#include "tensor.h"
#include "../../ccu_contract_mailbox.h"

inline __aicore__ void mailbox_sync(volatile __gm__ uint64_t *address) {
    dcci(const_cast<__gm__ uint64_t *>(address), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
}

inline __aicore__ uint64_t tensor_addr(int64_t arg) {
    auto *tensor = reinterpret_cast<__gm__ simpler::tmr::Tensor *>(arg);
    return tensor->buffer.addr + tensor->start_offset * sizeof(float);
}
