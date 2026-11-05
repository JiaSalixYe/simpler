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

#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *m = reinterpret_cast<volatile __gm__ CcuContractMailbox *>(args[2]);
    if (args[5]) {
        auto task = get_async_ctx(args);
        save_expected_notification_counter(task, &m->done, 1);
    }
    m->input = tensor_addr(args[0]);
    m->output = tensor_addr(args[1]);
    m->remote_output = static_cast<uint64_t>(args[3]);
    m->bytes = static_cast<uint64_t>(args[4]);
    m->ready = 1;
    mailbox_sync(&m->ready);
    pipe_barrier(PIPE_ALL);
}
