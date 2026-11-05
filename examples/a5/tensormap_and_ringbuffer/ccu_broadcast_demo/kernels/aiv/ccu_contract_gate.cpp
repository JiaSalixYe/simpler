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
    auto *m = reinterpret_cast<volatile __gm__ CcuContractMailbox *>(args[0]);
    uint64_t deadline = get_sys_cnt() + 10000000;
    do {
        mailbox_sync(&m->ready);
    } while (m->ready == 0 && get_sys_cnt() < deadline);

    // Bound the hold so that both the early and deferred consumer can make progress.
    deadline = get_sys_cnt() + 2000000;
    do {
        mailbox_sync(&m->observed);
    } while (m->observed == 0 && get_sys_cnt() < deadline);

    m->gate_saw_consumer = m->observed != 0;
    m->release = 1;
    mailbox_sync(&m->release);
    pipe_barrier(PIPE_ALL);
}
