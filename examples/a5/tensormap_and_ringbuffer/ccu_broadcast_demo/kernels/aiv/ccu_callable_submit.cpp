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
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "ccu_kernel_api.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *src = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *dst = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *domain = reinterpret_cast<__gm__ CommContext *>(args[2]);

    CcuCallArgs parameters;
    parameters.add_tensor(*src);
    parameters.add_tensor(*dst);
    parameters.add_scalar(static_cast<uint64_t>(args[3]));
    auto task = get_async_ctx(args);
    (void)send_request_entry(task, CcuCall(domain, 0, parameters));
}
