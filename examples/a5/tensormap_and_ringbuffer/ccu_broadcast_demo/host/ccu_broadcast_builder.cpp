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
#include "ccu_builder.h"

#include "hcomm/ccu/ccu_control_flow_macro.h"

namespace {

namespace ccu = AscendC::ccu;
constexpr uint64_t kElements = 16;
constexpr uint64_t kBytes = kElements * sizeof(float);
constexpr uint64_t kInvalidShape = 2;

CcuResult copy_to_domain(const simpler::ccu::BuilderContextV1 &ctx) {
    ccu::LocalAddr source, local_dst;
    ccu::RemoteAddr remote_dst;
    ccu::Variable source_address, local_address, remote_address;
    source_address = ctx.local_window + ctx.tensors[0];
    local_address = ctx.local_window + ctx.tensors[2];
    remote_address = ctx.peer_window + ctx.tensors[2];
    source.addr = source_address;
    source.token = ctx.local_token;
    local_dst.addr = local_address;
    local_dst.token = ctx.local_token;
    remote_dst.addr = remote_address;
    remote_dst.token = ctx.peer_token;
    ccu::Variable length;
    length = kBytes;
    ccu::CcuBuffer buffer;
    ccu::Event copied;

    CcuResult result = ccu::LocalCopy(buffer, source, length, copied);
    if (result == CCU_SUCCESS) result = ccu::EventWait(copied);
    if (result == CCU_SUCCESS) result = ccu::LocalCopy(local_dst, buffer, length, copied);
    if (result == CCU_SUCCESS) result = ccu::EventWait(copied);
    if (result == CCU_SUCCESS) result = ccu::Write(ctx.channel, remote_dst, source, length, copied);
    if (result == CCU_SUCCESS) result = ccu::EventWait(copied);
    return result;
}

}  // namespace

extern "C" CcuResult build_broadcast(const simpler::ccu::BuilderContextV1 *ctx) {
    if (ctx == nullptr || ctx->abi_version != simpler::ccu::kBuilderAbiVersion) return CCU_E_PARA;
    auto status = ctx->status;
    status = kInvalidShape;
    auto result = ccu::NotifyRecord(ctx->channel, 0, 1);
    if (result == CCU_SUCCESS) result = ccu::NotifyWait(ctx->channel, 0, 1);
    if (result != CCU_SUCCESS) return result;
    auto tensor_count = ctx->tensor_count;
    auto scalar_count = ctx->scalar_count;
    CCU_IF(tensor_count == 2) {
        CCU_IF(scalar_count == 1) {
            CCU_IF(ctx->scalars[0] == kElements) {
                CCU_IF(ctx->tensors[1] == kBytes) {
                    CCU_IF(ctx->tensors[3] == kBytes) {
                        status = simpler::ccu::kSuccess;
                        if (ctx->rank == 0) result = copy_to_domain(*ctx);
                    }
                }
            }
        }
    }
    if (result == CCU_SUCCESS) result = ccu::NotifyRecord(ctx->channel, 0, 2);
    if (result == CCU_SUCCESS) result = ccu::NotifyWait(ctx->channel, 0, 2);
    return result;
}
