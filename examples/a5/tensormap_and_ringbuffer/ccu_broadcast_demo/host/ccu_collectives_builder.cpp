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

#include <stdexcept>

namespace {
namespace ccu = AscendC::ccu;
using simpler::ccu::BuilderContextV1;
constexpr uint64_t kBytes = 16 * sizeof(float);

void check(CcuResult result, const char *operation) {
    if (result != CCU_SUCCESS) throw std::runtime_error(operation);
}

void sync_peer(ChannelHandle channel, uint16_t phase) {
    check(ccu::NotifyRecord(channel, 0, phase), "record peer notification");
    check(ccu::NotifyWait(channel, 0, phase), "wait for peer notification");
}

struct Operands {
    ccu::LocalAddr src, dst;
    ccu::RemoteAddr peer_src;
    ccu::Variable dst_base, bytes;
    ccu::Event done;

    explicit Operands(const BuilderContextV1 &ctx) {
        ccu::Variable src_base, peer_src_base;
        src_base = ctx.local_window + ctx.tensors[0];
        peer_src_base = ctx.peer_window + ctx.tensors[0];
        dst_base = ctx.local_window + ctx.tensors[2];
        src.addr = src_base;
        src.token = ctx.local_token;
        peer_src.addr = peer_src_base;
        peer_src.token = ctx.peer_token;
        dst.addr = dst_base;
        dst.token = ctx.local_token;
        bytes = kBytes;
    }

    void wait(CcuResult result, const char *operation) {
        check(result, operation);
        check(ccu::EventWait(done), "wait for CCU data operation");
    }
};
}  // namespace

extern "C" CcuResult build_allreduce(const BuilderContextV1 *ctx) {
    Operands op(*ctx);
    ccu::Array<ccu::CcuBuffer> operands(2);
    ccu::CcuBuffer contiguous[]{operands[0], operands[1]};
    sync_peer(ctx->channel, 1);
    op.wait(ccu::LocalCopy(operands[0], op.src, op.bytes, op.done), "load local input");
    op.wait(ccu::Read(ctx->channel, operands[1], op.peer_src, op.bytes, op.done), "load peer input");
    op.wait(
        ccu::LocalReduce(contiguous, 2, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, op.bytes, op.done),
        "reduce local and peer inputs"
    );
    op.wait(ccu::LocalCopy(op.dst, operands[0], op.bytes, op.done), "store sum");
    sync_peer(ctx->channel, 2);
    return CCU_SUCCESS;
}

extern "C" CcuResult build_allgather(const BuilderContextV1 *ctx) {
    Operands op(*ctx);
    ccu::LocalAddr second;
    ccu::Variable second_base;
    second_base = op.dst_base + op.bytes;
    second.addr = second_base;
    second.token = ctx->local_token;
    ccu::CcuBuffer buffer;
    auto own = ctx->rank == 0 ? op.dst : second;
    auto peer = ctx->rank == 0 ? second : op.dst;
    sync_peer(ctx->channel, 1);
    op.wait(ccu::LocalCopy(buffer, op.src, op.bytes, op.done), "load local input");
    op.wait(ccu::LocalCopy(own, buffer, op.bytes, op.done), "store own gather slice");
    op.wait(ccu::Read(ctx->channel, buffer, op.peer_src, op.bytes, op.done), "load peer input");
    op.wait(ccu::LocalCopy(peer, buffer, op.bytes, op.done), "store peer gather slice");
    sync_peer(ctx->channel, 2);
    return CCU_SUCCESS;
}
