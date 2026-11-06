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
#include "../../ccu_service_probe.h"
#include "platform_comm/ccu/service_device.h"

namespace {

constexpr uint64_t kWaitTicks = 200000000;

__aicore__ void flush_data(volatile __gm__ float *data) {
    for (uint32_t i = 0; i < kCcuServiceFloats; i += 16) {
        mailbox_sync(reinterpret_cast<volatile __gm__ uint64_t *>(data + i));
    }
}

}  // namespace

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<volatile __gm__ float *>(tensor_addr(args[0]));
    auto *output = reinterpret_cast<volatile __gm__ float *>(tensor_addr(args[1]));
    auto &probe = *reinterpret_cast<volatile __gm__ CcuServiceProbe *>(args[2]);
    const auto remote_output = static_cast<uint64_t>(args[3]);
    const auto rounds = static_cast<uint32_t>(args[4]);
    const auto local_seed = static_cast<uint32_t>(args[5]);
    const auto root_seed = static_cast<uint32_t>(args[6]);

    namespace service = simpler::ccu_service;
    uint64_t failure = 0;
    uint64_t verified = 0;
    bool protocol_ok = true;
    for (uint32_t round = 0; round < rounds; ++round) {
        const bool odd = round % 2 != 0;
        const uint32_t offset = odd ? kCcuServiceOffset : 0;
        const uint32_t count = odd ? kCcuServiceLongCount : kCcuServiceShortCount;
        const uint32_t seed = round * kCcuServiceFloats;
        for (uint32_t i = 0; i < kCcuServiceFloats; ++i) {
            input[i] =
                i >= offset && i < offset + count ? static_cast<float>(local_seed + seed + i) : kCcuServiceSentinel;
            output[i] = kCcuServiceSentinel;
        }
        flush_data(input);
        flush_data(output);

        const uint64_t byte_offset = offset * sizeof(float);
        const uint64_t payload[service::kPayloadWords] = {
            reinterpret_cast<uint64_t>(input) + byte_offset, reinterpret_cast<uint64_t>(output) + byte_offset,
            remote_output + byte_offset, count * sizeof(float)
        };
        const uint64_t generation = round + 1;
        if (!service::TrySubmit(probe.slot, generation, service::kExecute, payload) ||
            !service::Wait(probe.slot, generation, kWaitTicks)) {
            failure = 1;
            protocol_ok = false;
            break;
        }
        flush_data(output);
        bool result_ok = true;
        for (uint32_t i = 0; i < kCcuServiceFloats; ++i) {
            const float expected =
                i >= offset && i < offset + count ? static_cast<float>(root_seed + seed + i) : kCcuServiceSentinel;
            if (output[i] != expected) {
                failure = 2;
                result_ok = false;
                break;
            }
        }
        // Keep the paired ranks in the same protocol even after a numerical error.
        verified += result_ok;
    }

    const uint64_t empty[service::kPayloadWords] = {};
    if (protocol_ok) {
        // Replaying the last generation must leave the idle slot unchanged.
        if (rounds != 0 && service::TrySubmit(probe.slot, rounds, service::kStop, empty)) {
            failure = 4;
        } else if (!service::TrySubmit(probe.slot, rounds + 1, service::kStop, empty) ||
                   !service::Wait(probe.slot, rounds + 1, kWaitTicks)) {
            failure = 3;
        } else if (service::TrySubmit(probe.slot, rounds + 2, service::kStop, empty)) {
            failure = 5;
        }
    }
    probe.verified = verified;
    probe.failure = failure;
    mailbox_sync(&probe.verified);
    pipe_barrier(PIPE_ALL);
}
