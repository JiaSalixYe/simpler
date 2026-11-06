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

#include "service_mailbox.h"

#include "hcomm/ccu/ccu_control_flow_macro.h"
#include "hcomm/ccu/ccu_primitives.hpp"

namespace simpler::ccu_service {

// Called while recording the HCOMM program. The body appends one operation,
// including its cross-rank synchronization, and the loop executes it on CCU.
template <typename RecordBody>
CcuResult RecordService(const AscendC::ccu::Variable &base, RecordBody &&record_body) {
    namespace ccu = AscendC::ccu;
    ccu::Variable command, command_offset, completion_offset, command_addr, completion_addr, idle;
    command = kIdle;
    idle = kIdle;
    command_offset = offsetof(Mailbox, command);
    completion_offset = offsetof(Mailbox, completed);
    command_addr = base + command_offset;
    completion_addr = base + completion_offset;
    ccu::Array<ccu::Variable> request(1 + kPayloadWords);

    CCU_WHILE(command != kStop) {
        command = kIdle;
        CCU_WHILE(command == kIdle) {
            const auto ret = ccu::Load(command_addr, command);
            if (ret != CCU_SUCCESS) return ret;
        }
        auto ret = ccu::Load(base, request, 1 + kPayloadWords);
        if (ret != CCU_SUCCESS) return ret;
        CCU_IF(command == kExecute) {
            ret = record_body(request);
            if (ret != CCU_SUCCESS) return ret;
            ret = ccu::Store(command_addr, idle);
            if (ret != CCU_SUCCESS) return ret;
        }
        // Stop remains visible to device producers until the Host drains and reinitializes.
        ret = ccu::Store(completion_addr, request[0]);
        if (ret != CCU_SUCCESS) return ret;
    }
    return CCU_SUCCESS;
}

}  // namespace simpler::ccu_service
