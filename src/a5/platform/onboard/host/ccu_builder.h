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

#include <cstdint>

#include "hcomm/ccu/ccu_primitives.hpp"
#include "platform_comm/ccu/call_protocol.h"

namespace simpler::ccu {

inline constexpr uint32_t kBuilderAbiVersion = 1;

// Borrowed while the Host records one device operation. The builder appends
// its data operations and inter-rank synchronization to the current program.
// The dispatcher owns the request loop and publishes completion after the body.
struct BuilderContextV1 {
    uint32_t abi_version;
    uint32_t rank;
    ChannelHandle channel;
    AscendC::ccu::Variable local_window;
    AscendC::ccu::Variable local_token;
    AscendC::ccu::Variable peer_window;
    AscendC::ccu::Variable peer_token;
    AscendC::ccu::Variable tensor_count;
    AscendC::ccu::Variable scalar_count;
    // Contiguous pairs of byte offset and byte length in the domain window.
    AscendC::ccu::Array<AscendC::ccu::Variable> &tensors;
    AscendC::ccu::Array<AscendC::ccu::Variable> &scalars;
    // Initially success for a matched ID. A builder can set a nonzero status
    // for a device-side error after ensuring the peer does not wait forever.
    AscendC::ccu::Variable status;
};

using BuilderV1 = CcuResult (*)(const BuilderContextV1 *);

}  // namespace simpler::ccu
