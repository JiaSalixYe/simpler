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

#include <cstddef>
#include <cstdint>

namespace simpler::ccu {

struct ModuleEntry {
    const void *binary;
    size_t binary_size;
    const char *symbol;
};

struct DomainConfig {
    int32_t device;
    uint32_t rank;
    uint64_t domain_ctx;
    uint64_t window_base;
    uint64_t window_size;
    // Unique to this collective prepare attempt, inside an owner-only directory.
    const char *rendezvous_prefix;
};

// Both ranks call Prepare concurrently. A non-null owner, even on failure,
// must be passed to Close once all participating Prepare calls have returned.
int Prepare(
    const DomainConfig &config, const ModuleEntry *functions, size_t function_count, void **owner,
    uint64_t *device_binding
) noexcept;

// Bracket each native run on this binding. End requires a positive device fence.
int Begin(uint64_t device_binding) noexcept;
int End(uint64_t device_binding) noexcept;

// Reports whether all runs and the resident service have safely stopped.
bool Drained(void *owner) noexcept;

// Consumes the owner on success. A failure retains the owner and its window lease.
int Close(void *owner) noexcept;
bool Available() noexcept;
const char *LastError() noexcept;

}  // namespace simpler::ccu
