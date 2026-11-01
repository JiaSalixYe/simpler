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

#include "acl/acl.h"
#include "device_runner.h"
#include "runtime_c_api.h"

extern "C" int host_register_supported_ctx(DeviceContextHandle ctx, uint32_t *supported) {
    if (!ctx || !supported) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    *supported = 1;
    return 0;
}

extern "C" int host_register_map_ctx(
    DeviceContextHandle ctx, int device, uint64_t host, uint64_t bytes, uint64_t *mapped, uint32_t *registered
) {
    if (!ctx || !host || !bytes || !mapped || !registered) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    *mapped = 0;
    *registered = 0;
    try {
        auto *runner = static_cast<DeviceRunner *>(ctx);
        int rc = runner->ensure_acl_ready(device);
        if (rc != 0) return rc;
        auto *base = reinterpret_cast<void *>(host);
        rc = aclrtHostRegisterV2(base, static_cast<size_t>(bytes), ACL_HOST_REG_MAPPED);
        if (rc != 0) return rc;
        // Preserve the registration even if obtaining the device address fails:
        // the caller must be able to unregister it during cleanup.
        *registered = 1;
        void *address = nullptr;
        rc = aclrtHostGetDevicePointer(base, &address, 0);
        if (rc != 0) return rc;
        if (!address) return PTO_RUNTIME_ERR_INTERNAL;
        *mapped = reinterpret_cast<uint64_t>(address);
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

extern "C" int host_register_unmap_ctx(DeviceContextHandle ctx, int device, uint64_t host) {
    if (!ctx || !host) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    try {
        auto *runner = static_cast<DeviceRunner *>(ctx);
        int rc = runner->ensure_acl_ready(device);
        if (rc != 0) return rc;
        return aclrtHostUnregister(reinterpret_cast<void *>(host));
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

extern "C" int host_urma_capability_ctx(DeviceContextHandle ctx, uint32_t *supported) {
    if (!ctx || !supported) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // The A5 URMA transport is not qualified or not compiled. The transport
    // backend must report that explicitly; it must never fall back to
    // HostRegister or an A3 Fabric path.
    *supported = 0;
    return 0;
}
