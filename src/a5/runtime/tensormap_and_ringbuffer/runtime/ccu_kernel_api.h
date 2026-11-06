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

#include "async_completion_kernel.h"
#include "platform_comm/ccu/call_protocol.h"
#include "tensor.h"

struct CommContext;
class CcuCallDescriptor;

class CcuCallArgs {
public:
    // Accept both a dispatched GM tensor descriptor and a local view of one.
    template <typename TensorView>
    __aicore__ void add_tensor(const TensorView &tensor) {
        using namespace simpler::ccu;
        if (tensor_count_ == kMaxTensors || scalar_count_ != 0 || tensor.address_space != AddressSpace::DEVICE ||
            !tensor.is_contiguous || tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS) {
            valid_ = false;
            return;
        }

        const uint64_t element_size = element_bytes(tensor.dtype);
        const uint64_t base = tensor.buffer.addr;
        const uint64_t capacity = tensor.buffer.size;
        if (element_size == 0 || base == 0 || capacity > UINT64_MAX - base ||
            tensor.start_offset > capacity / element_size) {
            valid_ = false;
            return;
        }
        const uint64_t start = tensor.start_offset * element_size;
        uint64_t bytes = element_size;
        for (uint32_t i = 0; i < tensor.ndims; ++i) {
            if (tensor.shapes[i] == 0 || bytes > (capacity - start) / tensor.shapes[i]) {
                valid_ = false;
                return;
            }
            bytes *= tensor.shapes[i];
        }
        tensors_[tensor_count_++] = {base + start, bytes};
    }

    template <typename T>
    __aicore__ void add_scalar(T value) {
        static_assert(
            is_supported_scalar_arg_v<T> && sizeof(T) <= sizeof(uint64_t),
            "CCU scalar must be an arithmetic or enum value of at most 64 bits"
        );
        if (scalar_count_ == simpler::ccu::kMaxScalars) {
            valid_ = false;
            return;
        }
        scalars_[scalar_count_++] = to_u64(value);
    }

private:
    struct TensorViewRange {
        uint64_t addr;
        uint64_t bytes;
    };

    static __aicore__ uint64_t element_bytes(DataType type) {
        // This table is used on AIV, where the host get_element_size()
        // implementation is not callable.
        switch (type) {
        case DataType::FLOAT32:
        case DataType::INT32:
        case DataType::UINT32:
            return 4;
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
        case DataType::INT16:
        case DataType::UINT16:
            return 2;
        case DataType::INT64:
        case DataType::UINT64:
            return 8;
        case DataType::INT8:
        case DataType::UINT8:
        case DataType::BOOL:
        case DataType::FP8E4M3FN:
        case DataType::FP8E8M0:
        case DataType::FP4E2M1:
            return 1;
        default:
            return 0;
        }
    }

    TensorViewRange tensors_[simpler::ccu::kMaxTensors]{};
    uint64_t scalars_[simpler::ccu::kMaxScalars]{};
    uint32_t tensor_count_{0};
    uint32_t scalar_count_{0};
    bool valid_{true};
    friend class CcuCallDescriptor;
};

namespace pto2::ccu_backend {

using simpler::ccu::Mailbox;

inline __aicore__ uint64_t load_word(volatile __gm__ uint64_t *ptr) {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    dcci(const_cast<__gm__ uint64_t *>(ptr), SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    return *ptr;
#else
    return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
#endif
}

inline __aicore__ void publish_word(volatile __gm__ uint64_t *ptr, uint64_t value) {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    *ptr = value;
    dcci(const_cast<__gm__ uint64_t *>(ptr), SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_DDR);
#else
    __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
#endif
}

inline __aicore__ void sync_operands(volatile __gm__ Mailbox *slot) {
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    pipe_barrier(PIPE_ALL);
    dcci(const_cast<__gm__ Mailbox *>(slot), ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
#else
    (void)slot;
#endif
}

inline __aicore__ bool fail(AsyncCtx &task) {
    if (task.completion_error_code != nullptr && *task.completion_error_code == SIMPLER_ERROR_NONE) {
        pto2::detail::defer_error(task, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
    pto2::detail::defer_flush(task);
    return false;
}

}  // namespace pto2::ccu_backend

class CcuCallDescriptor {
private:
    __aicore__ CcuCallDescriptor(const __gm__ CommContext *ctx, uint32_t id, const CcuCallArgs &args) :
        domain_ctx_(reinterpret_cast<uint64_t>(ctx)),
        id_(id),
        args_(args) {}

    __aicore__ bool
    post(AsyncCtx &task, bool defer, volatile __gm__ simpler::ccu::Mailbox *&slot, uint64_t &generation) const {
        using namespace simpler::ccu;
        using namespace pto2::ccu_backend;
        if (!args_.valid_ || !async_ctx_is_deferred(task) || domain_ctx_ == 0 || task.ccu_binding == 0 ||
            task.completion_error_code == nullptr || *task.completion_error_code != SIMPLER_ERROR_NONE) {
            return fail(task);
        }
        const auto *binding = reinterpret_cast<const __gm__ Binding *>(task.ccu_binding);
        const uint64_t base = binding->window_base;
        const uint64_t size = binding->window_size;
        if (binding->domain_ctx != domain_ctx_ || id_ >= binding->function_count || base == 0 || size == 0 ||
            size > UINT64_MAX - base || binding->mailbox == 0 || binding->mailbox % alignof(Mailbox) != 0) {
            return fail(task);
        }
        for (uint32_t i = 0; i < args_.tensor_count_; ++i) {
            const auto &view = args_.tensors_[i];
            if (view.addr < base || view.addr - base >= size || view.bytes > size - (view.addr - base)) {
                return fail(task);
            }
        }

        // The orchestration order dependency serializes all wrappers sharing
        // this slot. The command and generation checks also reject a second
        // send from one AIV while its previous request is still in flight.
        slot = reinterpret_cast<volatile __gm__ Mailbox *>(binding->mailbox);
        const uint64_t recycled = load_word(&slot->recycled);
        if (recycled >= UINT64_MAX - 1 || load_word(&slot->command) != kIdle ||
            load_word(&slot->completed) != recycled || load_word(&slot->generation) != recycled) {
            return fail(task);
        }
        generation = recycled + 1;
        if (defer) {
            const CompletionToken token{
                binding->mailbox, 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU_REQUEST, generation
            };
            if (!register_completion_condition(task, token)) return fail(task);
            pto2::detail::defer_flush(task);
        }

        sync_operands(slot);
        slot->generation = generation;
        slot->function_id = id_;
        slot->tensor_count = args_.tensor_count_;
        slot->scalar_count = args_.scalar_count_;
        // The native dispatcher reads a fixed-size snapshot. Clear the unused
        // entries so a shorter request cannot inherit a previous payload.
        for (uint32_t i = 0; i < kMaxTensors; ++i) {
            slot->tensors[i].offset = i < args_.tensor_count_ ? args_.tensors_[i].addr - base : 0;
            slot->tensors[i].bytes = args_.tensors_[i].bytes;
        }
        for (uint32_t i = 0; i < kMaxScalars; ++i) {
            slot->scalars[i] = args_.scalars_[i];
        }
        pto2::detail::defer_flush_range(slot, offsetof(Mailbox, command));
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
        dsb(DSB_DDR);
#endif
        publish_word(&slot->command, kExecute);
        return true;
    }

    uint64_t domain_ctx_;
    uint32_t id_;
    CcuCallArgs args_;

    friend __aicore__ CcuCallDescriptor CcuCall(const __gm__ CommContext *, uint32_t, const CcuCallArgs &);
    friend __aicore__ bool wait_request_entry(AsyncCtx &, const CcuCallDescriptor &);
    friend __aicore__ bool send_request_entry(AsyncCtx &, const CcuCallDescriptor &);
};

inline __aicore__ CcuCallDescriptor
CcuCall(const __gm__ CommContext *domain_ctx, uint32_t ccu_id, const CcuCallArgs &parameters) {
    return CcuCallDescriptor(domain_ctx, ccu_id, parameters);
}

inline __aicore__ bool send_request_entry(AsyncCtx &task, const CcuCallDescriptor &call) {
    volatile __gm__ simpler::ccu::Mailbox *slot = nullptr;
    uint64_t generation = 0;
    return call.post(task, true, slot, generation);
}

inline __aicore__ bool wait_request_entry(AsyncCtx &task, const CcuCallDescriptor &call) {
    using namespace simpler::ccu;
    using namespace pto2::ccu_backend;
    volatile __gm__ Mailbox *slot = nullptr;
    uint64_t generation = 0;
    if (!call.post(task, false, slot, generation)) return false;

#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    constexpr uint64_t kWaitTicks = 200000000;
    const uint64_t start = get_sys_cnt();
    while (static_cast<uint64_t>(get_sys_cnt()) - start < kWaitTicks) {
#else
    for (uint32_t attempt = 0; attempt < 1000000; ++attempt) {
#endif
        const uint64_t completed = load_word(&slot->completed);
        if (completed < generation) continue;
        if (completed != generation || load_word(&slot->status) != kSuccess || load_word(&slot->command) != kIdle ||
            load_word(&slot->generation) != generation || load_word(&slot->recycled) != generation - 1) {
            break;
        }
        sync_operands(slot);
        publish_word(&slot->recycled, generation);
        return true;
    }
    return fail(task);
}
