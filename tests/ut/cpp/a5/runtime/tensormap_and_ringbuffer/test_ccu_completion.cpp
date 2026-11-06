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

#include <gtest/gtest.h>

#include <memory>

#include "ccu_kernel_api.h"
#include "platform_comm/comm_context.h"
#include "scheduler/scheduler.h"

namespace {

using simpler::ccu::Binding;
using simpler::ccu::kExecute;
using simpler::ccu::kIdle;
using simpler::ccu::Mailbox;

class CcuCompletionTest : public ::testing::Test {
protected:
    Mailbox mailbox{};
    CommContext domain{};
    CommContext other_domain{};
    alignas(64) float storage[32]{};
    Binding binding{
        reinterpret_cast<uint64_t>(&domain),
        reinterpret_cast<uint64_t>(storage),
        sizeof(storage),
        reinterpret_cast<uint64_t>(&mailbox),
        1,
        0
    };
    DeferredCompletionSlab slab{};
    AsyncCtx task{};
    Tensor src;
    Tensor dst;

    void SetUp() override {
        const uint32_t shape[] = {8};
        src.init_external(storage, sizeof(storage), shape, 1, DataType::FLOAT32, 0);
        dst = src;
        dst.start_offset = 8;
        task = task_context(true);
    }

    AsyncCtx task_context(bool eligible) {
        auto state = AsyncCtx::make(TaskId::make(0, 0), &slab);
        state.ccu_eligible = eligible;
        LocalContext local{0, 1, state};
        GlobalContext global{};
        global.ccu_binding = reinterpret_cast<uint64_t>(&binding);
        int64_t dispatch_args[PAYLOAD_GLOBAL_CONTEXT_INDEX + 1]{};
        dispatch_args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&local);
        dispatch_args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
        return get_async_ctx(dispatch_args);
    }

    CcuCallDescriptor copy(const CommContext &ctx, uint32_t id = 0) {
        CcuCallArgs args;
        args.add_tensor(src);
        args.add_tensor(dst);
        args.add_scalar(uint64_t{8});
        return CcuCall(&ctx, id, args);
    }
};

TEST_F(CcuCompletionTest, CopiesViewsAndRegistersFullGenerationBeforePublishing) {
    mailbox.generation = mailbox.completed = mailbox.recycled = uint64_t{1} << 32;
    const auto call = copy(domain);
    src.buffer.addr = 0;
    dst.start_offset = 0;
    ASSERT_TRUE(send_request_entry(task, call));
    EXPECT_EQ(mailbox.command, kExecute);
    EXPECT_EQ(mailbox.function_id, 0u);
    ASSERT_EQ(mailbox.tensor_count, 2u);
    EXPECT_EQ(mailbox.tensors[0].offset, 0u);
    EXPECT_EQ(mailbox.tensors[1].offset, 8 * sizeof(float));
    EXPECT_EQ(mailbox.tensors[1].bytes, 8 * sizeof(float));
    ASSERT_EQ(mailbox.scalar_count, 1u);
    EXPECT_EQ(mailbox.scalars[0], 8u);
    ASSERT_EQ(slab.count, 1u);
    EXPECT_EQ(slab.entries[0].backend_cookie, (uint64_t{1} << 32) + 1);
    EXPECT_EQ(slab.entries[0].completion_type, COMPLETION_TYPE_CCU_REQUEST);

    EXPECT_FALSE(send_request_entry(task, call));
    EXPECT_EQ(slab.count, 1u);
    EXPECT_EQ(mailbox.generation, (uint64_t{1} << 32) + 1);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_F(CcuCompletionTest, RejectsWrongDomainAndViewsOutsideTheWindow) {
    EXPECT_FALSE(send_request_entry(task, copy(other_domain)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    slab.error_code = SIMPLER_ERROR_NONE;
    EXPECT_FALSE(send_request_entry(task, copy(domain, 1)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    slab.error_code = SIMPLER_ERROR_NONE;

    // Still within the allocation described by Tensor, but crosses the
    // prepared binding's smaller window.
    binding.window_size = 12 * sizeof(float);
    EXPECT_FALSE(send_request_entry(task, copy(domain)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(slab.count, 0u);
    EXPECT_EQ(mailbox.command, kIdle);
    EXPECT_EQ(mailbox.generation, 0u);
}

TEST_F(CcuCompletionTest, IneligibleDispatchCannotUseTheBinding) {
    task = task_context(false);
    EXPECT_FALSE(send_request_entry(task, copy(domain)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(mailbox.command, kIdle);
}

TEST_F(CcuCompletionTest, RegistrationOverflowDoesNotPost) {
    slab.count = task.completion_capacity;
    EXPECT_FALSE(send_request_entry(task, copy(domain)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_EQ(mailbox.command, kIdle);
    EXPECT_EQ(mailbox.generation, 0u);
}

TEST_F(CcuCompletionTest, WaitTimeoutPreservesTheOutstandingRequest) {
    EXPECT_FALSE(wait_request_entry(task, copy(domain)));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(slab.count, 0u);
    EXPECT_EQ(mailbox.command, kExecute);
    EXPECT_EQ(mailbox.generation, 1u);
    EXPECT_EQ(mailbox.recycled, 0u);
}

TEST_F(CcuCompletionTest, ExactSuccessWaitsForNormalDoneBeforeRecycling) {
    ASSERT_TRUE(send_request_entry(task, copy(domain)));
    auto sched = std::make_unique<SchedulerState>();
    auto &waits = sched->async_wait_list;
    auto &entry = waits.entries[0];
    ChipTaskSlotState producer{};
    TaskPayload payload{};
    producer.task_state.store(CHIP_TASK_PENDING);
    producer.active_mask = LaneMask::aiv0();
    producer.total_required_subtasks = 1;
    producer.logical_block_num = 1;
    producer.fanout_count = 1;
    producer.payload = &payload;
    entry.slot_state = &producer;
    entry.task_token = task.task_token;
    waits.count = 1;
    int32_t error = SIMPLER_ERROR_NONE;
    ASSERT_TRUE(waits.append_condition_locked(
        entry, slab.entries[0].addr, slab.entries[0].backend_cookie, slab.entries[0].expected_value, ASYNC_ENGINE_CCU,
        COMPLETION_TYPE_CCU_REQUEST, error
    ));
    ChipTaskSlotState *releases[1]{};
    int32_t release_count = 0;
    auto poll = [&] {
        return waits.poll_and_complete<false>(nullptr, sched.get(), releases, release_count, 1);
    };

    EXPECT_EQ(poll().error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(entry.waiting_completion_count, 1);
    EXPECT_EQ(mailbox.recycled, 0u);
    mailbox.command = kIdle;
    mailbox.completed = mailbox.generation;
    EXPECT_EQ(poll().completed, 0);
    EXPECT_EQ(entry.waiting_completion_count, 0);
    EXPECT_EQ(mailbox.recycled, 0u);

    entry.normal_done = true;
    EXPECT_EQ(poll().completed, 1);
    EXPECT_EQ(releases[0], &producer);
    EXPECT_EQ(mailbox.recycled, 1u);
    EXPECT_TRUE(send_request_entry(task, copy(domain)));
    EXPECT_EQ(mailbox.generation, 2u);
}

TEST_F(CcuCompletionTest, AForeignOrFailedGenerationCannotCompleteTheWait) {
    const uint64_t address = reinterpret_cast<uint64_t>(&mailbox);
    mailbox.generation = 1;
    EXPECT_EQ(pto2::ccu_backend::poll_request(address, 1).state, CompletionPollState::PENDING);
    mailbox.completed = 2;
    EXPECT_EQ(pto2::ccu_backend::poll_request(address, 1).state, CompletionPollState::FAILED);
    mailbox.completed = 1;
    mailbox.status = 1;
    EXPECT_EQ(pto2::ccu_backend::poll_request(address, 1).state, CompletionPollState::FAILED);
    EXPECT_EQ(mailbox.recycled, 0u);
}

}  // namespace
