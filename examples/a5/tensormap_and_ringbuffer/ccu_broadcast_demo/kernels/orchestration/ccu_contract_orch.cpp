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
#include "orchestration_api.h"
#include "../../ccu_contract_mailbox.h"

extern "C" {

__attribute__((visibility("default"))) OrchestrationConfig ccu_contract_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 6};
}

__attribute__((visibility("default"))) void ccu_contract_orchestration(const ChipTaskArgs &args) {
    const auto &src = args.tensor(0).ref();
    const auto &dst = args.tensor(1).ref();

    for (uint32_t round = 0; round < kCcuContractRounds; ++round) {
        const uint32_t size[1] = {round == 0 ? 16U : 32U};
        const uint32_t offset[1] = {round * 32U};
        const auto input = src.view(size, offset);
        const auto output = dst.view(size, offset);
        const auto mailbox = args.scalar(0) + round * sizeof(CcuContractMailbox);

        CoreTaskArgs produce;
        produce.add_output(input);
        produce.add_scalar(size[0]);
        produce.add_scalar(args.scalar(3) + offset[0]);
        rt_submit_aiv_task(3, produce);

        CoreTaskArgs submit;
        submit.add_input(input);
        submit.add_output(output);
        submit.add_scalar(mailbox);
        submit.add_scalar(args.scalar(1) + offset[0] * sizeof(float));
        submit.add_scalar(size[0] * sizeof(float));
        submit.add_scalar(args.scalar(2));
        rt_submit_aiv_task(0, submit);

        CoreTaskArgs consume;
        consume.add_input(output);
        consume.add_scalar(mailbox);
        rt_submit_aiv_task(1, consume);

        CoreTaskArgs gate;
        gate.add_scalar(mailbox);
        rt_submit_aiv_task(2, gate);
    }
}
}
