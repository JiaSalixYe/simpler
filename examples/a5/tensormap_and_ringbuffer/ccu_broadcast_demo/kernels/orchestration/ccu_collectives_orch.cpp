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

extern "C" __attribute__((visibility("default"))) OrchestrationConfig pair_config(const ChipTaskArgs &) {
    return OrchestrationConfig{.expected_arg_count = 4};
}

extern "C" __attribute__((visibility("default"))) void pair_orch(const ChipTaskArgs &args) {
    CoreTaskArgs reduce;
    reduce.add_input(args.tensor(0).ref());
    reduce.add_output(args.tensor(1).ref());
    reduce.add_scalar(args.scalar(0));
    reduce.add_scalar(0);
    const auto first = rt_submit_aiv_task(0, reduce);

    CoreTaskArgsWithDeps<1> gather;
    gather.add_input(args.tensor(0).ref());
    gather.add_output(args.tensor(2).ref());
    gather.add_scalar(args.scalar(0));
    gather.add_scalar(1);
    gather.add_dep_wait(first.task_id());
    rt_submit_aiv_task(0, gather);
}
