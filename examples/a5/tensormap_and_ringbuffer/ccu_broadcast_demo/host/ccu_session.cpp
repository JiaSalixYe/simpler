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
#include "ccu_session.h"
#include "../ccu_contract_mailbox.h"
#include "../ccu_service_probe.h"
#include "platform_comm/ccu/service_program.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_control_flow_macro.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_primitives.hpp"
#include "hcomm/ccu/ccu_res.h"

namespace {

constexpr uint32_t kRanks = 2;
constexpr uint32_t kMaxCount = 1024;
constexpr uint32_t kGuardCount = 32;
constexpr uint32_t kOutputOffset = kMaxCount + 2 * kGuardCount;
constexpr uint32_t kWindowFloats = 2 * kMaxCount + 3 * kGuardCount;
constexpr float kSentinel = -987654.0f;
constexpr uint32_t kTriggered = 0x43435531;
thread_local std::string last_error;

void Check(int rc, const char *operation) {
    if (rc != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}

template <typename Fn>
int Protect(Fn &&fn) noexcept {
    last_error.clear();
    try {
        std::forward<Fn>(fn)();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
    } catch (...) {
        last_error = "Unknown CCU session failure";
    }
    return -1;
}

float InputValue(uint32_t rank, uint32_t iteration, uint32_t index) {
    return static_cast<float>(rank * 10000 + iteration * 1000 + index);
}

CcuResult ChannelDieProbe(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }

uint8_t FindChannelDie(ChannelHandle channel) {
    const void *arguments[] = {&channel};
    for (uint32_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle descriptor = 0;
        Check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate");
        // Query validates channel placement without registering or launching a task.
        const CcuResult query =
            HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(ChannelDieProbe), arguments, 1, descriptor);
        Check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy");
        if (query == CCU_SUCCESS) return static_cast<uint8_t>(die);
        if (query != CCU_E_PARA && query != CCU_E_UNAVAIL) Check(query, "HcommCcuKernelQueryResReq(channel die)");
    }
    throw std::runtime_error("CCU channel has no enabled matching IO die");
}

struct BroadcastArgs {
    ChannelHandle channel;
    CcuEventHandle gate;
    bool is_root;
};

// This Host function records a CCU instruction program during registration.
CcuResult BroadcastProgram(const BroadcastArgs *args) {
    namespace ccu = AscendC::ccu;
    try {
        ccu::Variable input, output, token, output_token, peer_output, peer_token, length;
        const std::array<ccu::Variable, 7> variables{input,       output,     token, output_token,
                                                     peer_output, peer_token, length};
        for (uint32_t i = 0; i < variables.size(); ++i)
            Check(ccu::LoadArg(variables[i], i), "ccu::LoadArg");
        ccu::Event gate(args->gate);
        Check(ccu::EventWait(gate), "ccu::EventWait(AIV gate)");
        Check(ccu::NotifyRecord(args->channel, 0, 1), "ccu::NotifyRecord(ready)");
        Check(ccu::NotifyWait(args->channel, 0, 1), "ccu::NotifyWait(ready)");
        if (args->is_root) {
            ccu::LocalAddr source, destination;
            ccu::RemoteAddr remote;
            source.addr = input;
            source.token = token;
            destination.addr = output;
            destination.token = output_token;
            remote.addr = peer_output;
            remote.token = peer_token;
            ccu::Event copied;
            ccu::CcuBuffer buffer;
            Check(ccu::LocalCopy(buffer, source, length, copied), "ccu::LocalCopy(HBM to CCU buffer)");
            Check(ccu::EventWait(copied), "ccu::EventWait(local read)");
            Check(ccu::LocalCopy(destination, buffer, length, copied), "ccu::LocalCopy(CCU buffer to HBM)");
            Check(ccu::EventWait(copied), "ccu::EventWait(local write)");
            Check(ccu::Write(args->channel, remote, source, length, copied), "ccu::Write(peer HBM)");
            Check(ccu::EventWait(copied), "ccu::EventWait(remote write)");
        }
        Check(ccu::NotifyRecord(args->channel, 0, 2), "ccu::NotifyRecord(done)");
        Check(ccu::NotifyWait(args->channel, 0, 2), "ccu::NotifyWait(done)");
        return CCU_SUCCESS;
    } catch (const std::exception &error) {
        last_error = error.what();
        return CCU_E_INTERNAL;
    }
}

void RecordMailboxBroadcast(
    const BroadcastArgs &args, const AscendC::ccu::Variable &source_token,
    const AscendC::ccu::Variable &destination_token, const AscendC::ccu::Variable &peer_token,
    const AscendC::ccu::Variable &input, const AscendC::ccu::Variable &output,
    const AscendC::ccu::Variable &peer_output, const AscendC::ccu::Variable &bytes
) {
    namespace ccu = AscendC::ccu;
    Check(ccu::NotifyRecord(args.channel, 0, 1), "mailbox notify ready");
    Check(ccu::NotifyWait(args.channel, 0, 1), "mailbox wait ready");
    if (args.is_root) {
        ccu::LocalAddr source, destination;
        ccu::RemoteAddr remote;
        source.addr = input;
        source.token = source_token;
        destination.addr = output;
        destination.token = destination_token;
        remote.addr = peer_output;
        remote.token = peer_token;
        ccu::Event copied;
        ccu::CcuBuffer buffer;
        Check(ccu::LocalCopy(buffer, source, bytes, copied), "mailbox local read");
        Check(ccu::EventWait(copied), "mailbox local read wait");
        Check(ccu::LocalCopy(destination, buffer, bytes, copied), "mailbox local write");
        Check(ccu::EventWait(copied), "mailbox local write wait");
        Check(ccu::Write(args.channel, remote, source, bytes, copied), "mailbox remote write");
        Check(ccu::EventWait(copied), "mailbox remote write wait");
    }
    Check(ccu::NotifyRecord(args.channel, 0, 2), "mailbox notify done");
    Check(ccu::NotifyWait(args.channel, 0, 2), "mailbox wait done");
}

CcuResult ServiceBroadcastProgram(const BroadcastArgs *args) {
    namespace ccu = AscendC::ccu;
    try {
        ccu::Variable base, source_token, destination_token, peer_token;
        Check(ccu::LoadArg(base, 0), "service mailbox");
        Check(ccu::LoadArg(source_token, 1), "source token");
        Check(ccu::LoadArg(destination_token, 2), "destination token");
        Check(ccu::LoadArg(peer_token, 3), "peer token");
        return simpler::ccu_service::RecordService(base, [&](ccu::Array<ccu::Variable> &request) {
            RecordMailboxBroadcast(
                *args, source_token, destination_token, peer_token, request[1], request[2], request[3], request[4]
            );
            return CCU_SUCCESS;
        });
    } catch (const std::exception &error) {
        last_error = error.what();
        return CCU_E_INTERNAL;
    }
}

CcuResult ResidentBroadcastProgram(const BroadcastArgs *args) {
    namespace ccu = AscendC::ccu;
    try {
        ccu::Variable cursor, source_token, destination_token, peer_token;
        Check(ccu::LoadArg(cursor, 0), "resident mailbox");
        Check(ccu::LoadArg(source_token, 1), "source token");
        Check(ccu::LoadArg(destination_token, 2), "destination token");
        Check(ccu::LoadArg(peer_token, 3), "peer token");

        ccu::Variable one, completed, record_bytes, release_offset, done_offset;
        one = 1;
        completed = 0;
        record_bytes = sizeof(CcuContractMailbox);
        release_offset = offsetof(CcuContractMailbox, release);
        done_offset = offsetof(CcuContractMailbox, done);
        ccu::Variable release_addr, done_addr, ready, released;
        ccu::Array<ccu::Variable> request(5);

        CCU_WHILE(completed != kCcuContractRounds) {
            ready = 0;
            CCU_WHILE(ready != 1) { Check(ccu::Load(cursor, ready), "resident wait request"); }
            Check(ccu::Load(cursor, request, 5), "resident load request");
            release_addr = cursor + release_offset;
            released = 0;
            CCU_WHILE(released != 1) { Check(ccu::Load(release_addr, released), "resident wait release"); }
            RecordMailboxBroadcast(
                *args, source_token, destination_token, peer_token, request[1], request[2], request[3], request[4]
            );
            done_addr = cursor + done_offset;
            Check(ccu::Store(done_addr, one), "resident store completion");
            cursor += record_bytes;
            completed += one;
        }
        return CCU_SUCCESS;
    } catch (const std::exception &error) {
        last_error = error.what();
        return CCU_E_INTERNAL;
    }
}

struct Session {
    int device = -1;
    uint32_t rank = 0;
    uint32_t count = 0;
    uint32_t iteration = 0;
    uint32_t root = 0;
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    ThreadHandle thread = 0;
    CcuInsHandle instance = 0;
    CcuEventHandle gate = 0;
    CcuKernelHandle kernel = 0;
    uint64_t gate_va = 0;
    void *window = nullptr;
    void *separate_output = nullptr;
    bool borrowed_data = false;
    void *trigger_status = nullptr;
    void *mailboxes = nullptr;
    bool service = false;
    CcuPeerInfo peer{};
    std::vector<ChannelHandle> channels;
    bool launched = false;
    bool drained = false;

    int Close() {
        if (launched && !drained) throw std::runtime_error("CCU session has no proven completion; owners retained");
        Check(aclrtSetDevice(device), "aclrtSetDevice(close)");
        if (comm != nullptr) {
            Check(HcclCommDestroy(comm), "HcclCommDestroy");
            comm = nullptr;
        }
        if (stream != nullptr) {
            Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
            stream = nullptr;
        }
        if (trigger_status != nullptr) {
            Check(aclrtFree(trigger_status), "aclrtFree(trigger_status)");
            trigger_status = nullptr;
        }
        if (mailboxes != nullptr) {
            Check(aclrtFree(mailboxes), "aclrtFree(mailboxes)");
            mailboxes = nullptr;
        }
        if (!borrowed_data && separate_output != nullptr) {
            Check(aclrtFree(separate_output), "aclrtFree(separate_output)");
            separate_output = nullptr;
        }
        if (!borrowed_data && window != nullptr) {
            Check(aclrtFree(window), "aclrtFree(window)");
            window = nullptr;
        }
        return 0;
    }
};

void InitializeData(Session &session) {
    constexpr size_t bytes = kWindowFloats * sizeof(float);
    std::vector<float> initial(kWindowFloats, kSentinel);
    if (session.separate_output != nullptr) {
        Check(
            aclrtMemcpy(session.separate_output, bytes, initial.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(plain output H2D)"
        );
    }
    for (uint32_t i = 0; i < session.count; ++i)
        initial[kGuardCount + i] = InputValue(session.rank, session.iteration, i);
    Check(
        aclrtMemcpy(session.window, bytes, initial.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(input H2D)"
    );
    const uint32_t status = 0;
    Check(
        aclrtMemcpy(session.trigger_status, sizeof(status), &status, sizeof(status), ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(status H2D)"
    );
}

Session &Get(void *handle) {
    if (handle == nullptr) throw std::invalid_argument("CCU session is null");
    return *static_cast<Session *>(handle);
}

void AcquireChannels(Session &session) {
    std::vector<HcclChannelDesc> requests;
    const uint32_t remote = 1 - session.rank;
    CommLink *links = nullptr;
    uint32_t link_count = 0;
    Check(HcclRankGraphGetLinks(session.comm, 0, session.rank, remote, &links, &link_count), "HcclRankGraphGetLinks");
    for (uint32_t i = 0; i < link_count; ++i) {
        if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
        HcclChannelDesc desc;
        Check(HcclChannelDescInit(&desc, 1), "HcclChannelDescInit");
        desc.remoteRank = remote;
        desc.notifyNum = 4;
        desc.channelProtocol = links[i].linkAttr.linkProtocol;
        desc.localEndpoint = links[i].srcEndpointDesc;
        desc.remoteEndpoint = links[i].dstEndpointDesc;
        requests.push_back(desc);
        break;
    }
    if (requests.empty()) throw std::runtime_error("No UBC_CTP link to peer; CCU Broadcast requires UB connectivity");
    session.channels.resize(requests.size());
    Check(
        HcclChannelAcquire(session.comm, COMM_ENGINE_CCU, requests.data(), requests.size(), session.channels.data()),
        "HcclChannelAcquire(CCU)"
    );
}

}  // namespace

extern "C" const char *ccu_demo_last_error() { return last_error.c_str(); }
extern "C" size_t ccu_demo_root_info_size() { return sizeof(HcclRootInfo); }

extern "C" int ccu_demo_get_root_info(int device, void *root_info, size_t size) {
    return Protect([&] {
        if (root_info == nullptr || size != sizeof(HcclRootInfo)) throw std::invalid_argument("Root-info buffer size");
        Check(aclrtSetDevice(device), "aclrtSetDevice(root)");
        Check(HcclGetRootInfo(static_cast<HcclRootInfo *>(root_info)), "HcclGetRootInfo");
    });
}

extern "C" int ccu_demo_open(
    int device, uint32_t rank, const void *root_info, size_t root_info_size, uint32_t count, uint32_t iteration,
    bool plain_gm, uint64_t external_input, uint64_t external_output, void **handle, CcuPeerInfo *peer
) {
    return Protect([&] {
        if (root_info == nullptr || root_info_size != sizeof(HcclRootInfo) || handle == nullptr || *handle != nullptr ||
            peer == nullptr || rank >= kRanks || count == 0 || count > kMaxCount || iteration > 10000 ||
            ((external_input != 0 || external_output != 0) &&
             (!plain_gm || external_input == 0 || external_output == 0 || external_input == external_output))) {
            throw std::invalid_argument("Invalid CCU session arguments");
        }
        *handle = nullptr;
        auto session = std::make_unique<Session>();
        session->device = device;
        session->rank = rank;
        session->count = count;
        session->iteration = iteration;
        Check(aclrtSetDevice(device), "aclrtSetDevice(open)");
        try {
            HcclRootInfo root{};
            std::memcpy(&root, root_info, sizeof(root));
            Check(HcclCommInitRootInfo(kRanks, &root, rank, &session->comm), "HcclCommInitRootInfo");
            Check(aclrtCreateStream(&session->stream), "aclrtCreateStream(CCU)");
            Check(
                HcclThreadAcquireWithStream(session->comm, COMM_ENGINE_CCU, session->stream, 1, &session->thread),
                "HcclThreadAcquireWithStream(CCU)"
            );
            AcquireChannels(*session);

            constexpr size_t bytes = kWindowFloats * sizeof(float);
            if (external_input != 0) {
                session->borrowed_data = true;
                session->window = reinterpret_cast<void *>(external_input);
                session->separate_output = reinterpret_cast<void *>(external_output);
            } else if (plain_gm) {
                Check(aclrtMalloc(&session->window, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(plain input)");
                Check(
                    aclrtMalloc(&session->separate_output, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                    "aclrtMalloc(plain output)"
                );
            } else {
                aclrtMallocAttrValue value{};
                value.moduleId = 3;
                aclrtMallocAttribute attribute{ACL_RT_MEM_ATTR_MODULE_ID, value};
                aclrtMallocConfig config{&attribute, 1};
                Check(
                    aclrtMallocWithCfg(&session->window, bytes, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &config),
                    "aclrtMallocWithCfg(CCU window)"
                );
            }
            Check(
                aclrtMalloc(&session->trigger_status, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(trigger_status)"
            );
            InitializeData(*session);
            const auto input_base = reinterpret_cast<uint64_t>(session->window);
            const auto output_base = plain_gm ? reinterpret_cast<uint64_t>(session->separate_output) : input_base;
            session->peer.input = input_base + kGuardCount * sizeof(float);
            session->peer.output = output_base + kOutputOffset * sizeof(float);
            Check(HcommCcuGetMemToken(input_base, bytes, &session->peer.token), "HcommCcuGetMemToken(input)");
            session->peer.output_token = session->peer.token;
            if (plain_gm) {
                Check(
                    HcommCcuGetMemToken(output_base, bytes, &session->peer.output_token), "HcommCcuGetMemToken(output)"
                );
            }
            session->peer.trigger_status = reinterpret_cast<uint64_t>(session->trigger_status);
            *peer = session->peer;
            *handle = session.release();
        } catch (...) {
            session->Close();
            throw;
        }
    });
}

extern "C" int ccu_demo_prepare(
    void *handle, const CcuPeerInfo *peers, size_t peer_count, uint32_t root, uint64_t *gate_va, uint32_t *gate_mask
) {
    return Protect([&] {
        auto &session = Get(handle);
        if (peers == nullptr || peer_count != kRanks || root >= kRanks || gate_va == nullptr || gate_mask == nullptr ||
            session.launched) {
            throw std::invalid_argument("Invalid CCU prepare arguments");
        }
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(prepare)");
        const auto &local = peers[session.rank];
        const auto &remote = peers[1 - session.rank];
        if (local.input != session.peer.input || local.output != session.peer.output ||
            local.token != session.peer.token || local.output_token != session.peer.output_token ||
            remote.output == 0) {
            throw std::invalid_argument("Peer metadata does not match this rank");
        }
        if (session.kernel == 0) {
            uint32_t instance_count = 0;
            Check(HcclCommQueryCcuIns(session.comm, &session.instance, &instance_count), "HcclCommQueryCcuIns");
            if (instance_count != 1) throw std::runtime_error("Expected one communicator-owned CCU instance");
            const uint8_t die = FindChannelDie(session.channels.at(0));
            Check(HcommCcuEventAlloc(session.instance, die, 1, &session.gate), "HcommCcuEventAlloc(AIV gate)");
            Check(HcommCcuEventGetAddr(session.gate, 0, &session.gate_va), "HcommCcuEventGetAddr(AIV gate)");
            if (session.gate_va == 0) throw std::runtime_error("HCOMM returned an empty CKE mapping");
            const BroadcastArgs program{session.channels.at(0), session.gate, session.rank == root};
            const void *program_args[] = {&program};
            Check(HcommCcuKernelRegisterStart(session.instance), "HcommCcuKernelRegisterStart");
            Check(
                HcommCcuKernelRegister(
                    session.instance, die, "SimplerAivGatedBroadcast", reinterpret_cast<const void *>(BroadcastProgram),
                    program_args, 1, &session.kernel
                ),
                "HcommCcuKernelRegister"
            );
            Check(HcommCcuKernelRegisterEnd(session.instance), "HcommCcuKernelRegisterEnd");
            session.root = root;
            std::fprintf(stderr, "CCU_REGISTER rank=%u io_die=%u protocol=UBC_CTP\n", session.rank, die);
        }
        if (session.root != root) throw std::invalid_argument("Root is fixed by the registered program");
        *gate_va = session.gate_va;
        *gate_mask = 1;
        const std::array<uint64_t, 7> task{
            local.input,
            local.output,
            local.token,
            local.output_token,
            remote.output,
            remote.output_token,
            session.count * sizeof(float)
        };
        uint32_t task_count = 0;
        Check(HcommCcuGetTaskArgsNum(session.kernel, &task_count), "HcommCcuGetTaskArgsNum");
        if (task_count != task.size()) throw std::runtime_error("Registered CCU task argument count differs");
        session.launched = true;
        Check(HcommCcuKernelLaunch(session.thread, session.kernel, task.data(), task.size()), "HcommCcuKernelLaunch");
    });
}

extern "C" int ccu_demo_wait_and_verify(void *handle, uint32_t root, int timeout_ms) {
    return Protect([&] {
        auto &session = Get(handle);
        if (!session.launched || root != session.root || timeout_ms <= 0) {
            throw std::invalid_argument("Invalid CCU completion arguments");
        }
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(verify)");
        uint32_t status = 0;
        Check(
            aclrtMemcpy(&status, sizeof(status), session.trigger_status, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(trigger proof)"
        );
        if (status != kTriggered) throw std::runtime_error("simpler AIV trigger did not publish its completion marker");
        Check(aclrtSynchronizeStreamWithTimeout(session.stream, timeout_ms), "CCU stream completion");
        session.drained = true;
        std::vector<float> actual(kWindowFloats);
        Check(
            aclrtMemcpy(
                actual.data(), actual.size() * sizeof(float), session.window, actual.size() * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(result D2H)"
        );
        std::vector<float> separate_output;
        if (session.separate_output != nullptr) {
            separate_output.resize(kWindowFloats);
            const auto bytes = separate_output.size() * sizeof(float);
            Check(
                aclrtMemcpy(separate_output.data(), bytes, session.separate_output, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(plain output D2H)"
            );
        }
        for (uint32_t i = 0; i < kWindowFloats; ++i) {
            float expected = kSentinel;
            if (i >= kGuardCount && i < kGuardCount + session.count) {
                expected = InputValue(session.rank, session.iteration, i - kGuardCount);
            }
            const float expected_output = i >= kOutputOffset && i < kOutputOffset + session.count ?
                                              InputValue(root, session.iteration, i - kOutputOffset) :
                                              kSentinel;
            if (separate_output.empty() && expected_output != kSentinel) {
                expected = expected_output;
            }
            if (!separate_output.empty() && separate_output[i] != expected_output) {
                throw std::runtime_error("Plain output result/guard mismatch at float " + std::to_string(i));
            }
            if (actual[i] != expected) {
                throw std::runtime_error(
                    "Broadcast result/guard mismatch at float " + std::to_string(i) + ": expected " +
                    std::to_string(expected) + ", got " + std::to_string(actual[i])
                );
            }
        }
    });
}

extern "C" int ccu_demo_prepare_resident(
    void *handle, const CcuPeerInfo *peers, size_t peer_count, uint32_t root, bool service, uint64_t *mailboxes
) {
    return Protect([&] {
        auto &session = Get(handle);
        if (peers == nullptr || peer_count != kRanks || root >= kRanks || mailboxes == nullptr || session.launched ||
            session.separate_output == nullptr || session.count != 64) {
            throw std::invalid_argument("Resident test requires separate GM allocations and 64 input floats");
        }
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(resident prepare)");
        const size_t bytes = service ? sizeof(CcuServiceProbe) : sizeof(CcuContractMailbox) * kCcuContractRounds;
        if (session.kernel == 0) {
            Check(aclrtMalloc(&session.mailboxes, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(mailboxes)");
            uint32_t instance_count = 0;
            Check(HcclCommQueryCcuIns(session.comm, &session.instance, &instance_count), "HcclCommQueryCcuIns");
            if (instance_count != 1) throw std::runtime_error("Expected one communicator-owned CCU instance");
            const uint8_t die = FindChannelDie(session.channels.at(0));
            const BroadcastArgs program{session.channels.at(0), 0, session.rank == root};
            const void *program_args[] = {&program};
            Check(HcommCcuKernelRegisterStart(session.instance), "resident register start");
            const auto build_program = service ? ServiceBroadcastProgram : ResidentBroadcastProgram;
            const char *name = service ? "SimplerServiceBroadcast" : "SimplerResidentBroadcast";
            const auto ret = HcommCcuKernelRegister(
                session.instance, die, name, reinterpret_cast<const void *>(build_program), program_args, 1,
                &session.kernel
            );
            if (ret != CCU_SUCCESS && !last_error.empty()) throw std::runtime_error(last_error);
            Check(ret, "resident register");
            Check(HcommCcuKernelRegisterEnd(session.instance), "resident register end");
            session.root = root;
            session.service = service;
            std::fprintf(stderr, "CCU_RESIDENT_REGISTER rank=%u io_die=%u service=%d\n", session.rank, die, service);
        }
        if (session.root != root || session.service != service || session.mailboxes == nullptr)
            throw std::invalid_argument("Incompatible program");
        const std::vector<float> empty_input(kWindowFloats, kSentinel);
        const auto input_bytes = empty_input.size() * sizeof(float);
        Check(
            aclrtMemcpy(session.window, input_bytes, empty_input.data(), input_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
            "resident initialize producer output"
        );
        Check(aclrtMemset(session.mailboxes, bytes, 0, bytes), "aclrtMemset(mailboxes)");
        *mailboxes = reinterpret_cast<uint64_t>(session.mailboxes);
        const std::array<uint64_t, 4> task{
            *mailboxes, session.peer.token, session.peer.output_token, peers[1 - session.rank].output_token
        };
        uint32_t task_count = 0;
        Check(HcommCcuGetTaskArgsNum(session.kernel, &task_count), "resident args count");
        if (task_count != task.size()) throw std::runtime_error("Unexpected resident argument count");
        session.launched = true;
        Check(HcommCcuKernelLaunch(session.thread, session.kernel, task.data(), task.size()), "resident launch");
    });
}

extern "C" int ccu_demo_verify_resident(void *handle, bool deferred) {
    return Protect([&] {
        auto &session = Get(handle);
        if (!session.launched || session.service || session.mailboxes == nullptr)
            throw std::invalid_argument("No resident launch");
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(resident verify)");
        Check(aclrtSynchronizeStreamWithTimeout(session.stream, 30000), "resident stream completion");
        session.drained = true;
        std::array<CcuContractMailbox, kCcuContractRounds> status{};
        Check(
            aclrtMemcpy(status.data(), sizeof(status), session.mailboxes, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST),
            "resident mailbox readback"
        );
        for (uint32_t round = 0; round < status.size(); ++round) {
            const auto &m = status[round];
            const float expected_sample =
                deferred ? InputValue(session.root, session.iteration, round * 32) : kSentinel;
            uint32_t expected_bits = 0;
            std::memcpy(&expected_bits, &expected_sample, sizeof(expected_bits));
            if (m.ready != 1 || m.release != 1 || m.done != 1 || m.observed != 1 || m.observed_value != expected_bits ||
                m.gate_saw_consumer != (deferred ? 0U : 1U)) {
                throw std::runtime_error(
                    "Resident completion mismatch: rank " + std::to_string(session.rank) + " round " +
                    std::to_string(round) + " ready " + std::to_string(m.ready) + " release " +
                    std::to_string(m.release) + " done " + std::to_string(m.done) + " observed " +
                    std::to_string(m.observed) + " early " + std::to_string(m.gate_saw_consumer)
                );
            }
        }
        std::vector<float> output(kWindowFloats), input(kWindowFloats);
        const auto bytes = output.size() * sizeof(float);
        Check(
            aclrtMemcpy(input.data(), bytes, session.window, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "resident producer readback"
        );
        Check(
            aclrtMemcpy(output.data(), bytes, session.separate_output, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "resident output readback"
        );
        for (uint32_t i = 0; i < kWindowFloats; ++i) {
            const bool written =
                (i >= kOutputOffset && i < kOutputOffset + 16) || (i >= kOutputOffset + 32 && i < kOutputOffset + 64);
            const float expected = written ? InputValue(session.root, session.iteration, i - kOutputOffset) : kSentinel;
            if (output[i] != expected)
                throw std::runtime_error("Resident result/guard mismatch at " + std::to_string(i));
            const bool produced =
                (i >= kGuardCount && i < kGuardCount + 16) || (i >= kGuardCount + 32 && i < kGuardCount + 64);
            const float expected_input =
                produced ? InputValue(session.rank, session.iteration, i - kGuardCount) : kSentinel;
            if (input[i] != expected_input)
                throw std::runtime_error("Resident producer/guard mismatch at " + std::to_string(i));
        }
        std::fprintf(
            stderr, "CCU_RESIDENT_VERIFIED rank=%u rounds=%u deferred=%d\n", session.rank, kCcuContractRounds, deferred
        );
    });
}

extern "C" int ccu_demo_verify_service(void *handle, uint32_t rounds) {
    return Protect([&] {
        auto &session = Get(handle);
        if (!session.launched || !session.service || session.mailboxes == nullptr || rounds > kCcuServiceMaxRounds)
            throw std::invalid_argument("No compatible service launch");
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(service verify)");
        CcuServiceProbe status{};
        Check(
            aclrtMemcpy(&status, sizeof(status), session.mailboxes, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST),
            "service status readback"
        );
        const bool stop_acknowledged =
            status.slot.completed == rounds + 1 && status.slot.command == simpler::ccu_service::kStop;
        if (stop_acknowledged) {
            Check(aclrtSynchronizeStreamWithTimeout(session.stream, 30000), "service exit");
            session.drained = true;
        }
        if (!stop_acknowledged || status.failure != 0 || status.verified != rounds) {
            throw std::runtime_error(
                "CCU service rank " + std::to_string(session.rank) + " verified " + std::to_string(status.verified) +
                " completed " + std::to_string(status.slot.completed) + " command " +
                std::to_string(status.slot.command) + " failure " + std::to_string(status.failure)
            );
        }
        std::vector<float> input(kWindowFloats), output(kWindowFloats);
        const size_t bytes = output.size() * sizeof(float);
        Check(aclrtMemcpy(input.data(), bytes, session.window, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "service input");
        Check(
            aclrtMemcpy(output.data(), bytes, session.separate_output, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
            "service output"
        );
        const uint32_t last = rounds == 0 ? 0 : rounds - 1;
        const uint32_t offset = last % 2 == 0 ? 0 : kCcuServiceOffset;
        const uint32_t length = last % 2 == 0 ? kCcuServiceShortCount : kCcuServiceLongCount;
        const uint32_t count = rounds == 0 ? 0 : length;
        const uint32_t seed = last * kCcuServiceFloats;
        for (uint32_t i = 0; i < kWindowFloats; ++i) {
            const bool produced = i >= kGuardCount + offset && i < kGuardCount + offset + count;
            const bool received = i >= kOutputOffset + offset && i < kOutputOffset + offset + count;
            const float expected_input =
                produced ? InputValue(session.rank, session.iteration, seed + i - kGuardCount) : kSentinel;
            const float expected_output =
                received ? InputValue(session.root, session.iteration, seed + i - kOutputOffset) : kSentinel;
            if (input[i] != expected_input || output[i] != expected_output) {
                throw std::runtime_error("Service result/guard mismatch at float " + std::to_string(i));
            }
        }
        std::fprintf(
            stderr, "CCU_SERVICE_VERIFIED rank=%u rounds=%u stop_generation=%llu\n", session.rank, rounds,
            static_cast<unsigned long long>(status.slot.completed)
        );
    });
}

extern "C" int ccu_demo_verify_program_retired(int device, uint64_t kernel) {
    return Protect([&] {
        Check(aclrtSetDevice(device), "aclrtSetDevice(retired handle probe)");
        uint32_t count = 0;
        const auto ret = HcommCcuGetTaskArgsNum(kernel, &count);
        if (ret != CCU_E_NOT_FOUND) {
            throw std::runtime_error(
                "Expected retired CCU handle lookup to return CCU_E_NOT_FOUND, got " + std::to_string(ret)
            );
        }
        std::fprintf(stderr, "CCU_POST_DESTROY_QUERY device=%d rc=%d status=CCU_E_NOT_FOUND\n", device, ret);
    });
}

extern "C" uint64_t ccu_demo_program_handle(void *handle) {
    return handle == nullptr ? 0 : static_cast<Session *>(handle)->kernel;
}

extern "C" int ccu_demo_next_run(void *handle, uint32_t count, uint32_t iteration) {
    return Protect([&] {
        auto &session = Get(handle);
        if (!session.launched || !session.drained || count == 0 || count > kMaxCount || iteration > 10000) {
            throw std::invalid_argument("Previous run must be drained before rebinding data");
        }
        Check(aclrtSetDevice(session.device), "aclrtSetDevice(next run)");
        session.count = count;
        session.iteration = iteration;
        InitializeData(session);
        session.launched = false;
        session.drained = false;
    });
}

extern "C" int ccu_demo_close(void *handle) {
    return Protect([&] {
        if (handle == nullptr) return;
        auto *session = static_cast<Session *>(handle);
        session->Close();
        delete session;
    });
}
