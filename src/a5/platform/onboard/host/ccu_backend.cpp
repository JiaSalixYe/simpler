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
#include "ccu_backend.h"

#if defined(SIMPLER_A5_CCU_CANDIDATE) && \
    __has_include("hccl/hccl_ccu_res.h") && __has_include("hccl/hccl_rank_graph.h") && __has_include("hcomm/ccu/ccu_primitives.hpp") && __has_include("hcomm/ccu/ccu_control_flow_macro.h") && __has_include("hcomm/ccu/ccu_launch.h") && __has_include("hcomm/ccu/ccu_res.h") && __has_include("ccu_primitives_impl.h")
#define SIMPLER_A5_HAS_CCU
#endif

#ifdef SIMPLER_A5_HAS_CCU
#include "ccu_builder.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_ccu_res.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hcomm/ccu/ccu_launch.h"
#include "hcomm/ccu/ccu_res.h"

namespace simpler::ccu {
namespace {

constexpr uint32_t kRanks = 2;
constexpr uint32_t kLaunchArgs = 5;
constexpr int kControlTimeoutMs = 5000;
constexpr int kStopTimeoutMs = 30000;
constexpr auto kRendezvousTimeout = std::chrono::seconds(30);
constexpr uint64_t kInvalidFunction = 1;
thread_local char last_error[512];

void SetError(const char *message) noexcept { std::snprintf(last_error, sizeof(last_error), "%s", message); }

template <typename Fn>
bool HasSymbol(Fn *symbol) noexcept {
    return symbol != nullptr;
}

void Check(int result, const char *operation) {
    if (result != 0) {
        throw std::runtime_error(std::string(operation) + " failed (" + std::to_string(result) + ")");
    }
}

template <typename Fn>
int Protect(Fn fn) noexcept {
    last_error[0] = '\0';
    try {
        fn();
        return 0;
    } catch (const std::exception &error) {
        SetError(error.what());
    } catch (...) {
        SetError("Unknown CCU backend failure");
    }
    return -1;
}

void CheckIo(const char *operation) { throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno)); }

struct File {
    int fd;
    ~File() {
        if (fd >= 0) (void)::close(fd);
    }

    void Close() {
        int result = ::close(fd);
        fd = -1;
        if (result != 0) CheckIo("Close CCU file");
    }
};

struct TempFile {
    std::string path;
    File file;

    explicit TempFile(const std::string &prefix) :
        path(prefix + ".XXXXXX"),
        file{::mkstemp(path.data())} {
        if (file.fd < 0) CheckIo("Create CCU file");
    }

    ~TempFile() { (void)::unlink(path.c_str()); }

    void Write(const void *data, size_t size) {
        auto *source = static_cast<const uint8_t *>(data);
        while (size != 0) {
            const auto count = ::write(file.fd, source, size);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) CheckIo("Write CCU file");
            if (count == 0) throw std::runtime_error("Incomplete CCU file write");
            source += count;
            size -= count;
        }
        file.Close();
    }
};

template <typename T>
void Publish(const std::string &path, const T &value) {
    TempFile temporary(path);
    temporary.Write(&value, sizeof(value));
    if (::rename(temporary.path.c_str(), path.c_str()) != 0) CheckIo("Publish CCU rendezvous file");
}

template <typename T>
T Await(const std::string &path) {
    const auto deadline = std::chrono::steady_clock::now() + kRendezvousTimeout;
    for (;;) {
        File file{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
        if (file.fd >= 0) {
            T value{};
            auto *target = reinterpret_cast<uint8_t *>(&value);
            size_t remaining = sizeof(value);
            while (remaining != 0) {
                const auto count = ::read(file.fd, target, remaining);
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) CheckIo("Read CCU rendezvous file");
                if (count == 0) throw std::runtime_error("Truncated CCU rendezvous file");
                target += count;
                remaining -= count;
            }
            return value;
        }
        if (errno != ENOENT && errno != EINTR) CheckIo("Open CCU rendezvous file");
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("Timed out waiting for CCU peer: " + path);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

struct Window {
    uint64_t base;
    uint64_t token;
};

enum class ServiceState { IDLE, RUNNING, UNCERTAIN };

struct Session {
    int32_t device = -1;
    uint32_t rank = 0;
    std::string prefix;
    std::vector<void *> modules;
    std::vector<BuilderV1> functions;
    HcclComm comm = nullptr;
    aclrtStream service = nullptr;
    aclrtStream control = nullptr;
    ChannelHandle channel = 0;
    ThreadHandle thread = 0;
    CcuKernelHandle kernel = 0;
    std::array<uint64_t, kLaunchArgs> launch_args{};
    void *mailbox = nullptr;
    void *binding = nullptr;
    void *host_mailbox = nullptr;
    uint64_t binding_key = 0;
    size_t users = 0;
    uint64_t stop_generation = 0;
    ServiceState state = ServiceState::IDLE;

    std::string RankFile(const char *kind, uint32_t file_rank) const {
        return prefix + "." + kind + "." + std::to_string(file_rank);
    }

    void LoadBuilders(const ModuleEntry *entries, size_t count);
    void AcquireChannel();
    void Open(const DomainConfig &config);
    Mailbox ReadMailbox();
    void PublishMailbox(size_t offset, size_t bytes);
    void Begin();
    void End();
    void CheckDrained() const;
    void Stop();
    void Close();
};

// Native runs borrow a binding by its device address. Serializing lookup and
// lease transitions also prevents the domain owner from closing an active binding.
std::mutex sessions_mutex;
std::unordered_map<uint64_t, Session *> sessions;

Session &FindSession(uint64_t binding) {
    const auto found = sessions.find(binding);
    if (found == sessions.end()) throw std::invalid_argument("Unknown CCU binding");
    return *found->second;
}

void Session::LoadBuilders(const ModuleEntry *entries, size_t count) {
    modules.reserve(count);
    functions.reserve(count);
    std::array<void *, kMaxFunctions> entry_modules{};
    for (size_t i = 0; i < count; ++i) {
        const auto &entry = entries[i];
        if (entry.binary == nullptr || entry.binary_size == 0 || entry.symbol == nullptr || entry.symbol[0] == '\0') {
            throw std::invalid_argument("Invalid CCU module entry");
        }
        void *module = nullptr;
        for (size_t j = 0; j < i; ++j) {
            if (entry.binary_size == entries[j].binary_size &&
                std::memcmp(entry.binary, entries[j].binary, entry.binary_size) == 0) {
                module = entry_modules[j];
                break;
            }
        }
        if (module == nullptr) {
            TempFile image(RankFile("module", rank));
            image.Write(entry.binary, entry.binary_size);
            module = ::dlopen(image.path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (module == nullptr) throw std::runtime_error(std::string("Load CCU module: ") + ::dlerror());
            modules.push_back(module);
        }
        entry_modules[i] = module;
        (void)::dlerror();
        const auto builder = ::dlsym(module, entry.symbol);
        const auto error = ::dlerror();
        if (error != nullptr || builder == nullptr) {
            throw std::runtime_error(
                std::string("Resolve CCU entry ") + entry.symbol + ": " + (error == nullptr ? "empty symbol" : error)
            );
        }
        functions.push_back(reinterpret_cast<BuilderV1>(builder));
    }
}

void Session::AcquireChannel() {
    const uint32_t peer = 1 - rank;
    CommLink *links = nullptr;
    uint32_t link_count = 0;
    Check(HcclRankGraphGetLinks(comm, 0, rank, peer, &links, &link_count), "HcclRankGraphGetLinks");
    for (uint32_t i = 0; i < link_count; ++i) {
        const auto &link = links[i];
        if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) continue;
        HcclChannelDesc desc;
        Check(HcclChannelDescInit(&desc, 1), "HcclChannelDescInit");
        desc.remoteRank = peer;
        desc.notifyNum = 4;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        Check(HcclChannelAcquire(comm, COMM_ENGINE_CCU, &desc, 1, &channel), "HcclChannelAcquire(CCU)");
        return;
    }
    throw std::runtime_error("CCU requires a UBC_CTP link to the other rank");
}

CcuResult ProbeDie(const ChannelHandle *channel) { return AscendC::ccu::NotifyWait(*channel, 0); }

uint8_t ChannelDie(ChannelHandle channel) {
    const void *arguments[] = {&channel};
    for (uint8_t die = 0; die < 2; ++die) {
        HcommCcuResDescHandle descriptor = 0;
        Check(HcommCcuInsResDescCreate(die, &descriptor), "HcommCcuInsResDescCreate");
        const auto result =
            HcommCcuKernelQueryResReq(reinterpret_cast<const void *>(ProbeDie), arguments, 1, descriptor);
        Check(HcommCcuInsResDescDestroy(descriptor), "HcommCcuInsResDescDestroy");
        if (result == CCU_SUCCESS) return die;
        if (result != CCU_E_PARA && result != CCU_E_UNAVAIL) Check(result, "HcommCcuKernelQueryResReq");
    }
    throw std::runtime_error("CCU channel has no enabled matching IO die");
}

CcuResult RecordDispatcher(const Session *session) {
    namespace isa = AscendC::ccu;
    try {
        isa::Variable base, local_window, local_token, peer_window, peer_token;
        const std::array<isa::Variable, kLaunchArgs> launch_args{
            base, local_window, local_token, peer_window, peer_token
        };
        for (uint32_t i = 0; i < launch_args.size(); ++i)
            Check(isa::LoadArg(launch_args[i], i), "Load CCU launch arg");

        isa::Variable command, idle, status, command_addr, completed_addr, status_addr, tensors_addr, scalars_addr;
        idle = kIdle;
        command = kIdle;
        command_addr = base + static_cast<uint16_t>(offsetof(Mailbox, command));
        completed_addr = base + static_cast<uint16_t>(offsetof(Mailbox, completed));
        status_addr = base + static_cast<uint16_t>(offsetof(Mailbox, status));
        tensors_addr = base + static_cast<uint16_t>(offsetof(Mailbox, tensors));
        scalars_addr = base + static_cast<uint16_t>(offsetof(Mailbox, scalars));
        isa::Array<isa::Variable> request(4), tensors(kMaxTensors * 2), scalars(kMaxScalars);
        const BuilderContextV1 context{kBuilderAbiVersion, session->rank, session->channel, local_window,
                                       local_token,        peer_window,   peer_token,       request[2],
                                       request[3],         tensors,       scalars,          status};

        constexpr auto service_loop = "simpler_ccu_service";
        constexpr auto poll_loop = "simpler_ccu_poll";
        constexpr auto execute_if = "simpler_ccu_execute";
        Check(CcuWhileBegin(command.handle, kStop, CCU_CONDITION_NE, service_loop), "Begin CCU service");
        command = kIdle;
        Check(CcuWhileBegin(command.handle, kIdle, CCU_CONDITION_EQ, poll_loop), "Begin CCU request poll");
        Check(isa::Load(command_addr, command), "Read CCU command");
        Check(CcuWhileEnd(poll_loop), "End CCU request poll");

        Check(isa::Load(base, request, 4), "Read CCU request header");
        Check(CcuIfBegin(command.handle, kExecute, CCU_CONDITION_EQ, execute_if), "Begin CCU request");
        Check(isa::Load(tensors_addr, tensors, kMaxTensors * 2), "Read CCU tensor arguments");
        Check(isa::Load(scalars_addr, scalars, kMaxScalars), "Read CCU scalar arguments");
        status = kInvalidFunction;
        for (size_t id = 0; id < session->functions.size(); ++id) {
            // Distinct labels are required for entries recorded at the same host source location.
            const std::string label = "simpler_ccu_entry_" + std::to_string(id);
            Check(CcuIfBegin(request[1].handle, id, CCU_CONDITION_EQ, label.c_str()), "Begin CCU entry");
            status = kSuccess;
            Check(session->functions[id](&context), "Record CCU entry");
            Check(CcuFlushPendingIfs(), "Flush CCU entry conditions");
            Check(CcuIfEnd(label.c_str()), "End CCU entry");
        }
        Check(isa::Store(status_addr, status), "Publish CCU status");
        Check(isa::Store(command_addr, idle), "Clear CCU command");
        Check(CcuIfEnd(execute_if), "End CCU request");
        Check(isa::Store(completed_addr, request[0]), "Publish CCU generation");
        Check(CcuWhileEnd(service_loop), "End CCU service");
        return CCU_SUCCESS;
    } catch (const std::exception &error) {
        SetError(error.what());
    } catch (...) {
        SetError("Unknown CCU builder failure");
    }
    return CCU_E_INTERNAL;
}

void Session::Open(const DomainConfig &config) {
    const uint32_t peer_rank = 1 - rank;
    Check(aclrtSetDevice(device), "aclrtSetDevice(CCU prepare)");
    Check(aclrtCreateStream(&service), "aclrtCreateStream(CCU)");
    Check(aclrtCreateStream(&control), "aclrtCreateStream(CCU control)");
    Check(aclrtMalloc(&mailbox, sizeof(Mailbox), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(CCU mailbox)");
    Check(aclrtMemset(mailbox, sizeof(Mailbox), 0, sizeof(Mailbox)), "aclrtMemset(CCU mailbox)");
    Check(aclrtMalloc(&binding, sizeof(Binding), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(CCU binding)");
    Check(aclrtMallocHost(&host_mailbox, sizeof(Mailbox)), "aclrtMallocHost(CCU mailbox)");
    const Binding view{
        config.domain_ctx,
        config.window_base,
        config.window_size,
        reinterpret_cast<uint64_t>(mailbox),
        static_cast<uint32_t>(functions.size()),
        0
    };
    Check(aclrtMemcpy(binding, sizeof(view), &view, sizeof(view), ACL_MEMCPY_HOST_TO_DEVICE), "Upload CCU binding");

    Window local{config.window_base, 0};
    Check(HcommCcuGetMemToken(local.base, config.window_size, &local.token), "HcommCcuGetMemToken(domain)");
    Publish(RankFile("window", rank), local);
    const auto peer = Await<Window>(RankFile("window", peer_rank));

    HcclRootInfo root{};
    const auto root_path = prefix + ".root";
    if (rank == 0) {
        Check(HcclGetRootInfo(&root), "HcclGetRootInfo(CCU)");
        Publish(root_path, root);
    } else {
        root = Await<HcclRootInfo>(root_path);
    }
    // The Simpler domain uses the ordinary HCCL communicator. This private
    // communicator owns only this callable's CCU channel, kernel and thread.
    HcclCommConfig comm_config{};
    HcclCommConfigInit(&comm_config);
    comm_config.hcclOpExpansionMode = HCCL_OP_EXPANSION_MODE_CCU_SCHED;
    Check(HcclCommInitRootInfoConfig(kRanks, &root, rank, &comm_config, &comm), "HcclCommInitRootInfoConfig(CCU)");

    Check(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, service, 1, &thread), "HcclThreadAcquireWithStream(CCU)");
    AcquireChannel();
    std::array<CcuInsHandle, 2> instances{};
    uint32_t instance_count = 0;
    Check(HcclCommQueryCcuIns(comm, instances.data(), &instance_count), "HcclCommQueryCcuIns");
    if (instance_count != 1) throw std::runtime_error("Expected one communicator-owned CCU instance");
    const auto instance = instances[0];
    const auto die = ChannelDie(channel);
    const void *builder_args[] = {this};
    Check(HcommCcuKernelRegisterStart(instance), "HcommCcuKernelRegisterStart");
    const auto registered = HcommCcuKernelRegister(
        instance, die, "SimplerDomainDispatcher", reinterpret_cast<const void *>(RecordDispatcher), builder_args, 1,
        &kernel
    );
    const auto ended = HcommCcuKernelRegisterEnd(instance);
    if (registered != CCU_SUCCESS && last_error[0] != '\0') throw std::runtime_error(last_error);
    Check(registered, "HcommCcuKernelRegister");
    Check(ended, "HcommCcuKernelRegisterEnd");

    launch_args = {view.mailbox, local.base, local.token, peer.base, peer.token};
    uint32_t argument_count = 0;
    Check(HcommCcuGetTaskArgsNum(kernel, &argument_count), "HcommCcuGetTaskArgsNum");
    if (argument_count != launch_args.size()) throw std::runtime_error("Unexpected CCU launch argument count");
    Publish(RankFile("registered", rank), true);
    (void)Await<bool>(RankFile("registered", peer_rank));
}

Mailbox Session::ReadMailbox() {
    Check(
        aclrtMemcpyAsync(host_mailbox, sizeof(Mailbox), mailbox, sizeof(Mailbox), ACL_MEMCPY_DEVICE_TO_HOST, control),
        "Read CCU mailbox"
    );
    Check(aclrtSynchronizeStreamWithTimeout(control, kControlTimeoutMs), "Fence CCU control stream");
    Mailbox snapshot;
    std::memcpy(&snapshot, host_mailbox, sizeof(snapshot));
    return snapshot;
}

void Session::PublishMailbox(size_t offset, size_t bytes) {
    auto *device_base = static_cast<uint8_t *>(mailbox);
    auto *host_base = static_cast<const uint8_t *>(host_mailbox);
    Check(
        aclrtMemcpyAsync(device_base + offset, bytes, host_base + offset, bytes, ACL_MEMCPY_HOST_TO_DEVICE, control),
        "Write CCU mailbox"
    );
}

void Session::Begin() {
    if (state == ServiceState::UNCERTAIN || stop_generation == UINT64_MAX || users == SIZE_MAX) {
        throw std::runtime_error("CCU binding is not reusable");
    }
    if (users == 0) {
        Check(aclrtSetDevice(device), "aclrtSetDevice(CCU start)");
        // A failed launch may have enqueued a task. Retain the binding and
        // its domain if that outcome cannot be determined.
        state = ServiceState::UNCERTAIN;
        Check(HcommCcuKernelLaunch(thread, kernel, launch_args.data(), launch_args.size()), "HcommCcuKernelLaunch");
        state = ServiceState::RUNNING;
    }
    ++users;
}

void Session::End() {
    if (state != ServiceState::RUNNING || users == 0) throw std::runtime_error("CCU run has no active binding lease");
    if (--users != 0) return;
    state = ServiceState::UNCERTAIN;
    Check(aclrtSetDevice(device), "aclrtSetDevice(CCU stop)");
    Stop();
    state = ServiceState::IDLE;
}

void Session::CheckDrained() const {
    if (users != 0 || state != ServiceState::IDLE) {
        throw std::runtime_error("CCU service has an active or unconfirmed run; keeping domain resources");
    }
}

void Session::Stop() {
    auto snapshot = ReadMailbox();
    if (snapshot.command != kIdle || snapshot.generation != snapshot.completed ||
        snapshot.completed != snapshot.recycled || snapshot.status != kSuccess || snapshot.completed == UINT64_MAX) {
        throw std::runtime_error("CCU mailbox has unretired work; keeping domain resources");
    }
    stop_generation = snapshot.completed + 1;
    snapshot.generation = stop_generation;
    snapshot.command = kStop;
    std::memcpy(host_mailbox, &snapshot, sizeof(snapshot));
    PublishMailbox(offsetof(Mailbox, generation), sizeof(snapshot.generation));
    PublishMailbox(offsetof(Mailbox, command), sizeof(snapshot.command));
    Check(aclrtSynchronizeStreamWithTimeout(control, kControlTimeoutMs), "Fence CCU stop publication");

    Check(aclrtSynchronizeStreamWithTimeout(service, kStopTimeoutMs), "Drain CCU service");
    snapshot = ReadMailbox();
    if (snapshot.command != kStop || snapshot.generation != stop_generation || snapshot.completed != stop_generation) {
        throw std::runtime_error("CCU service did not acknowledge its stop generation; keeping domain resources");
    }
    // No AIV or CCU can still access the slot. Recycle the exact terminal
    // generation before the next run so no request identity is reused.
    snapshot.command = kIdle;
    snapshot.recycled = stop_generation;
    std::memcpy(host_mailbox, &snapshot, sizeof(snapshot));
    PublishMailbox(0, sizeof(snapshot));
    Check(aclrtSynchronizeStreamWithTimeout(control, kControlTimeoutMs), "Fence CCU mailbox recycle");
}

void Session::Close() {
    Check(aclrtSetDevice(device), "aclrtSetDevice(CCU close)");
    if (comm != nullptr) {
        Check(HcclCommDestroy(comm), "HcclCommDestroy(CCU)");
        comm = nullptr;
    }
    if (service != nullptr) {
        Check(aclrtDestroyStream(service), "aclrtDestroyStream(CCU)");
        service = nullptr;
    }
    if (control != nullptr) {
        Check(aclrtDestroyStream(control), "aclrtDestroyStream(CCU control)");
        control = nullptr;
    }
    if (binding != nullptr) {
        Check(aclrtFree(binding), "aclrtFree(CCU binding)");
        binding = nullptr;
    }
    if (mailbox != nullptr) {
        Check(aclrtFree(mailbox), "aclrtFree(CCU mailbox)");
        mailbox = nullptr;
    }
    if (host_mailbox != nullptr) {
        Check(aclrtFreeHost(host_mailbox), "aclrtFreeHost(CCU mailbox)");
        host_mailbox = nullptr;
    }
    while (!modules.empty()) {
        if (::dlclose(modules.back()) != 0) throw std::runtime_error(std::string("Unload CCU module: ") + ::dlerror());
        modules.pop_back();
    }
    if (rank == 0) (void)::unlink((prefix + ".root").c_str());
    (void)::unlink(RankFile("window", rank).c_str());
    (void)::unlink(RankFile("registered", rank).c_str());
}

}  // namespace

int Prepare(
    const DomainConfig &config, const ModuleEntry *functions, size_t function_count, void **owner,
    uint64_t *device_binding
) noexcept {
    return Protect([&] {
        if (owner == nullptr || *owner != nullptr || device_binding == nullptr) {
            throw std::invalid_argument("Invalid CCU binding output arguments");
        }
        *device_binding = 0;
        if (!Available()) throw std::runtime_error("CCU backend libraries are missing required symbols");
        if (config.device < 0 || config.rank >= kRanks || config.domain_ctx == 0 || config.window_base == 0 ||
            config.window_size == 0 || config.window_size > UINT64_MAX - config.window_base ||
            config.rendezvous_prefix == nullptr || config.rendezvous_prefix[0] == '\0' || functions == nullptr ||
            function_count == 0 || function_count > kMaxFunctions) {
            throw std::invalid_argument("Invalid CCU domain or callable arguments");
        }
        auto *session = new Session;
        *owner = session;
        session->device = config.device;
        session->rank = config.rank;
        session->prefix = config.rendezvous_prefix;
        session->LoadBuilders(functions, function_count);
        session->Open(config);
        const auto key = reinterpret_cast<uint64_t>(session->binding);
        std::lock_guard<std::mutex> lock(sessions_mutex);
        if (!sessions.emplace(key, session).second) throw std::runtime_error("Duplicate CCU binding address");
        session->binding_key = key;
        *device_binding = key;
    });
}

int Begin(uint64_t device_binding) noexcept {
    return Protect([&] {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        FindSession(device_binding).Begin();
    });
}

int End(uint64_t device_binding) noexcept {
    return Protect([&] {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        FindSession(device_binding).End();
    });
}

bool Drained(void *owner) noexcept {
    return Protect([&] {
               std::lock_guard<std::mutex> lock(sessions_mutex);
               if (owner != nullptr) static_cast<Session *>(owner)->CheckDrained();
           }) == 0;
}

int Close(void *owner) noexcept {
    return Protect([&] {
        if (owner == nullptr) return;
        std::lock_guard<std::mutex> lock(sessions_mutex);
        auto *session = static_cast<Session *>(owner);
        session->CheckDrained();
        if (session->binding_key != 0) {
            sessions.erase(session->binding_key);
            session->binding_key = 0;
        }
        session->Close();
        delete session;
    });
}

bool Available() noexcept {
    // Older runtime libraries may load successfully while leaving SDK weak
    // references unresolved. Do not start a collective with a missing entry.
    return HasSymbol(HcclGetRootInfo) && HasSymbol(HcclCommInitRootInfoConfig) && HasSymbol(HcclCommDestroy) &&
           HasSymbol(HcommCcuKernelRegisterStart) && HasSymbol(HcommCcuKernelRegister) &&
           HasSymbol(HcommCcuKernelRegisterEnd) && HasSymbol(HcommCcuKernelLaunch);
}

const char *LastError() noexcept { return last_error; }

}  // namespace simpler::ccu

#else

namespace simpler::ccu {

int Prepare(const DomainConfig &, const ModuleEntry *, size_t, void **, uint64_t *device_binding) noexcept {
    if (device_binding != nullptr) *device_binding = 0;
    return -1;
}

int Begin(uint64_t) noexcept { return -1; }

int End(uint64_t) noexcept { return -1; }

bool Drained(void *owner) noexcept { return owner == nullptr; }

int Close(void *owner) noexcept { return owner == nullptr ? 0 : -1; }

bool Available() noexcept { return false; }

const char *LastError() noexcept { return "CCU backend is unavailable in this A5 runtime build"; }

}  // namespace simpler::ccu

#endif
