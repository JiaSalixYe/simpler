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

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "../platform_comm/ccu/call_protocol.h"
#include "callable.h"

namespace simpler::ccu {

// Optional Host-only trailer immediately after the canonical device image.
// Little-endian u32 fields: magic, Host-builder format version, module count,
// function count. Modules are (size, bytes), functions are (module index,
// entry name size, entry name bytes). Function position is its callable-local ID.
inline constexpr uint32_t kCallableSectionMagic = 0x31554343;  // "CCU1"
inline constexpr uint32_t kCallableSectionVersion = 1;

struct FunctionEntry {
    uint32_t module_index;
    std::string_view name;
};

struct CallableSection {
    std::array<std::string_view, kMaxFunctions> modules{};
    std::array<FunctionEntry, kMaxFunctions> functions{};
    uint32_t module_count = 0;
    uint32_t function_count = 0;
};

inline bool valid_entry_name(std::string_view name) {
    if (name.empty()) return false;
    for (unsigned char c : name) {
        if (c < '!' || c > '~') return false;
    }
    return true;
}

inline void append_section_word(std::vector<uint8_t> &image, uint32_t word) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        image.push_back(static_cast<uint8_t>(word >> shift));
    }
}

inline void append_callable_section(std::vector<uint8_t> &image, const CallableSection &section) {
    if (section.function_count == 0) return;
    append_section_word(image, kCallableSectionMagic);
    append_section_word(image, kCallableSectionVersion);
    append_section_word(image, section.module_count);
    append_section_word(image, section.function_count);
    for (uint32_t i = 0; i < section.module_count; ++i) {
        const std::string_view module = section.modules[i];
        append_section_word(image, static_cast<uint32_t>(module.size()));
        image.insert(image.end(), module.begin(), module.end());
    }
    for (uint32_t i = 0; i < section.function_count; ++i) {
        const auto &function = section.functions[i];
        append_section_word(image, function.module_index);
        append_section_word(image, static_cast<uint32_t>(function.name.size()));
        image.insert(image.end(), function.name.begin(), function.name.end());
    }
}

inline bool parse_callable_section(const char *data, size_t size, CallableSection &section) {
    section = {};
    if (size == 0) return true;

    const auto read_word = [&data, &size](uint32_t &word) {
        if (size < sizeof(word)) return false;
        word = 0;
        for (unsigned i = 0; i < sizeof(word); ++i) {
            word |= static_cast<uint32_t>(static_cast<unsigned char>(data[i])) << (8 * i);
        }
        data += sizeof(word);
        size -= sizeof(word);
        return true;
    };
    const auto read_bytes = [&data, &size, &read_word](std::string_view &bytes) {
        uint32_t length;
        if (!read_word(length) || length == 0 || length > size) return false;
        bytes = std::string_view(data, length);
        data += length;
        size -= length;
        return true;
    };

    uint32_t magic;
    uint32_t version;
    if (!read_word(magic) || magic != kCallableSectionMagic || !read_word(version) ||
        version != kCallableSectionVersion || !read_word(section.module_count) || !read_word(section.function_count) ||
        section.module_count == 0 || section.module_count > section.function_count ||
        section.function_count > kMaxFunctions) {
        return false;
    }
    for (uint32_t i = 0; i < section.module_count; ++i) {
        if (!read_bytes(section.modules[i])) return false;
    }
    for (uint32_t i = 0; i < section.function_count; ++i) {
        auto &function = section.functions[i];
        if (!read_word(function.module_index) || function.module_index >= section.module_count ||
            !read_bytes(function.name) || !valid_entry_name(function.name)) {
            return false;
        }
    }
    return size == 0;
}

}  // namespace simpler::ccu

// Validate the complete Host wire image. Device upload continues to derive
// its span from the fixed header and canonical child layout, excluding the trailer.
inline bool validate_chip_callable_image(
    const void *image, size_t image_size, simpler::ccu::CallableSection *ccu_section = nullptr
) {
    if (image == nullptr || image_size < sizeof(ChipCallable) ||
        reinterpret_cast<uintptr_t>(image) % alignof(ChipCallable) != 0) {
        return false;
    }
    const auto *callable = static_cast<const ChipCallable *>(image);
    if (callable->sig_count_ < 0 || callable->sig_count_ > CHIP_MAX_TENSOR_ARGS || callable->child_count_ < 0 ||
        callable->child_count_ > ChipCallable::MAX_CHILDREN || callable->func_name_len_ >= CALLABLE_FUNC_NAME_MAX ||
        callable->config_name_len_ >= CALLABLE_FUNC_NAME_MAX ||
        callable->func_name_[callable->func_name_len_] != '\0' ||
        callable->config_name_[callable->config_name_len_] != '\0') {
        return false;
    }

    const size_t storage_size = image_size - offsetof(ChipCallable, storage_);
    size_t used = callable->binary_size_;
    if (used > storage_size) return false;

    for (int32_t i = 0; i < callable->child_count_; ++i) {
        if (used > SIZE_MAX - (CALLABLE_ALIGN - 1)) return false;
        const size_t aligned = (used + CALLABLE_ALIGN - 1) & ~(static_cast<size_t>(CALLABLE_ALIGN) - 1);
        const size_t offset = callable->child_offsets_[i];
        const size_t child_header = CoreCallable::binary_data_offset();
        if (offset != aligned || offset > storage_size || child_header > storage_size - offset) return false;

        const auto *child = reinterpret_cast<const CoreCallable *>(callable->storage_ + offset);
        if (child->sig_count_ < 0 || child->sig_count_ > CORE_MAX_TENSOR_ARGS ||
            child->binary_size_ > storage_size - offset - child_header) {
            return false;
        }
        used = offset + child_header + child->binary_size_;
    }

    if (used == storage_size && ccu_section == nullptr) return true;
    simpler::ccu::CallableSection unused_section;
    return simpler::ccu::parse_callable_section(
        callable->storage_ + used, storage_size - used, ccu_section == nullptr ? unused_section : *ccu_section
    );
}
