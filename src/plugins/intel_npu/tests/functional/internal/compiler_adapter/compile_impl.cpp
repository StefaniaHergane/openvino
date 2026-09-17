// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

// ov::CompilationTarget's operator<< is a compact, whitespace-free wire format (required so
// std::vector<CompilationTarget> round-trips through ov::Any); this mirrors the proposal's
// "{platform, [device_ids], [configs]}" notation and is for test logs only.
std::string toDebugString(const ov::CompilationTarget& target) {
    std::ostringstream oss;
    oss << "{" << target.platform << ", [";
    for (size_t i = 0; i < target.device_ids.size(); ++i) {
        oss << (i ? ", " : "") << "0x" << std::hex << std::uppercase << target.device_ids[i] << std::dec
            << std::nouppercase;
    }
    oss << "], [";
    for (size_t i = 0; i < target.configs.size(); ++i) {
        oss << (i ? ", " : "") << "\"" << target.configs[i] << "\"";
    }
    oss << "]}";
    return oss.str();
}

using OfflineCompilationTargetsTests = ::testing::Test;

TEST_F(OfflineCompilationTargetsTests, EnumeratesEveryOfflineTargetTheCompilerSupports) {
    ov::Core core;
    auto targets = core.get_property("NPU", ov::offline_compilation_targets);

    std::ostringstream report;
    report << "Enumerating offline compilation targets ({platform, [device_ids], [configs]}): " << targets.size()
          << " found\n";
    for (const auto& target : targets) {
        report << "  " << toDebugString(target) << '\n';
    }
    std::cout << report.str();

    std::vector<std::string> returnedPlatforms;
    for (const auto& target : targets) {
        returnedPlatforms.push_back(target.platform);
    }
    for (const auto& expectedPlatform : {ov::intel_npu::Platform::NPU3720,
                                         ov::intel_npu::Platform::NPU4000,
                                         ov::intel_npu::Platform::NPU5010,
                                         ov::intel_npu::Platform::NPU5020,
                                         ov::intel_npu::Platform::NPU6010}) {
        // An unfiltered query must enumerate every platform the compiler exports, none dropped.
        EXPECT_TRUE(std::find(returnedPlatforms.begin(), returnedPlatforms.end(), std::string(expectedPlatform)) !=
                    returnedPlatforms.end());
    }
}

TEST_F(OfflineCompilationTargetsTests, QueryTargetsForOnePlatform) {
    ov::Core core;
    for (const auto& platform : {ov::intel_npu::Platform::NPU3720,
                                 ov::intel_npu::Platform::NPU4000,
                                 ov::intel_npu::Platform::NPU5010,
                                 ov::intel_npu::Platform::NPU5020,
                                 ov::intel_npu::Platform::NPU6010}) {
        auto targets = core.get_property(
            "NPU",
            ov::offline_compilation_targets,
            ov::AnyMap{{ov::intel_npu::platform.name(), std::string(platform)}});

        std::ostringstream report;
        report << "Enumerating offline compilation targets for platform " << platform
              << " ({platform, [device_ids], [configs]}): " << targets.size() << " found\n";
        for (const auto& target : targets) {
            report << "  " << toDebugString(target) << '\n';
        }
        std::cout << report.str();

        // Filtering by a single platform must return exactly one target, matching that platform.
        ASSERT_EQ(targets.size(), 1u);
        EXPECT_EQ(targets.front().platform, platform);
    }
}

TEST_F(OfflineCompilationTargetsTests, QueryTargetsForPlatform6010WithTiles1) {
    ov::Core core;
    auto targets = core.get_property(
        "NPU",
        ov::offline_compilation_targets,
        ov::AnyMap{{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)},
                   {ov::intel_npu::tiles.name(), 1}});

    std::ostringstream report;
    report << "Enumerating offline compilation targets for platform " << ov::intel_npu::Platform::NPU6010
          << " ({platform, [device_ids], [configs]}): " << targets.size() << " found\n";
    for (const auto& target : targets) {
        report << "  " << toDebugString(target) << '\n';
    }
    std::cout << report.str();

    ASSERT_EQ(targets.size(), 1u);
    // A platform pinned together with tiles still resolves to that same platform.
    EXPECT_EQ(targets.front().platform, ov::intel_npu::Platform::NPU6010);
    // Pinning tiles collapses 6010's multi-SKU expansion to a single config bundle.
    ASSERT_EQ(targets.front().configs.size(), 1u);
}

TEST_F(OfflineCompilationTargetsTests, QueryTargetsForPlatform6010WithThroughputHint) {
    ov::Core core;
    auto targets = core.get_property(
        "NPU",
        ov::offline_compilation_targets,
        ov::AnyMap{{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)},
                   {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::THROUGHPUT}});

    std::ostringstream report;
    report << "Enumerating offline compilation targets for platform " << ov::intel_npu::Platform::NPU6010
          << " ({platform, [device_ids], [configs]}): " << targets.size() << " found\n";
    for (const auto& target : targets) {
        report << "  " << toDebugString(target) << '\n';
    }
    std::cout << report.str();

    ASSERT_EQ(targets.size(), 1u);
    // A platform pinned together with a performance hint still resolves to that same platform.
    EXPECT_EQ(targets.front().platform, ov::intel_npu::Platform::NPU6010);
    // Pinning the performance hint collapses 6010's multi-SKU expansion to a single config bundle.
    ASSERT_EQ(targets.front().configs.size(), 1u);
}

TEST_F(OfflineCompilationTargetsTests, ThrowsWhenExplicitlyPinnedToDriverCompiler) {
    ov::Core core;
    // Explicitly pinning DRIVER must fail: this property only ever answers via the plugin compiler,
    // which is the only one that can work with no device present.
    OV_EXPECT_THROW_HAS_SUBSTRING(
        core.get_property(
            "NPU",
            ov::offline_compilation_targets,
            ov::AnyMap{{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::DRIVER}}),
        ov::Exception,
        "Unsupported configuration key");
}

TEST_F(OfflineCompilationTargetsTests, SucceedsWhenExplicitlyPinnedToPluginCompiler) {
    ov::Core core;
    OV_ASSERT_NO_THROW(
        core.get_property(
            "NPU",
            ov::offline_compilation_targets,
            ov::AnyMap{{ov::intel_npu::compiler_type.name(), ov::intel_npu::CompilerType::PLUGIN}}));
}

}  // namespace
