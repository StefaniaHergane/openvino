// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "plugin_property_manager.hpp"

// Matches the [ INFO ] convention used elsewhere in these tests (see core_integration.cpp).
#define NPU_TEST_LOG std::cout << "[          ] [ LOG ] "

namespace {

std::string toString(const ov::CompilationTarget& target) {
    std::ostringstream oss;
    oss << target;
    return oss.str();
}

// No backend and no device involved: offline_compilation_targets must be answerable from the
// plugin's own platform table alone.
class OfflineCompilationTargetsTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;

    void SetUp() override {
        using namespace ::intel_npu;

        options->add<PLATFORM>();
        options->add<DEVICE_ID>();
        options->add<COMPILER_TYPE>();

        const ov::SoPtr<IEngineBackend> backend;
        propertiesManager = std::make_unique<PluginPropertyManager>(
            options,
            backend,
            std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory()),
            Logger::global());
    }
};

// The property must be advertised as supported even with no backend/device present.
TEST_F(OfflineCompilationTargetsTests, IsSupportedWithNoBackend) {
    bool isSupported = false;
    OV_ASSERT_NO_THROW(isSupported = propertiesManager->isPropertySupported(ov::offline_compilation_targets.name()));
    NPU_TEST_LOG << "isPropertySupported(" << ov::offline_compilation_targets.name() << ") = " << std::boolalpha
                 << isSupported << std::endl;
    ASSERT_TRUE(isSupported);
}

// Every known platform must appear, each carrying at least one PCI device ID.
TEST_F(OfflineCompilationTargetsTests, EnumeratesEveryExportablePlatform) {
    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(ov::offline_compilation_targets.name()));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();

    NPU_TEST_LOG << "offline_compilation_targets returned " << targets.size() << " target(s):" << std::endl;
    for (const auto& target : targets) {
        NPU_TEST_LOG << "  " << toString(target) << std::endl;
    }

    std::vector<std::string> platforms;
    for (const auto& target : targets) {
        ASSERT_FALSE(target.device_ids.empty()) << "Target for platform '" << target.platform << "' has no device IDs";
        platforms.push_back(target.platform);
    }
    for (const auto& expectedPlatform : {ov::intel_npu::Platform::NPU3720,
                                         ov::intel_npu::Platform::NPU4000,
                                         ov::intel_npu::Platform::NPU5010,
                                         ov::intel_npu::Platform::NPU5020,
                                         ov::intel_npu::Platform::NPU6010}) {
        ASSERT_TRUE(std::find(platforms.begin(), platforms.end(), std::string(expectedPlatform)) != platforms.end())
            << "Platform '" << expectedPlatform << "' is missing from the offline compilation targets";
    }
}

TEST_F(OfflineCompilationTargetsTests, IndependentOfConfiguredPlatform) {
    // The query must return the full platform list regardless of the platform currently configured -
    // it enumerates what the compiler can build for, not what the current config happens to select.
    propertiesManager->setProperty({{ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU6010))}});

    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(ov::offline_compilation_targets.name()));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();
    NPU_TEST_LOG << "With NPU_PLATFORM=6010 configured, query still returned " << targets.size() << " target(s)"
                 << std::endl;
    ASSERT_GT(targets.size(), 1u);
}

// Unlike the persisted config above, an NPU_PLATFORM passed as a query argument does narrow the result -
// still answered from the plugin's own table alone, no backend or device involved.
TEST_F(OfflineCompilationTargetsTests, FiltersByPlatformArgument) {
    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(
                            ov::offline_compilation_targets.name(),
                            {{ov::intel_npu::platform.name(), std::string(ov::intel_npu::Platform::NPU6010)}}));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();

    NPU_TEST_LOG << "offline_compilation_targets filtered by NPU_PLATFORM=6010 returned " << targets.size()
                 << " target(s)" << std::endl;
    ASSERT_EQ(targets.size(), 1u);
    ASSERT_EQ(targets.front().platform, std::string(ov::intel_npu::Platform::NPU6010));
}

}  // namespace
