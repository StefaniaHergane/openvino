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

TEST_F(OfflineCompilationTargetsTests, IsSupportedWithNoBackend) {
    bool isSupported = false;
    OV_ASSERT_NO_THROW(isSupported = propertiesManager->isPropertySupported(ov::offline_compilation_targets.name()));
    NPU_TEST_LOG << "isPropertySupported(" << ov::offline_compilation_targets.name() << ") = " << std::boolalpha
                 << isSupported << std::endl;
    ASSERT_TRUE(isSupported);
}

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

// No backend and no device involved: the selector only has to rewrite the config, which needs
// neither.
class CompilationTargetSelectorTests : public ::testing::Test {
protected:
    std::shared_ptr<::intel_npu::OptionsDesc> options = std::make_shared<::intel_npu::OptionsDesc>();
    std::unique_ptr<::intel_npu::PluginPropertyManager> propertiesManager;

    void SetUp() override {
        using namespace ::intel_npu;

        options->add<PLATFORM>();
        options->add<DEVICE_ID>();
        options->add<COMPILER_TYPE>();
        options->add<COMPILATION_TARGET>();
        options->add<TILES>();

        const ov::SoPtr<IEngineBackend> backend;
        propertiesManager = std::make_unique<PluginPropertyManager>(
            options,
            backend,
            std::make_shared<CompilerOptionSupportHelper>(backend, CompilerAdapterFactory()),
            Logger::global());
    }
};

TEST_F(CompilationTargetSelectorTests, IsSupported) {
    bool isSupported = false;
    OV_ASSERT_NO_THROW(isSupported = propertiesManager->isPropertySupported(ov::compilation_target.name()));
    NPU_TEST_LOG << "isPropertySupported(" << ov::compilation_target.name() << ") = " << std::boolalpha << isSupported
                 << std::endl;
    ASSERT_TRUE(isSupported);
}

TEST_F(CompilationTargetSelectorTests, ThrowsWhenQueriedBeforeBeingSet) {
    try {
        propertiesManager->getProperty(ov::compilation_target.name());
        FAIL() << "Expected getProperty(" << ov::compilation_target.name() << ") to throw";
    } catch (const ov::Exception& e) {
        NPU_TEST_LOG << "getProperty(" << ov::compilation_target.name() << ") threw as expected: \"" << e.what()
                     << "\"" << std::endl;
        ASSERT_NE(std::string(e.what()).find("was not provided"), std::string::npos);
    }
}

TEST_F(CompilationTargetSelectorTests, InjectsItsPlatformIntoTheMergedConfig) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};
    NPU_TEST_LOG << "Merging compilation_target = " << toString(target) << std::endl;

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(merged = propertiesManager->getMergedConfigAndUnknownProperties(
                            {ov::compilation_target(target)}, ::intel_npu::ConfigMergeMode::Compile));

    NPU_TEST_LOG << "Merged config: PLATFORM = " << merged.first.get<::intel_npu::PLATFORM>()
                 << ", unknown properties left over = " << merged.second.size() << std::endl;
    ASSERT_EQ(merged.first.get<::intel_npu::PLATFORM>(), std::string(ov::intel_npu::Platform::NPU6010));
}

TEST_F(CompilationTargetSelectorTests, ForwardsOtherCompilePropertiesUnchanged) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};
    NPU_TEST_LOG << "Merging compilation_target = " << toString(target) << " with tiles = 1" << std::endl;

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(
        merged = propertiesManager->getMergedConfigAndUnknownProperties(
            {ov::compilation_target(target), ov::intel_npu::tiles(1)},
            ::intel_npu::ConfigMergeMode::Compile));

    NPU_TEST_LOG << "Merged config: PLATFORM = " << merged.first.get<::intel_npu::PLATFORM>() << ", TILES = "
                 << merged.first.get<::intel_npu::TILES>() << std::endl;
    ASSERT_EQ(merged.first.get<::intel_npu::PLATFORM>(), std::string(ov::intel_npu::Platform::NPU6010));
    ASSERT_EQ(merged.first.get<::intel_npu::TILES>(), 1);
}

TEST_F(CompilationTargetSelectorTests, RejectsBeingCombinedWithAnExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};
    NPU_TEST_LOG << "Merging compilation_target = " << toString(target)
                 << " together with an explicit NPU_PLATFORM=3720" << std::endl;

    try {
        propertiesManager->getMergedConfigAndUnknownProperties(
            {ov::compilation_target(target),
             ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU3720))},
            ::intel_npu::ConfigMergeMode::Compile);
        FAIL() << "Expected the merge to throw";
    } catch (const ov::Exception& e) {
        NPU_TEST_LOG << "Merge threw as expected: \"" << e.what() << "\"" << std::endl;
        ASSERT_NE(std::string(e.what()).find("cannot be combined"), std::string::npos);
    }
}

}  // namespace
