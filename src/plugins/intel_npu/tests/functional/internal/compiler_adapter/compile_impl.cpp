// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "plugin_property_manager.hpp"

namespace {

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
    ASSERT_TRUE(isSupported);
}

TEST_F(OfflineCompilationTargetsTests, EnumeratesEveryExportablePlatform) {
    ov::Any value;
    OV_ASSERT_NO_THROW(value = propertiesManager->getProperty(ov::offline_compilation_targets.name()));
    const auto targets = value.as<std::vector<ov::CompilationTarget>>();

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
    ASSERT_GT(value.as<std::vector<ov::CompilationTarget>>().size(), 1u);
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
    ASSERT_TRUE(isSupported);
}

TEST_F(CompilationTargetSelectorTests, ThrowsWhenQueriedBeforeBeingSet) {
    OV_EXPECT_THROW_HAS_SUBSTRING(propertiesManager->getProperty(ov::compilation_target.name()),
                                 ov::Exception,
                                 "was not provided");
}

TEST_F(CompilationTargetSelectorTests, InjectsItsPlatformIntoTheMergedConfig) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(merged = propertiesManager->getMergedConfigAndUnknownProperties(
                            {ov::compilation_target(target)}, ::intel_npu::ConfigMergeMode::Compile));

    ASSERT_EQ(merged.first.get<::intel_npu::PLATFORM>(), std::string(ov::intel_npu::Platform::NPU6010));
}

TEST_F(CompilationTargetSelectorTests, ForwardsOtherCompilePropertiesUnchanged) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};

    std::pair<::intel_npu::Config, ov::AnyMap> merged{::intel_npu::Config{options}, {}};
    OV_ASSERT_NO_THROW(
        merged = propertiesManager->getMergedConfigAndUnknownProperties(
            {ov::compilation_target(target), ov::intel_npu::tiles(1)},
            ::intel_npu::ConfigMergeMode::Compile));

    ASSERT_EQ(merged.first.get<::intel_npu::PLATFORM>(), std::string(ov::intel_npu::Platform::NPU6010));
    ASSERT_EQ(merged.first.get<::intel_npu::TILES>(), 1);
}

TEST_F(CompilationTargetSelectorTests, RejectsBeingCombinedWithAnExplicitPlatform) {
    const ov::CompilationTarget target{std::string(ov::intel_npu::Platform::NPU6010), {0xD71D}};

    OV_EXPECT_THROW_HAS_SUBSTRING(
        propertiesManager->getMergedConfigAndUnknownProperties(
            {ov::compilation_target(target), ov::intel_npu::platform(std::string(ov::intel_npu::Platform::NPU3720))},
            ::intel_npu::ConfigMergeMode::Compile),
        ov::Exception,
        "cannot be combined");
}

}  // namespace
