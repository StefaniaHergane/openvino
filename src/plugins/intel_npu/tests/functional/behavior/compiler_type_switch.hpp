// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "common/functions.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

void printCompilerVersion(const ov::CompiledModel& compiledModel, const std::string& compilerName) {
    const auto compilerVersion = compiledModel.get_property(ov::intel_npu::compiler_version);
    std::cout << compilerName << " compiler version: " << (compilerVersion >> 16) << "."
              << (compilerVersion & 0xFFFF) << std::endl;
}

class TestCompilerTypeSwitch : public ov::test::behavior::OVPluginTestBase,
                               public testing::WithParamInterface<std::string> {
public:
    void SetUp() override {
        target_device = GetParam();
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<std::string>& obj) {
        std::string targetDevice = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

TEST_P(TestCompilerTypeSwitch, compileSequentiallyWithBothCompilersPluginFirst) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& model = buildSingleLayerSoftMaxNetwork();

        std::getchar();

        // First: compile with PLUGIN compiler
        {
            ov::AnyMap configuration;
            configuration[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;

            ov::CompiledModel compiledModel;
            std::cout << "Compiling with PLUGIN compiler..." << std::endl;
            std::getchar();
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
            OV_ASSERT_NO_THROW(printCompilerVersion(compiledModel, "PLUGIN"));
            std::cout << "Done" << std::endl;
            std::getchar();
            std::stringstream modelStream;
            OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));
        }

        // Second: compile with DRIVER compiler
        {
            ov::AnyMap configuration;
            configuration[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::DRIVER;

            ov::CompiledModel compiledModel;
            std::cout << "Compiling with DRIVER compiler..." << std::endl;
            std::getchar();
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
            OV_ASSERT_NO_THROW(printCompilerVersion(compiledModel, "DRIVER"));
            std::cout << "Done" << std::endl;
            std::getchar();
            std::stringstream modelStream;
            OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));
        }
    }
}

TEST_P(TestCompilerTypeSwitch, compileSequentiallyWithBothCompilersDriverFirst) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& model = buildSingleLayerSoftMaxNetwork();

        // First: compile with DRIVER compiler
        {
            ov::AnyMap configuration;
            configuration[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::DRIVER;

            ov::CompiledModel compiledModel;
            std::cout << "Compiling with DRIVER compiler..." << std::endl;
            std::getchar();
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
            OV_ASSERT_NO_THROW(printCompilerVersion(compiledModel, "DRIVER"));
            std::cout << "Done" << std::endl;
            std::getchar();
            std::stringstream modelStream;
            OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));
        }

        // Second: compile with PLUGIN compiler
        {
            ov::AnyMap configuration;
            configuration[ov::intel_npu::compiler_type.name()] = ov::intel_npu::CompilerType::PLUGIN;

            ov::CompiledModel compiledModel;
            std::cout << "Compiling with PLUGIN compiler..." << std::endl;
            std::getchar();
            OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
            OV_ASSERT_NO_THROW(printCompilerVersion(compiledModel, "PLUGIN"));
            std::cout << "Done" << std::endl;
            std::getchar();
            std::stringstream modelStream;
            OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));
        }
    }
}


}  // namespace
