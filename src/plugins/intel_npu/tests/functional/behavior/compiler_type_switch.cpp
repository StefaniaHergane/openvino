// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiler_type_switch.hpp"

#include "common/utils.hpp"

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         TestCompilerTypeSwitch,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         TestCompilerTypeSwitch::getTestCaseName);

}  // namespace
