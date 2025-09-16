// Copyright (C) 2025 Intel Corporation
//

#include <openvino/openvino.hpp>
#include "openvino/opsets/opset11.hpp"

int main() {
    using namespace std;
    ov::Shape input_shape{1};
    auto param_node = make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    shared_ptr<ov::Model> model = make_shared<ov::Model>(param_node, ov::ParameterVector{param_node});
    ov::Core core;
    ov::CompiledModel identity = core.compile_model(model, "NPU", {
        {"NPU_USE_NPUW", "YES"},
        {"NPUW_DEVICES", "CPU"},
        {"NPUW_ONLINE_PIPELINE", "NONE"},
        {"NPU_EXECUTION_MODE_HINT", "PERFORMANCE"},
        //{"NPU_PLATFORM", "3720"},
        //{"NPU_COMPILER_TYPE", "MLIR"},
    });

    cout << "ov::supported_properties: " << '\n';
    for (const auto& cfg : identity.get_property(ov::supported_properties)) {
        cout << cfg << '\n';
        //   what():  Exception from src/inference/src/cpp/compiled_model.cpp:140:
        // Exception from src/plugins/intel_npu/src/plugin/src/properties.cpp:647:
        // Unsupported configuration key: EXECUTION_MODE_HINT
        identity.get_property(cfg);
    }

    cout << "ov::execution_devices: " << '\n';
    for (const auto& device : identity.get_property(ov::execution_devices)) {
        cout << device << '\n';
        //   what():  Exception from src/inference/src/cpp/core.cpp:240:
        // Exception from src/plugins/intel_npu/src/plugin/src/metrics.cpp:168:
        // No available devices
        core.get_property(device, ov::device::full_name);
    }
}