// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Exercises the private NPU_DEVICE_DESCRIPTION and NPU_COMPILATION_DEVICE_DESCRIPTION properties of the NPU plugin.
//
// Usage:
//   test_app [DEVICE_ID]
//       Reads NPU_DEVICE_DESCRIPTION from the device (needs an NPU). DEVICE_ID is optional: a device name or index
//       (e.g. "0", "4000"); the default device is used when omitted.
//   test_app --parse <DESCRIPTOR>
//       Parses a descriptor string, e.g. "device=0x643e,stepping=4,tiles=6", and prints its fields (no NPU needed).
//   test_app --compile <MODEL> [DESCRIPTOR] [--output <BLOB>]
//       Compiles MODEL (an OpenVINO IR .xml, or any format ov::Core reads) for a device description, passed through
//       NPU_COMPILATION_DEVICE_DESCRIPTION. Without DESCRIPTOR, it is first read from the local NPU through
//       NPU_DEVICE_DESCRIPTION, as in the two stages of hardware-free compilation. With DESCRIPTOR, no NPU is needed.
//       Then tries one inference, which only works when the descriptor names the local device, and optionally
//       exports the compiled blob to BLOB.

#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include "intel_npu/npu_private_properties.hpp"
#include "openvino/openvino.hpp"

namespace {

constexpr const char* DEVICE_NPU = "NPU";

void printDescription(const ov::intel_npu::DeviceDescription& description) {
    std::cout << "  device   (deviceId)    : 0x" << std::hex << description.deviceId << std::dec << "\n"
              << "  stepping (subdeviceId) : " << description.subdeviceId << "\n"
              << "  tiles    (numSlices)   : " << description.numSlices << "\n";
}

void printUsage(const char* program) {
    std::cerr << "Usage:\n"
              << "  " << program << " [DEVICE_ID]\n"
              << "  " << program << " --parse <DESCRIPTOR>\n"
              << "  " << program << " --compile <MODEL> [DESCRIPTOR] [--output <BLOB>]\n";
}

struct CompileOptions {
    std::string modelPath;
    std::string descriptor;  // empty: read it from the local NPU
    std::string outputPath;  // empty: do not export
};

// Parses "--compile <MODEL> [DESCRIPTOR] [--output <BLOB>]"; returns false on malformed arguments.
bool parseCompileOptions(int argc, char* argv[], CompileOptions& options) {
    if (argc < 3) {
        return false;
    }
    options.modelPath = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--output") {
            if (i + 1 >= argc || !options.outputPath.empty()) {
                return false;
            }
            options.outputPath = argv[++i];
        } else if (options.descriptor.empty()) {
            options.descriptor = argument;
        } else {
            return false;
        }
    }
    return true;
}

// Runs one inference on zero-filled inputs. Returns false when the compiled model cannot run on this machine, which
// is expected when the device description names another device or when no driver is present.
bool tryInference(ov::CompiledModel& compiledModel) {
    try {
        auto request = compiledModel.create_infer_request();
        for (const auto& input : compiledModel.inputs()) {
            auto tensor = request.get_tensor(input);
            if (tensor.get_byte_size() > 0) {
                std::memset(tensor.data(), 0, tensor.get_byte_size());
            }
        }
        request.infer();
        return true;
    } catch (const std::exception& ex) {
        std::cout << "  " << ex.what() << "\n";
        return false;
    }
}

// Compiles a model for a device description: stage 1 (read the descriptor) only when none is given, then stage 2.
int compileForDescription(const CompileOptions& options) {
    ov::Core core;
    std::cout << "OpenVINO " << ov::get_openvino_version().buildNumber << "\n";

    // Stage 1: capture the descriptor while hardware access is available, as the text an application would persist
    std::string descriptor = options.descriptor;
    if (descriptor.empty()) {
        descriptor = core.get_property(DEVICE_NPU, ov::intel_npu::device_description.name()).as<std::string>();
        std::cout << "\n[stage 1] " << ov::intel_npu::device_description.name() << ": " << descriptor << "\n";
    } else {
        std::cout << "\n[stage 1] skipped, descriptor given: " << descriptor << "\n";
    }
    printDescription(ov::Any(descriptor).as<ov::intel_npu::DeviceDescription>());

    // Stage 2: compile for the persisted descriptor, passed back unchanged as text
    std::cout << "\n[stage 2] compiling " << options.modelPath << " with "
              << ov::intel_npu::compilation_device_description.name() << "\n";
    auto compiledModel =
        core.compile_model(options.modelPath,
                           DEVICE_NPU,
                           ov::AnyMap{{ov::intel_npu::compilation_device_description.name(), descriptor}});
    std::cout << "  compiled\n";

    try {
        std::cout << "  " << ov::runtime_requirements.name() << ": "
                  << compiledModel.get_property(ov::runtime_requirements) << "\n";
    } catch (const std::exception&) {
        std::cout << "  " << ov::runtime_requirements.name() << ": <not available>\n";
    }

    std::cout << "\n[inference]\n";
    const bool inferenceOk = tryInference(compiledModel);
    std::cout << "  " << (inferenceOk ? "OK" : "not available on this machine") << "\n";

    if (!options.outputPath.empty()) {
        std::ofstream blob(options.outputPath, std::ios::binary);
        if (!blob) {
            std::cerr << "Error: cannot open " << options.outputPath << " for writing\n";
            return EXIT_FAILURE;
        }
        compiledModel.export_model(blob);
        std::cout << "\n[export] " << options.outputPath << "\n";
    }
    return EXIT_SUCCESS;
}

// Parses a descriptor string the way the plugin would, without touching the device.
int parseDescriptor(const std::string& descriptor) {
    std::cout << "[parse] " << descriptor << "\n";
    const auto description = ov::Any(descriptor).as<ov::intel_npu::DeviceDescription>();
    printDescription(description);

    // Serializing the parsed value must give back the canonical text form.
    std::cout << "\n[canonical] " << ov::Any(description).as<std::string>() << "\n";
    return EXIT_SUCCESS;
}

// Reads NPU_DEVICE_DESCRIPTION from the device, typed and as a string, and checks the two agree.
int readFromDevice(const ov::AnyMap& arguments) {
    ov::Core core;
    std::cout << "OpenVINO " << ov::get_openvino_version().buildNumber << "\n";

    const auto availableDevices = core.get_property(DEVICE_NPU, ov::available_devices);
    std::cout << "Available NPU devices:";
    for (const auto& device : availableDevices) {
        std::cout << " " << device;
    }
    std::cout << (availableDevices.empty() ? " <none>\n" : "\n");

    const auto deviceIdIt = arguments.find(ov::device::id.name());
    if (deviceIdIt != arguments.end()) {
        std::cout << "Querying DEVICE_ID: " << deviceIdIt->second.as<std::string>() << "\n";
    }

    // 1. Typed read: the property value is the struct itself.
    const auto description = core.get_property(DEVICE_NPU, ov::intel_npu::device_description.name(), arguments)
                                 .as<ov::intel_npu::DeviceDescription>();
    std::cout << "\n[typed] " << ov::intel_npu::device_description.name() << ":\n";
    printDescription(description);

    // 2. String read: goes through operator<<.
    const auto serialized =
        core.get_property(DEVICE_NPU, ov::intel_npu::device_description.name(), arguments).as<std::string>();
    std::cout << "\n[string] " << serialized << "\n";

    // 3. Round trip: parse the string back through operator>> and compare with the typed value.
    const auto parsed = ov::Any(serialized).as<ov::intel_npu::DeviceDescription>();
    const bool roundTripOk = parsed == description;
    std::cout << "\n[round trip] " << (roundTripOk ? "OK" : "MISMATCH") << "\n";
    if (!roundTripOk) {
        printDescription(parsed);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::string firstArgument = argc > 1 ? argv[1] : "";
        if (firstArgument == "-h" || firstArgument == "--help") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (firstArgument == "--parse") {
            if (argc != 3) {
                printUsage(argv[0]);
                return EXIT_FAILURE;
            }
            return parseDescriptor(argv[2]);
        }
        if (firstArgument == "--compile") {
            CompileOptions options;
            if (!parseCompileOptions(argc, argv, options)) {
                printUsage(argv[0]);
                return EXIT_FAILURE;
            }
            return compileForDescription(options);
        }
        if (argc > 2) {
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }

        ov::AnyMap arguments;
        if (!firstArgument.empty()) {
            arguments.emplace(ov::device::id.name(), firstArgument);
        }
        return readFromDevice(arguments);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
