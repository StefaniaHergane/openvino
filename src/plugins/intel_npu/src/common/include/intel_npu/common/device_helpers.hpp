// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>

#include "intel_npu/common/npu.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace intel_npu {

namespace utils {
bool isNPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
std::string getPlatformByDeviceName(const std::string_view deviceName);

/**
 * @brief Maps a PCI device ID to the NPU platform name; needs no device to be present.
 * @return The platform name, or ov::intel_npu::Platform::AUTO_DETECT for a device ID that is not known.
 * @throws for the device IDs of legacy NPUs, which are not supported.
 */
std::string getPlatformByDeviceId(const uint32_t deviceId);
std::string getCompilationPlatform(const ov::SoPtr<IEngineBackend>& engineBackend,
                                   const std::string_view platform,
                                   const std::string_view deviceId);

/**
 * @brief Gets the compilation platform for a device description, checked against the requested NPU_PLATFORM.
 * @param description The device to compile for.
 * @param platform The requested NPU_PLATFORM; ov::intel_npu::Platform::AUTO_DETECT when not requested.
 * @return The platform of the description's device ID. ov::intel_npu::Platform::AUTO_DETECT when the plugin does not
 * know the device ID, leaving the compiler to resolve it.
 * @throws if the requested platform differs from the description's, or if a platform is requested together with a
 * device ID the plugin does not know, since the two cannot be checked against each other.
 */
std::string getCompilationPlatform(const ov::intel_npu::DeviceDescription& description,
                                   const std::string_view platform);

/**
 * @brief Gets the device by its ID.
 */
std::shared_ptr<IDevice> getDeviceById(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& deviceId);
std::string getFullDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
std::string getDeviceArchitecture(const ov::SoPtr<IEngineBackend>& engineBackend,
                                  const std::string& specifiedDeviceName);
IDevice::Uuid getDeviceUuid(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
ov::device::LUID getDeviceLUID(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getSteppingNumber(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint32_t getMaxTiles(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);

/**
 * @brief Gets the device descriptor of the specified device.
 * @throws if no device with the given name is available.
 */
ov::intel_npu::DeviceDescription getDeviceDescription(const ov::SoPtr<IEngineBackend>& engineBackend,
                                                      const std::string& specifiedDeviceName);

uint64_t getDeviceAllocMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
uint64_t getDeviceTotalMemSize(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
std::string getDeviceName(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
ov::device::PCIInfo getPciInfo(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);
std::map<ov::element::Type, float> getGops(const ov::SoPtr<IEngineBackend>& engineBackend,
                                           const std::string& specifiedDeviceName);
ov::device::Type getDeviceType(const ov::SoPtr<IEngineBackend>& engineBackend, const std::string& specifiedDeviceName);

/**
 * @brief Gets the optimal number of infer requests in parallel for the given platform and performance mode.
 * @param platform The platform for which to get the optimal number of infer requests.
 * @param performanceMode The performance mode for which to get the optimal number of infer requests.
 * @return The optimal number of infer requests in parallel.
 * @note This is the value provided by the plugin, application should query and consider it, but may supply its own
 * preference for number of parallel requests via dedicated configuration
 */
uint32_t getOptimalNumberOfInferRequestsInParallel(std::string_view platform,
                                                   const ov::hint::PerformanceMode performanceMode);

}  // namespace utils

}  // namespace intel_npu
