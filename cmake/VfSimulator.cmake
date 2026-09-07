# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

option(PTO_ENABLE_VFSIM_COSTMODEL
       "Enable source-level VfSimulator cost model integration" OFF)

set(PTO_VFSIM_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/3rdparty/VfSimulator"
    CACHE PATH "Path to the VfSimulator source tree")

if(PTO_ENABLE_VFSIM_COSTMODEL)
  # TODO: Export/install embedded VfSimulator targets when the costmodel
  # integration becomes a supported install-tree feature.
  message(WARNING
    "PTO_ENABLE_VFSIM_COSTMODEL=ON is currently supported only for "
    "build-tree development. Install/export is not supported yet because "
    "embedded VfSimulator targets are not exported.")

  if(NOT EXISTS "${PTO_VFSIM_SOURCE_DIR}/README.md")
    message(FATAL_ERROR
      "PTO_ENABLE_VFSIM_COSTMODEL=ON requires VfSimulator sources at "
      "${PTO_VFSIM_SOURCE_DIR}. Run `git submodule update --init "
      "3rdparty/VfSimulator` or set PTO_VFSIM_SOURCE_DIR.")
  endif()
  if(NOT EXISTS "${PTO_VFSIM_SOURCE_DIR}/native/CMakeLists.txt")
    message(FATAL_ERROR
      "PTO_ENABLE_VFSIM_COSTMODEL=ON requires the native C++ VfSimulator "
      "CMake entry at ${PTO_VFSIM_SOURCE_DIR}/native. Update the "
      "3rdparty/VfSimulator submodule to a cpp-native-vfsim-compatible "
      "revision.")
  endif()

  message(STATUS "VfSimulator cost model source: ${PTO_VFSIM_SOURCE_DIR}")
  set(VFSIM_ENABLE_MLIR_PLANNER ON CACHE BOOL
      "Build VfSimulator MLIR IR planner when embedded in PTOAS")
  if(NOT VFSIM_ENABLE_MLIR_PLANNER)
    message(FATAL_ERROR
      "PTO_ENABLE_VFSIM_COSTMODEL=ON requires "
      "VFSIM_ENABLE_MLIR_PLANNER=ON.")
  endif()
  if(NOT TARGET vfsim_native_core)
    add_subdirectory("${PTO_VFSIM_SOURCE_DIR}/native"
                     "${CMAKE_BINARY_DIR}/3rdparty/VfSimulator/native")
  endif()
endif()
