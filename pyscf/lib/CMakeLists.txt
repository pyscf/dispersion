# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.22)
project (pyscf-dispersion)

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp -ffree-line-length-none -Wall")

include(ExternalProject)
# statically compile deps mctc mstore
ExternalProject_Add(dftd3_static
  GIT_REPOSITORY "https://github.com/dftd3/simple-dftd3"
  # DFT-D3 v1.7.0
  GIT_TAG 84cb7cddd8d6669c4ecd59b06d53e0d373656239
  GIT_SHALLOW FALSE
  GIT_PROGRESS TRUE
  PREFIX ${PROJECT_BINARY_DIR}/deps
  INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
  CMAKE_ARGS -DWITH_OpenMP=OFF
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_INSTALL_LIBDIR:PATH=lib
          -DCMAKE_CURRENT_SOURCE_DIR:PATH=dftd3
)
ExternalProject_Add(dftd3
  GIT_REPOSITORY "https://github.com/dftd3/simple-dftd3"
  # DFT-D3 v1.7.0
  GIT_TAG 84cb7cddd8d6669c4ecd59b06d53e0d373656239
  GIT_SHALLOW FALSE
  GIT_PROGRESS TRUE
  PREFIX ${PROJECT_BINARY_DIR}/deps
  INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
  CMAKE_ARGS -DWITH_OpenMP=OFF -DBUILD_SHARED_LIBS=ON
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_INSTALL_LIBDIR:PATH=${PROJECT_SOURCE_DIR}
          -DCMAKE_CURRENT_SOURCE_DIR:PATH=dftd3
)
add_dependencies(dftd3 dftd3_static)

# statically compile deps mctc mstore
ExternalProject_Add(dftd4_static
  GIT_REPOSITORY "https://github.com/dftd4/dftd4"
  # DFT-D4 v4.3.1
  GIT_TAG 1107ea7c7c2131719e306fd57a5f4a1579e2577e
  GIT_SHALLOW FALSE
  GIT_PROGRESS TRUE
  PREFIX ${PROJECT_BINARY_DIR}/deps
  INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
  CMAKE_ARGS -DWITH_OpenMP=OFF
          -DCMAKE_Fortran_FLAGS=-cpp\ -ffree-line-length-none\ -Wall
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_INSTALL_LIBDIR:PATH=lib
)
ExternalProject_Add(dftd4
  GIT_REPOSITORY "https://github.com/dftd4/dftd4"
  # DFT-D4 v4.3.1
  GIT_TAG 1107ea7c7c2131719e306fd57a5f4a1579e2577e
  GIT_SHALLOW FALSE
  GIT_PROGRESS TRUE
  PREFIX ${PROJECT_BINARY_DIR}/deps
  INSTALL_DIR ${PROJECT_SOURCE_DIR}/deps
  CMAKE_ARGS -DWITH_OpenMP=OFF -DBUILD_SHARED_LIBS=ON
          -DCMAKE_Fortran_FLAGS=-cpp\ -ffree-line-length-none\ -Wall
          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
          -DCMAKE_INSTALL_LIBDIR:PATH=${PROJECT_SOURCE_DIR}
)
add_dependencies(dftd4 dftd4_static)
