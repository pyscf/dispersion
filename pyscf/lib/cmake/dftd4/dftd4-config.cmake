
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was template.cmake                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../deps" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set("dftd4_WITH_API" ON)
set("dftd4_WITH_API_V2" OFF)
set("dftd4_WITH_OpenMP" OFF)
set(
  "dftd4_INCLUDE_DIRS"
  "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/deps/include"
  "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/deps/include/dftd4/GNU-10.2.1"
)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if(NOT TARGET "dftd4::dftd4")
  include("${CMAKE_CURRENT_LIST_DIR}/dftd4-targets.cmake")

  include(CMakeFindDependencyMacro)

  if(NOT TARGET "OpenMP::OpenMP_Fortran" AND "dftd4_WITH_OpenMP")
    find_dependency("OpenMP")
  endif()

  if(NOT TARGET "BLAS::BLAS")
    find_dependency("BLAS")
  endif()

  if(NOT TARGET "mctc-lib::mctc-lib")
    find_dependency("mctc-lib")
  endif()

  if(NOT TARGET "multicharge::multicharge")
    find_dependency("multicharge")
  endif()
endif()
