
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

set("s-dftd3_WITH_API" ON)
set("s-dftd3_WITH_OpenMP" OFF)
set("s-dftd3_WITH_BLAS" ON)
set(
  "s-dftd3_INCLUDE_DIRS"
  "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/deps/include"
  "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/deps/include/s-dftd3/GNU-10.2.1"
)

if(NOT TARGET "s-dftd3::s-dftd3")
  include("${CMAKE_CURRENT_LIST_DIR}/s-dftd3-targets.cmake")

  include(CMakeFindDependencyMacro)

  if(NOT TARGET "OpenMP::OpenMP_Fortran" AND "s-dftd3_WITH_OpenMP")
    find_dependency("OpenMP")
  endif()

  if(NOT TARGET "BLAS::BLAS" AND "s-dftd3_WITH_BLAS")
    find_dependency("BLAS")
  endif()

  if(NOT TARGET "mctc-lib::mctc-lib")
    find_dependency("mctc-lib")
  endif()
endif()
