#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dftd4::dftd4-lib" for configuration "RelWithDebInfo"
set_property(TARGET dftd4::dftd4-lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dftd4::dftd4-lib PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/libdftd4.so.3.7.0"
  IMPORTED_SONAME_RELWITHDEBINFO "libdftd4.so.3"
  )

list(APPEND _cmake_import_check_targets dftd4::dftd4-lib )
list(APPEND _cmake_import_check_files_for_dftd4::dftd4-lib "/mlx_devbox/users/xiaojie.wu/playground/dispersion/pyscf/lib/libdftd4.so.3.7.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
