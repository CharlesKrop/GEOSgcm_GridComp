esma_set_this()

option(BUILD_SERIALBOX_SER "Build with Serialbox serialization statements" OFF)

set (srcs
  GEOS_TurbulenceGridComp.F90
  LockEntrain.F90
  shoc.F90
  edmf.F90
  scm_surface.F90
  )

esma_add_library (${this}
  SRCS ${srcs}
  DEPENDENCIES GEOS_Shared MAPL ESMF::ESMF NetCDF::NetCDF_Fortran)

if(BUILD_SERIALBOX_SER)
  message(STATUS "Building with Serialbox serialization statements")
  message(STATUS "  Serialbox root ${SERIALBOX_ROOT}")
  add_definitions(-DSERIALIZE)
  target_include_directories(${this} PRIVATE ${SERIALBOX_ROOT}/include)
  target_link_directories(${this} PRIVATE ${SERIALBOX_ROOT}/lib/)
  find_library(SERIALBOX_FORTRAN_LIB
      NAMES SerialboxFortran
      HINTS ${SERIALBOX_ROOT}/lib/
      REQUIRED
  )
  find_library(SERIALBOX_C_LIB
      NAMES SerialboxC
      HINTS ${SERIALBOX_ROOT}/lib/
      REQUIRED
  )
  message(STATUS "  Serialbox Fortran library found: ${SERIALBOX_FORTRAN_LIB}")
  message(STATUS "  Serialbox C library found: ${SERIALBOX_C_LIB}")
  target_link_libraries(${this} PRIVATE ${SERIALBOX_FORTRAN_LIB} ${SERIALBOX_C_LIB})
endif()
