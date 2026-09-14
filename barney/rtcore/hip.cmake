set(CMAKE_HIP_PLATFORM amd)
enable_language(HIP)
# enable_language(HIP) above auto-detects the host GPU arch (and errors on a
# no-GPU build host); pass -DCMAKE_HIP_ARCHITECTURES=... to override.
message("#barney: building GPU code with HIP for CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
# the cuda backend's GPU code is reused under HIP; OptiX is NVIDIA-only.
set(BARNEY_HAVE_HIP ON)
set(BARNEY_HAVE_CUDA OFF)
set(CMAKE_CUDA_ARCHITECTURES)
if (WIN32)
  # On Windows, CMake's Windows-Clang platform module injects -fuse-ld=lld-link
  # into HIP link commands, but the AMD clang driver rejects it when doing HIP
  # device-link (--hip-link); lld-link is the default host linker already.
  set(CMAKE_HIP_USING_LINKER_DEFAULT "")
endif()

# Optional hardware-RT backend on AMD GPUs via AMD HIPRT. HIPRT supplies the
# BVH build + ray traversal (hardware-accelerated on RDNA2+, software on CDNA
# such as gfx90a); barney keeps its function-pointer shading dispatch. HIPRT
# is a discovered dependency (find via hiprt_ROOT / HIPRT_PATH), never
# vendored -- mirrors how barney finds OptiX/OIDN.
option(BARNEY_BACKEND_HIPRT "Enable HIPRT hardware-RT backend (AMD)?" OFF)
if (BARNEY_BACKEND_HIPRT)
  find_package(hiprt REQUIRED)
  message("#barney: HIPRT backend enabled (hiprt at ${hiprt_LIBRARY})")
endif()

# On AMD GPUs the cuda backend's .cu sources are compiled as HIP. Mark the
# given sources LANGUAGE HIP so hipcc handles the device code (no effect on the
# CUDA/OptiX build, where this macro is never invoked).
macro(configure_source)
  foreach(src ${ARGN})
    get_filename_component(ext "${src}" EXT)
    if (ext STREQUAL ".cu")
      set_source_files_properties(${src} PROPERTIES
        LANGUAGE HIP
        #        COMPILE_OPTIONS "-fgpu-rdc")
        COMPILE_OPTIONS "-fgpu-rdc --hip-link  -mprintf-kind=hostcall")
    endif()
    set_source_files_properties(${src} PROPERTIES
      LANGUAGE HIP
      COMPILE_OPTIONS "-fgpu-rdc --hip-link  -mprintf-kind=hostcall")
  endforeach()
endmacro()


function(rtc_library_properties lib)
  set_target_properties(${lib}
    PROPERTIES
    POSITION_INDEPENDENT_CODE    ON
    VISIBILITY_INLINES_HIDDEN    ON
    HIP_SEPARABLE_COMPILATION   ON
#    HIP_USE_STATIC_CUDA_RUNTIME ON
    HIP_RESOLVE_DEVICE_SYMBOLS  ON
#    CUDA_VISIBILITY_PRESET       hidden
    CXX_VISIBILITY_PRESET        hidden
  )
  if (APPLE)
    set_target_properties(${lib} PROPERTIES INSTALL_RPATH "$loader_path")
  else()
    set_target_properties(${lib} PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
  target_link_options(${lib} PUBLIC
    "-fgpu-rdc;--hip-link;-mprintf-kind=hostcall"
#    "-fgpu-rdc"
  )
endfunction()


function(rtc_build_device_sources libname)
  set(DEVICE_PROGRAM_SOURCES ${ARGN})
  add_library(${libname} OBJECT
    ${DEVICE_PROGRAM_SOURCES}
  )
  target_compile_definitions(${libname} PRIVATE
    -DBARNEY_DEVICE_PROGRAM=1)
  target_link_libraries(${libname}
    barney_rtc_hip
    barney_rtc_cudaCommon_${backend}
    barney_config_${backend}
    )
  rtc_library_properties(${libname})
endfunction()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cudaCommon build_cudaCommon_hip)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/hip        build_cuda_hip)




