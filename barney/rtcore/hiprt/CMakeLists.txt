set(HIPRT_SOURCES
  Buffer.cpp
  Device.cpp
  Geom.cpp
  GeomType.cpp
  Group.cpp
  TraceKernel.cpp
)
set_source_files_properties(${HIPRT_SOURCES} PROPERTIES LANGUAGE HIP COMPILE_OPTIONS "-fgpu-rdc")

add_library(barney_rtc_${backend} STATIC ${HIPRT_SOURCES})
target_link_libraries(barney_rtc_${backend} PUBLIC
  barney_config_${backend}
  barney_rtc_cudaCommon_${backend}
  hiprt::hiprt
)
target_include_directories(barney_rtc_${backend} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)
target_compile_definitions(barney_rtc_${backend} PUBLIC
  BARNEY_RTC_HIPRT=1
)
rtc_library_properties(barney_rtc_${backend})
