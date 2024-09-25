cmake_minimum_required(VERSION 3.14)
project(llamacpp-embedder C CXX)
include(CheckIncludeFileCXX)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9")
endif()
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
add_subdirectory(vendor/llama.cpp)
set(TARGET llama-embedder)
set(CMAKE_CXX_STANDARD 11)
if (LLAMA_EMBEDDER_BUILD_STATIC)
    add_library(${TARGET} STATIC src/embedder.cpp src/embedder.h)
    install(TARGETS ${TARGET} ARCHIVE)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/static)
    add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET}> ${CMAKE_BINARY_DIR}/static/
    )
    add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:common> ${CMAKE_BINARY_DIR}/static/
    )
    add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:llama> ${CMAKE_BINARY_DIR}/static/
    )
    add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ggml> ${CMAKE_BINARY_DIR}/static/
    )
else()
    add_library(${TARGET} SHARED src/embedder.cpp src/embedder.h)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(${TARGET} PRIVATE -fvisibility=default)
    target_compile_definitions(${TARGET} PRIVATE BUILDING_DLL)
    install(TARGETS ${TARGET} LIBRARY)
endif()
target_include_directories(${TARGET} PUBLIC .)
target_compile_features(${TARGET} PUBLIC cxx_std_11)
target_link_libraries(${TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})

# Add GoogleTest dependency
# Add Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add the test file
add_executable(embedder_test test/cpp/test_embedder.cpp src/embedder.h)

if(APPLE)
    # build for latest OXS due to gtest version
    set_target_properties(embedder_test PROPERTIES OSX_DEPLOYMENT_TARGET "14.0")
endif()
# Link GoogleTest libraries
target_link_libraries(embedder_test PRIVATE ${TARGET} common llama ${CMAKE_THREAD_LIBS_INIT} gtest gtest_main)
# Enable testing
enable_testing()
include(GoogleTest)
# Add test to CTest
gtest_discover_tests(embedder_test)
