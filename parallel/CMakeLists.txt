cmake_minimum_required (VERSION 2.6)
project (Parallel_Test)
cmake_policy(SET CMP0022 OLD)
cmake_policy(SET CMP0023 OLD)
cmake_policy(SET CMP0011 OLD)

MESSAGE ("HPX_DIR:" ${HPX_DIR})
set (CMAKE_CXX_STANDARD 14)

add_definitions( -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBOOST_LOG_DYN_LINK=1 )
add_compile_options(-Wall -g -O3 )

find_package(HPX REQUIRED PATHS ${HPX_ROOT} ${HPX_DIR} NO_DEFAULT_PATH)

set (TEST_SOURCES matrix2.cc )

add_executable(random_test parallel_test.cc)
add_executable(random_basic_test parallel_basic.cc)
add_executable(random_extended_test parallel_extended.cc)


set(BUILD_TYPE "RelWithDebInfo")


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options(-Wno-unused-lambda-capture -Wno-unused-variable -Wno-unknown-pragmas -Wno-unused-local-typedefs)
  set (GTestLib  "libgtest.clang.a" )
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-ftree-vectorize -fno-signed-zeros -fno-trapping-math -Wno-unused-local-typedefs)
  set (GTestLib "libgtest.gcc.a" )
endif()


  set (Boost_Lib_Path "${BOOST_ROOT}/stage/lib")



  include_directories("./external/include/gtest/include"
    "/local/src/hpx_test/matrix/external/include/"
    "/local/src/hpx_test/matrix/external/include/gtest/"
    ${BOOST_ROOT}
    )

set (LibPath ${PROJECT_SOURCE_DIR}/external/lib, ${Boost_Lib_Path} )

link_directories( ${Boost_Lib_Path}  ${LibPath} ${HPX_Root}/lib ${MKL_Lib_Path}  ${Intel_Compiler_Lib_Path} ${Boost_LIBRARY_DIR} )

hpx_setup_target(random_test)
hpx_setup_target(random_basic_test)
hpx_setup_target(random_extended_test)

