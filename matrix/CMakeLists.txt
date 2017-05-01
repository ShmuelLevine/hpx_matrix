cmake_minimum_required (VERSION 2.6)
project (Matrix_Test)
cmake_policy(SET CMP0022 OLD)
cmake_policy(SET CMP0023 OLD)
cmake_policy(SET CMP0011 OLD)

MESSAGE ("HPX_DIR:" ${HPX_DIR})
set (CMAKE_CXX_STANDARD 14)

add_definitions( -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBOOST_LOG_DYN_LINK=1 )
add_compile_options(-Wall -g -O3 -Wno-unused-lambda-capture -Wno-unused-variable -Wno-unknown-pragmas -Wno-unused-local-typedefs -ftree-vectorize -fno-signed-zeros -fno-trapping-math)

find_package(HPX REQUIRED PATHS ${HPX_ROOT} ${HPX_DIR} NO_DEFAULT_PATH)

set (TEST_SOURCES matrix.cc )

add_executable(matrix_test ${TEST_SOURCES})

#set (GTest_Additional_Libs irc pthread)


  #IF (CLANG)
    set (GTestLib  "libgtest.clang.a" )
 # ELSE(CLANG)
 #   set (GTestLib "libgtest.gcc.a" )
 # ENDIF(CLANG )
  set (Boost_Lib_Path "${BOOST_ROOT}/stage/lib")



  include_directories("./external/include/gtest/include"
    "/local/src/hpx_test/matrix/external/include/"
    "/local/src/hpx_test/matrix/external/include/gtest/"
    ${BOOST_ROOT}
    )

set (LibPath ${PROJECT_SOURCE_DIR}/external/lib, ${Boost_Lib_Path}, "/usr/src/boost/1.63/clang/stage/lib")

link_directories( ${Boost_Lib_Path}  ${LibPath} ${HPX_Root}/lib ${MKL_Lib_Path}  ${Intel_Compiler_Lib_Path} ${Boost_LIBRARY_DIR} )


  target_link_libraries (matrix_test
    "/local/src/hpx_test/matrix/external/lib/libgtest.clang.a"
    ${Boost_Lib_Path}/libboost_log.so
    ${GTest_Additional_Libs}
    )


   hpx_setup_target(matrix_test
    )



hpx_setup_target(matrix_test
    )

#add_sanitizers(gtest_all)
