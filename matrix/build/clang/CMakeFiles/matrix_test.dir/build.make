# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shmuel/src/hpx_test/matrix

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shmuel/src/hpx_test/matrix/build/clang

# Include any dependencies generated for this target.
include CMakeFiles/matrix_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_test.dir/flags.make

CMakeFiles/matrix_test.dir/matrix.cc.o: CMakeFiles/matrix_test.dir/flags.make
CMakeFiles/matrix_test.dir/matrix.cc.o: ../../matrix.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shmuel/src/hpx_test/matrix/build/clang/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/matrix_test.dir/matrix.cc.o"
	/usr/local/bin/clang++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/matrix_test.dir/matrix.cc.o -c /home/shmuel/src/hpx_test/matrix/matrix.cc

CMakeFiles/matrix_test.dir/matrix.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_test.dir/matrix.cc.i"
	/usr/local/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shmuel/src/hpx_test/matrix/matrix.cc > CMakeFiles/matrix_test.dir/matrix.cc.i

CMakeFiles/matrix_test.dir/matrix.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_test.dir/matrix.cc.s"
	/usr/local/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shmuel/src/hpx_test/matrix/matrix.cc -o CMakeFiles/matrix_test.dir/matrix.cc.s

CMakeFiles/matrix_test.dir/matrix.cc.o.requires:

.PHONY : CMakeFiles/matrix_test.dir/matrix.cc.o.requires

CMakeFiles/matrix_test.dir/matrix.cc.o.provides: CMakeFiles/matrix_test.dir/matrix.cc.o.requires
	$(MAKE) -f CMakeFiles/matrix_test.dir/build.make CMakeFiles/matrix_test.dir/matrix.cc.o.provides.build
.PHONY : CMakeFiles/matrix_test.dir/matrix.cc.o.provides

CMakeFiles/matrix_test.dir/matrix.cc.o.provides.build: CMakeFiles/matrix_test.dir/matrix.cc.o


# Object files for target matrix_test
matrix_test_OBJECTS = \
"CMakeFiles/matrix_test.dir/matrix.cc.o"

# External object files for target matrix_test
matrix_test_EXTERNAL_OBJECTS =

matrix_test: CMakeFiles/matrix_test.dir/matrix.cc.o
matrix_test: CMakeFiles/matrix_test.dir/build.make
matrix_test: /libgtest.clang.a
matrix_test: /opt/hpx/1.0/clang/lib/libhpx_init.a
matrix_test: /opt/hpx/1.0/clang/lib/libhpx_init.a
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_filesystem.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_program_options.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_regex.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_context.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_random.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
matrix_test: /usr/lib/x86_64-linux-gnu/libhwloc.so
matrix_test: /opt/intel/vtune_amplifier_xe/lib64/libittnotify.a
matrix_test: /usr/lib/libmpi.so
matrix_test: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
matrix_test: /usr/lib/libmpi.so
matrix_test: /opt/hpx/1.0/clang/lib/libhpx_init.a
matrix_test: /opt/hpx/1.0/clang/lib/libhpx.so.1.0.0
matrix_test: /opt/hpx/1.0/clang/lib/libhpx_init.a
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_filesystem.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_program_options.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_regex.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_context.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_random.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
matrix_test: /usr/lib/x86_64-linux-gnu/libhwloc.so
matrix_test: /opt/intel/vtune_amplifier_xe/lib64/libittnotify.a
matrix_test: /usr/lib/libmpi.so
matrix_test: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
matrix_test: /usr/lib/libmpi.so
matrix_test: /libgtest.clang.a
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_filesystem.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_program_options.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_regex.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_context.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_random.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_chrono.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_date_time.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_filesystem.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_program_options.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_regex.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_system.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_thread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_atomic.so
matrix_test: /usr/lib/x86_64-linux-gnu/libpthread.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_context.so
matrix_test: /usr/src/boost/1.63/clang/stage/lib/libboost_random.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
matrix_test: /usr/lib/x86_64-linux-gnu/libtbbmalloc_proxy.so
matrix_test: /usr/lib/x86_64-linux-gnu/libhwloc.so
matrix_test: /opt/intel/vtune_amplifier_xe/lib64/libittnotify.a
matrix_test: /usr/lib/libmpi.so
matrix_test: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
matrix_test: /usr/lib/libmpi.so
matrix_test: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
matrix_test: CMakeFiles/matrix_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shmuel/src/hpx_test/matrix/build/clang/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable matrix_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_test.dir/build: matrix_test

.PHONY : CMakeFiles/matrix_test.dir/build

CMakeFiles/matrix_test.dir/requires: CMakeFiles/matrix_test.dir/matrix.cc.o.requires

.PHONY : CMakeFiles/matrix_test.dir/requires

CMakeFiles/matrix_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix_test.dir/clean

CMakeFiles/matrix_test.dir/depend:
	cd /home/shmuel/src/hpx_test/matrix/build/clang && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shmuel/src/hpx_test/matrix /home/shmuel/src/hpx_test/matrix /home/shmuel/src/hpx_test/matrix/build/clang /home/shmuel/src/hpx_test/matrix/build/clang /home/shmuel/src/hpx_test/matrix/build/clang/CMakeFiles/matrix_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_test.dir/depend

