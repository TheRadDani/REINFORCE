# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/daniel/REINFORCE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/daniel/REINFORCE/build

# Include any dependencies generated for this target.
include CMakeFiles/reinforce.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/reinforce.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/reinforce.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reinforce.dir/flags.make

CMakeFiles/reinforce.dir/src/main.cpp.o: CMakeFiles/reinforce.dir/flags.make
CMakeFiles/reinforce.dir/src/main.cpp.o: /home/daniel/REINFORCE/src/main.cpp
CMakeFiles/reinforce.dir/src/main.cpp.o: CMakeFiles/reinforce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/daniel/REINFORCE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/reinforce.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reinforce.dir/src/main.cpp.o -MF CMakeFiles/reinforce.dir/src/main.cpp.o.d -o CMakeFiles/reinforce.dir/src/main.cpp.o -c /home/daniel/REINFORCE/src/main.cpp

CMakeFiles/reinforce.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/reinforce.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/REINFORCE/src/main.cpp > CMakeFiles/reinforce.dir/src/main.cpp.i

CMakeFiles/reinforce.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/reinforce.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/REINFORCE/src/main.cpp -o CMakeFiles/reinforce.dir/src/main.cpp.s

CMakeFiles/reinforce.dir/src/policy_network.cpp.o: CMakeFiles/reinforce.dir/flags.make
CMakeFiles/reinforce.dir/src/policy_network.cpp.o: /home/daniel/REINFORCE/src/policy_network.cpp
CMakeFiles/reinforce.dir/src/policy_network.cpp.o: CMakeFiles/reinforce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/daniel/REINFORCE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/reinforce.dir/src/policy_network.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reinforce.dir/src/policy_network.cpp.o -MF CMakeFiles/reinforce.dir/src/policy_network.cpp.o.d -o CMakeFiles/reinforce.dir/src/policy_network.cpp.o -c /home/daniel/REINFORCE/src/policy_network.cpp

CMakeFiles/reinforce.dir/src/policy_network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/reinforce.dir/src/policy_network.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/REINFORCE/src/policy_network.cpp > CMakeFiles/reinforce.dir/src/policy_network.cpp.i

CMakeFiles/reinforce.dir/src/policy_network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/reinforce.dir/src/policy_network.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/REINFORCE/src/policy_network.cpp -o CMakeFiles/reinforce.dir/src/policy_network.cpp.s

CMakeFiles/reinforce.dir/src/grid_world.cpp.o: CMakeFiles/reinforce.dir/flags.make
CMakeFiles/reinforce.dir/src/grid_world.cpp.o: /home/daniel/REINFORCE/src/grid_world.cpp
CMakeFiles/reinforce.dir/src/grid_world.cpp.o: CMakeFiles/reinforce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/daniel/REINFORCE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/reinforce.dir/src/grid_world.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reinforce.dir/src/grid_world.cpp.o -MF CMakeFiles/reinforce.dir/src/grid_world.cpp.o.d -o CMakeFiles/reinforce.dir/src/grid_world.cpp.o -c /home/daniel/REINFORCE/src/grid_world.cpp

CMakeFiles/reinforce.dir/src/grid_world.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/reinforce.dir/src/grid_world.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/REINFORCE/src/grid_world.cpp > CMakeFiles/reinforce.dir/src/grid_world.cpp.i

CMakeFiles/reinforce.dir/src/grid_world.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/reinforce.dir/src/grid_world.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/REINFORCE/src/grid_world.cpp -o CMakeFiles/reinforce.dir/src/grid_world.cpp.s

CMakeFiles/reinforce.dir/src/reinforce.cpp.o: CMakeFiles/reinforce.dir/flags.make
CMakeFiles/reinforce.dir/src/reinforce.cpp.o: /home/daniel/REINFORCE/src/reinforce.cpp
CMakeFiles/reinforce.dir/src/reinforce.cpp.o: CMakeFiles/reinforce.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/daniel/REINFORCE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/reinforce.dir/src/reinforce.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/reinforce.dir/src/reinforce.cpp.o -MF CMakeFiles/reinforce.dir/src/reinforce.cpp.o.d -o CMakeFiles/reinforce.dir/src/reinforce.cpp.o -c /home/daniel/REINFORCE/src/reinforce.cpp

CMakeFiles/reinforce.dir/src/reinforce.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/reinforce.dir/src/reinforce.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/daniel/REINFORCE/src/reinforce.cpp > CMakeFiles/reinforce.dir/src/reinforce.cpp.i

CMakeFiles/reinforce.dir/src/reinforce.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/reinforce.dir/src/reinforce.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/daniel/REINFORCE/src/reinforce.cpp -o CMakeFiles/reinforce.dir/src/reinforce.cpp.s

# Object files for target reinforce
reinforce_OBJECTS = \
"CMakeFiles/reinforce.dir/src/main.cpp.o" \
"CMakeFiles/reinforce.dir/src/policy_network.cpp.o" \
"CMakeFiles/reinforce.dir/src/grid_world.cpp.o" \
"CMakeFiles/reinforce.dir/src/reinforce.cpp.o"

# External object files for target reinforce
reinforce_EXTERNAL_OBJECTS =

reinforce: CMakeFiles/reinforce.dir/src/main.cpp.o
reinforce: CMakeFiles/reinforce.dir/src/policy_network.cpp.o
reinforce: CMakeFiles/reinforce.dir/src/grid_world.cpp.o
reinforce: CMakeFiles/reinforce.dir/src/reinforce.cpp.o
reinforce: CMakeFiles/reinforce.dir/build.make
reinforce: CMakeFiles/reinforce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/daniel/REINFORCE/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable reinforce"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reinforce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reinforce.dir/build: reinforce
.PHONY : CMakeFiles/reinforce.dir/build

CMakeFiles/reinforce.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reinforce.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reinforce.dir/clean

CMakeFiles/reinforce.dir/depend:
	cd /home/daniel/REINFORCE/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/daniel/REINFORCE /home/daniel/REINFORCE /home/daniel/REINFORCE/build /home/daniel/REINFORCE/build /home/daniel/REINFORCE/build/CMakeFiles/reinforce.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/reinforce.dir/depend

