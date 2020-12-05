cd `dirname $0`

# Select the cpu CMakeList
mv CMakeLists_cpu.txt CMakeLists.txt

# Compile code.
mkdir -p build_cpu
cd build_cpu
cmake ..
make -j `nproc` $*

# Switch back the CMakeList
cd ..
mv CMakeLists.txt CMakeLists_cpu.txt
