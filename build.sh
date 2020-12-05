cd `dirname $0`

# Select the cpu CMakeList
mv CMakeLists_gpu.txt CMakeLists.txt

# Compile code.
mkdir -p build
cd build
cmake ..
make -j `nproc` $*


# Switch back the CMakeList
cd ..
mv CMakeLists.txt CMakeLists_gpu.txt
