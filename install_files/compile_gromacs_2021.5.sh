#!/bin/bash


# '

# COMPILE GROMACS 2021.5

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_gromacs_2021.5.sh

# '


NUM_CPUS=$(lscpu | grep 'CPU(s):' | awk '{print $NF}')

DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

SIMD="AVX2_256"

MPI="OFF" # Recommended to keep this off and install the MPI version with PLUMED (you'll have two active versions: gmx and gmx_mpi)

C_COMPILER="cc"

CXX_COMPILER="c++"

GMX_VERSION="2021.5"

GMX_DIR=${INSTALL_DIR}/gromacs_${GMX_VERSION}


### DETECT GPU INFO FOR GROMACS INSTALLATION

INSTALLATION="${1:-GPU}"

if [[ "${INSTALLATION}" == "CPU" ]]; then #echo "REQUESTED CPU VERSION OF GROMACS"; fi

echo -e "=================================================== "

echo -e "=== CPU INSTALLATION SPECIFIED "

echo -e "=== PROCEEDING WITH CPU VERSION OF GROMACS "

echo -e "=================================================== "

sleep 5

elif [[ ! -f $(type -P nvidia-smi) ]]; then

echo -e "=================================================== "

echo -e "=== NO NVIDIA GPU DETECTED "

echo -e "=== PROCEEDING WITH CPU VERSION OF GROMACS "

echo -e "=================================================== "

sleep 5

INSTALLATION="CPU"

elif [[ "${INSTALLATION}" == "GPU" ]]; then

echo -e "=================================================== "

echo -e "=== GPU INSTALLATION SPECIFIED "

echo -e "=== PROCEEDING WITH GPU VERSION OF GROMACS "

echo -e "=== CHECKING REQUIREMENTS... "

echo -e "=================================================== "

sleep 5

CHECK_COUNTER_NUMGPUS=$(nvidia-smi -q | grep 'Attached GPUs' -c)

CHECK_COUNTER_DRIVERVERSION=$(nvidia-smi -q | grep 'Driver Version' -c)

CHECK_COUNTER_CUDAVERSION=$(nvidia-smi -q | grep 'CUDA Version' -c)

CHECK_NUMGPUS=$(nvidia-smi -q | grep 'Attached GPUs' | awk '{print $NF}')

CHECK_DRIVERVERSION=$(nvidia-smi -q | grep 'Driver Version' | awk '{print $NF}')

CHECK_CUDAVERSION=$(nvidia-smi -q | grep 'CUDA Version' | awk '{print $NF}')

if [[ -f $(type -P nvcc) ]]; then

CHECK_TOOLKITVERSION=$(nvcc --version | tail -n 1 | grep -o cuda_[0-9]*.[0-9]* | sed s/'cuda_'//g)

else

CHECK_TOOLKITVERSION=""

fi

echo -e "=================================================== "

echo -e "=== APPROPRIATE NVIDIA GPU PARAMETERS "

echo -e "=== NUMBER OF NVIDIA GPUs: ${CHECK_NUMGPUS} "

echo -e "=== DRIVER VERSION: ${CHECK_DRIVERVERSION} "

echo -e "=== CUDA VERSION: ${CHECK_CUDAVERSION} "

echo -e "=== CUDA TOOLKIT: ${CHECK_TOOLKITVERSION} "

echo -e "=== PROCEEDING WITH GPU VERSION OF GROMACS "

echo -e "=================================================== "

sleep 5


if [[ ${CHECK_COUNTER_NUMGPUS} != 1 ]] || [[ ${CHECK_COUNTER_DRIVERVERSION} != 1 ]] || [[ ${CHECK_COUNTER_CUDAVERSION} != 1 ]]; then

echo -e "=================================================== "

echo -e "=== POTENTIAL ERRORS WITH NVIDIA GPU DRIVERS "

echo -e "=== EXITING NOW... "

echo -e "=== IT IS RECOMMENDED TO CHECK NVIDIA DRIVERS "

echo -e "=== OTHERWISE INSTALL CPU VERSION WITH: "

echo -e "sudo -E env \"PATH=\${PATH}\" \\ "

echo -e "\"LIBRARY_PATH=\${LIBRARY_PATH}\" \\ "

echo -e "\"LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}\" \\ "

echo -e "bash compile_gromacs_2021.5.sh CPU "

echo -e "=================================================== "

exit 0

elif [[ ${CHECK_TOOLKITVERSION} == "" ]]; then

echo -e "=================================================== "

echo -e "=== POTENTIAL ERRORS WITH CUDA DRIVERS/TOOLKIT "

echo -e "=== EXITING NOW... "

echo -e "=== IT IS RECOMMENDED TO CHECK CUDA "

echo -e "=== OTHERWISE INSTALL CPU VERSION WITH: "

echo -e "sudo -E env \"PATH=\${PATH}\" \\ "

echo -e "\"LIBRARY_PATH=\${LIBRARY_PATH}\" \\ "

echo -e "\"LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}\" \\ "

echo -e "bash compile_gromacs_2021.5.sh CPU "

echo -e "=================================================== "

exit 0

fi

fi


### COMPILE FROM SOURCE

cd ${TMP_DIR} && wget https://ftp.gromacs.org/gromacs/gromacs-${GMX_VERSION}.tar.gz

tar -xzvf gromacs-${GMX_VERSION}.tar.gz && rm -r gromacs-${GMX_VERSION}.tar.gz

cd gromacs-${GMX_VERSION}

mkdir build && cd build

declare -a CMAKE_FLAGS

CMAKE_FLAGS+=(-DGMX_BUILD_OWN_FFTW=ON)

CMAKE_FLAGS+=(-DREGRESSIONTEST_DOWNLOAD=OFF)

CMAKE_FLAGS+=(-DGMX_USE_RDTSCP=ON)

CMAKE_FLAGS+=(-DGMX_DOUBLE=OFF)

CMAKE_FLAGS+=(-DGMX_BUILD_SHARED_EXE=OFF)

CMAKE_FLAGS+=(-DBUILD_SHARED_LIBS=OFF)

CMAKE_FLAGS+=(-DGMX_PREFER_STATIC_LIBS=ON)

CMAKE_FLAGS+=(-DGMX_BUILD_MDRUN_ONLY=OFF)

CMAKE_FLAGS+=(-DGMXAPI=OFF)

CMAKE_FLAGS+=(-DGMX_OPENMP_MAX_THREADS=256)

CMAKE_FLAGS+=(-DGMX_SIMD=${SIMD})

CMAKE_FLAGS+=(-DCMAKE_C_COMPILER=${C_COMPILER})

CMAKE_FLAGS+=(-DCMAKE_CXX_COMPILER=${CXX_COMPILER})

CMAKE_FLAGS+=(-DCMAKE_INSTALL_PREFIX=${GMX_DIR})

if [[ "${MPI}" == "ON" ]]; then

CMAKE_FLAGS+=(-DGMX_MPI=ON)

fi

if [[ "${INSTALLATION}" == "GPU" ]]; then

CUDA_DIR=/usr/local/cuda

CMAKE_FLAGS+=(-DGMX_GPU=CUDA)

CMAKE_FLAGS+=(-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_DIR})

CMAKE_FLAGS+=(-DGMX_CUDA_TARGET_SM="60;61;62;70;75;80;86;89")

CMAKE_FLAGS+=(-DGMX_CUDA_TARGET_COMPUTE="60;61;62;70;75;80;86;89")

fi

cmake .. ${CMAKE_FLAGS[@]}

make -j ${NUM_CPUS} && make install -j ${NUM_CPUS}

cd ${TMP_DIR} && rm -r gromacs-${GMX_VERSION}


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING GROMACS ${GMX_VERSION} TO PATH... "

echo -e "================================================== "

echo -e "\n### ADD GROMACS ${GMX_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${GMX_DIR}/bin:\${PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P gmx) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "================================================== "

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "================================================== "

fi
