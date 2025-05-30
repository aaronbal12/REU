#!/bin/bash


# '

# COMPILE CMAKE 3.18.4

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_cmake_3.18.4.sh

# '


NUM_CPUS=$(lscpu | grep 'CPU(s):' | awk '{print $NF}')

DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

CMAKE_VERSION="3.18.4"

CMAKE_DIR=${INSTALL_DIR}/cmake_${CMAKE_VERSION}


### COMPILE CMAKE

cd ${TMP_DIR} && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz

tar -xzvf cmake-${CMAKE_VERSION}.tar.gz && rm -r cmake-${CMAKE_VERSION}.tar.gz

cd cmake-${CMAKE_VERSION}

./bootstrap --prefix=${CMAKE_DIR} -- -DCMAKE_USE_OPENSSL=OFF

make -j ${NUM_CPUS} && make install -j ${NUM_CPUS}

cd ${TMP_DIR} && rm -r cmake-${CMAKE_VERSION}


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING CMAKE ${CMAKE_VERSION} TO PATH..."

echo -e "=================================================="

echo -e "\n### ADD CMAKE ${CMAKE_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${CMAKE_DIR}/bin:\${PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P cmake) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "=================================================="

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "=================================================="

fi
