#!/bin/bash


# '

# COMPILE PLUMED 2.8.0

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_plumed_2.8.0.sh

# '


NUM_CPUS=$(lscpu | grep 'CPU(s):' | awk '{print $NF}')

DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

C_COMPILER="mpicc"

CXX_COMPILER="mpic++"

PLUMED_VERSION="2.8.0"

PLUMED_DIR=${INSTALL_DIR}/plumed_${PLUMED_VERSION}


### COMPILE FROM SOURCE

if [[ "$PLUMED_VERSION" == *"master"* ]]; then

cd ${TMP_DIR} && git clone https://github.com/plumed/plumed2.git

PLUMED_SRC_DIR=plumed2

else

cd ${TMP_DIR} && wget https://github.com/plumed/plumed2/releases/download/v${PLUMED_VERSION}/plumed-${PLUMED_VERSION}.tgz

tar -xzvf plumed-${PLUMED_VERSION}.tgz && rm -r plumed-${PLUMED_VERSION}.tgz

PLUMED_SRC_DIR=plumed-${PLUMED_VERSION}

fi

cd ${PLUMED_SRC_DIR}

./configure --prefix=${PLUMED_DIR} --enable-modules=all CC=${C_COMPILER} CXX=${CXX_COMPILER}

make -j ${NUM_CPUS} && make install -j ${NUM_CPUS}

cd ${TMP_DIR} && rm -r ${PLUMED_SRC_DIR}


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING PLUMED ${PLUMED_VERSION} TO PATH... "

echo -e "================================================== "

echo -e "\n### ADD PLUMED ${PLUMED_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${PLUMED_DIR}/bin:\${PATH}" >> ~/.bashrc

echo -e "PATH=${PLUMED_DIR}/include:\${PATH}" >> ~/.bashrc

echo -e "PATH=${PLUMED_DIR}/lib:\${PATH}" >> ~/.bashrc

echo -e "PATH=${PLUMED_DIR}/lib/pkgconfig:\${PATH}" >> ~/.bashrc

echo -e "LIBRARY_PATH=${PLUMED_DIR}/lib:\${LIBRARY_PATH}" >> ~/.bashrc

echo -e "LD_LIBRARY_PATH=${PLUMED_DIR}/lib:\${LD_LIBRARY_PATH}" >> ~/.bashrc

echo -e "PLUMED_KERNEL=${PLUMED_DIR}/lib/libplumedKernel.so" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

echo -e "export LIBRARY_PATH" >> ~/.bashrc

echo -e "export LD_LIBRARY_PATH" >> ~/.bashrc

echo -e "export PLUMED_KERNEL" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P plumed) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "================================================== "

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "================================================== "

fi
