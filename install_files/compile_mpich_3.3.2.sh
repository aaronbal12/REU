#!/bin/bash


# '

# COMPILE MPICH 3.3.2

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_mpich_3.3.2.sh

# '


NUM_CPUS=$(lscpu | grep 'CPU(s):' | awk '{print $NF}')

DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

MPICH_VERSION="3.3.2"

MPICH_DIR=${INSTALL_DIR}/mpich_${MPICH_VERSION}


### COMPILE FROM SOURCE

cd ${TMP_DIR} && wget http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz

tar -xzvf mpich-${MPICH_VERSION}.tar.gz && rm -r mpich-${MPICH_VERSION}.tar.gz

cd mpich-${MPICH_VERSION}

./configure --prefix=${MPICH_DIR} 2>&1 | tee c.txt

make -j ${NUM_CPUS} 2>&1 | tee m.txt && make install -j ${NUM_CPUS} 2>&1 | tee mi.txt

cd ${TMP_DIR} && rm -r mpich-${MPICH_VERSION}


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING MPICH ${MPICH_VERSION} TO PATH..."

echo -e "=================================================="

echo -e "\n### ADD MPICH ${MPICH_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${MPICH_DIR}/bin:\${PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P mpichversion) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "=================================================="

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "=================================================="

fi
