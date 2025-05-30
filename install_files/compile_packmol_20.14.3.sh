#!/bin/bash


# '

# COMPILE PACKMOL 20.14.3

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_packmol_20.14.3.sh

# '


NUM_CPUS=$(lscpu | grep 'CPU(s):' | awk '{print $NF}')

DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

PACKMOL_VERSION="20.14.3"

PACKMOL_DIR=${INSTALL_DIR}/packmol_${PACKMOL_VERSION}


### COMPILE FROM SOURCE

cd ${TMP_DIR} && wget https://github.com/m3g/packmol/archive/refs/tags/v${PACKMOL_VERSION}.tar.gz

tar -xzvf v${PACKMOL_VERSION}.tar.gz && rm -r v${PACKMOL_VERSION}.tar.gz

mv packmol-${PACKMOL_VERSION} ${PACKMOL_DIR}

cd ${PACKMOL_DIR}

make -j ${NUM_CPUS}


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING PACKMOL ${PACKMOL_VERSION} TO PATH... "

echo -e "================================================== "

echo -e "\n### ADD PACKMOL ${PACKMOL_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${PACKMOL_DIR}:\${PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P packmol) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "================================================== "

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "================================================== "

fi
