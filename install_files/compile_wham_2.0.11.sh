#!/bin/bash


# '

# COMPILE WHAM 2.0.11

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash compile_wham_2.0.11.sh

# '


DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

WHAM_VERSION="2.0.11"

WHAM_DIR=${INSTALL_DIR}/wham_${WHAM_VERSION}

mkdir -p ${WHAM_DIR}


### COMPILE CMAKE

cd ${TMP_DIR} && wget http://membrane.urmc.rochester.edu/sites/default/files/wham/wham-release-${WHAM_VERSION}.tgz

tar -xvf wham-release-${WHAM_VERSION}.tgz && rm -r wham-release-${WHAM_VERSION}.tgz

mv wham ${WHAM_DIR}

cd ${WHAM_DIR} && cd wham

cd wham

sed -i s/'#define k_B 0.001982923700'/'\/\/\#define k_B 0.001982923700'/ wham.h

sed -i s/'\/\/#define k_B 0.0083144621'/'#define k_B 0.0083144621'/ wham.h

make clean && make

cd ../

cd wham-2d

sed -i s/'#define k_B 0.001982923700'/'\/\/\#define k_B 0.001982923700'/ wham-2d.h

sed -i s/'\/\/#define k_B 0.0083144621'/'#define k_B 0.0083144621'/ wham-2d.h

make clean && make


### EXPORT

echo -e "\n=================================================="

echo -e "=== ADDING WHAM ${WHAM_VERSION} TO PATH... "

echo -e "================================================== "

echo -e "\n### ADD WHAM ${WHAM_VERSION} TO PATH" >> ~/.bashrc

echo -e "PATH=${WHAM_DIR}/wham/wham:\${PATH}" >> ~/.bashrc

echo -e "PATH=${WHAM_DIR}/wham/wham-2d:\${PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P wham) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "================================================== "

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "================================================== "

fi
