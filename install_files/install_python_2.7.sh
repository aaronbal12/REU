#!/bin/bash


# '

# INSTALL PYTHON 2.7

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash install_python_2.7.sh

# '


DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

PY2_MAJOR="2" && PY2_MINOR="7"

PY2_WHEELS_DIR=${INSTALL_DIR}/wheels_pip_${PY2_MAJOR}.${PY2_MINOR}


### COMPILE FROM SOURCE

apt install -y python${PY2_MAJOR}.${PY2_MINOR} python${PY2_MAJOR}.${PY2_MINOR}-dev

wget https://bootstrap.pypa.io/pip/${PY2_MAJOR}.${PY2_MINOR}/get-pip.py

python${PY2_MAJOR}.${PY2_MINOR} get-pip.py

rm get-pip.py

pip${PY2_MAJOR}.${PY2_MINOR} wheel \

--no-cache-dir \

--wheel-dir ${PY2_WHEELS_DIR} \

numpy==1.16.6 \

networkx==1.11

pip${PY2_MAJOR}.${PY2_MINOR} install \

--no-cache \

${PY2_WHEELS_DIR}/*


### EXPORT

echo -e "\n=================================================="

echo -e "=== PYTHON ${PY2_MAJOR}.${PY2_MINOR} SHOULD BE IN BIN..."

echo -e "=================================================="

if [ -f $(type -P python2.7) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "=================================================="

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "=================================================="

fi
