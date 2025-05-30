#!/bin/bash


# '

# INSTALL PYTHON 3.10

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash install_python_3.10.sh

# '


DEBIAN_FRONTEND=noninteractive

TMP_DIR=/tmp

USR_LOCAL_DIR=/usr/local

INSTALL_DIR=${USR_LOCAL_DIR}


### DEFINE ARGUMENTS

PY3_MAJOR="3" && PY3_MINOR="10"

PY3_WHEELS_DIR=${INSTALL_DIR}/wheels_pip_${PY3_MAJOR}.${PY3_MINOR}


### COMPILE FROM SOURCE

apt install -y python${PY3_MAJOR}.${PY3_MINOR} python${PY3_MAJOR}.${PY3_MINOR}-dev python${PY3_MAJOR}.${PY3_MINOR}-distutils python${PY3_MAJOR}-testresources && \

wget https://bootstrap.pypa.io/pip/get-pip.py

python${PY3_MAJOR}.${PY3_MINOR} get-pip.py

rm get-pip.py

pip${PY3_MAJOR}.${PY3_MINOR} wheel \

--no-cache-dir \

--wheel-dir ${PY3_WHEELS_DIR} \

numpy==1.26.4 \

matplotlib==3.8.4 \

mdtraj==1.9.9 \

scikit-learn==1.4.2 \

rdkit==2024.3.5 \

scipy==1.13.0 \

openbabel-wheel==3.1.1.19 \

acpype==2022.7.21

pip${PY3_MAJOR}.${PY3_MINOR} install \

--no-cache \

${PY3_WHEELS_DIR}/*


### EXPORT

echo -e "\n=================================================="

echo -e "=== PYTHON ${PY3_MAJOR}.${PY3_MINOR} SHOULD BE IN BIN..."

echo -e "=================================================="

if [ -f $(type -P python3.10) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "=================================================="

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "=================================================="

fi
