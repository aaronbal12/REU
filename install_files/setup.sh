#!/bin/bash


# '

# SETUP

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash setup.sh

# '


DEBIAN_FRONTEND=noninteractive


apt update -y

apt upgrade -y

echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections

apt install -y --no-install-recommends apt-utils build-essential pkg-config software-properties-common

add-apt-repository -y ppa:deadsnakes/ppa

apt install -y --no-install-recommends gfortran libarpack2-dev libparpack2-dev libarpack++2-dev libatlas3-base libhdf4-0 libhdf5-dev

apt install -y --no-install-recommends gcc g++ nano vim wget curl bc autoconf libtool openmpi-bin openmpi-doc libopenmpi-dev

apt install -y --no-install-recommends gawk dos2unix git openbabel unzip ttf-mscorefonts-installer grace gnuplot
