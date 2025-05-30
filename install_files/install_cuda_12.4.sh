#!/bin/bash


# '

# INSTALL CUDA 12.4

# Run the following command:

# sudo -E env "PATH=${PATH}" \

# "LIBRARY_PATH=${LIBRARY_PATH}" \

# "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \

# bash install_cuda_12.4.sh

# '


DEBIAN_FRONTEND=noninteractive


### INSTALL CUDA TOOLKIT AND DRIVERS

### CUDA DRIVERS WILL BE MANAGED BY GEFORCE EXPERIENCE

### DOWNLOAD: https://www.nvidia.com/en-us/geforce/geforce-experience/

### SET UP YOUR ACCOUNT AND UPDATE DRIVERS


### THESE ARE THE NON-SUDO COMMANDS (SUDO BASH THE SCRIPT)

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin

mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2004-12-4-local_12.4.1-550.54.15-1_amd64.deb

dpkg -i cuda-repo-ubuntu2004-12-4-local_12.4.1-550.54.15-1_amd64.deb

cp /var/cuda-repo-ubuntu2004-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/

apt-get update

apt-get -y install cuda-toolkit-12-4

apt-get install -y cuda-drivers

rm -rv cuda-repo-ubuntu2004-12-4-local_12.4.1-550.54.15-1_amd64.deb


### POST-SETUP FOR CUDA TOOLKIT INSTALLATION

echo -e "\n=================================================="

echo -e "=== ADDING CUDA TO PATH..."

echo -e "=================================================="

echo -e "### ADD CUDA TO PATH" >> ~/.bashrc

echo -e "PATH=/usr/local/cuda/bin:\${PATH}" >> ~/.bashrc

echo -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >> ~/.bashrc

echo -e "export PATH" >> ~/.bashrc

echo -e "export LD_LIBRARY_PATH" >> ~/.bashrc

source ~/.bashrc

if [ -f $(type -P nvcc) ]; then

echo -e "\n=================================================="

echo -e "=== SUCCESSFUL INSTALLATION!!! "

echo -e "=================================================="

else

echo -e "\n=================================================="

echo -e "=== INSTALLATION FAILED!!! "

echo -e "=================================================="

fi
