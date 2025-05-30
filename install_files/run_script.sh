#!/bin/bash


### DEFINE SOURCE PATH

SOURCE_PATH=$(pwd)


### PREPARE BASIC ENVIRONMENT

# SCRIPTS_NAMES_LIST=(setup install_cuda_12.4 install_python_2.7 install_python_3.10 compile_mpich_3.3.2 compile_cmake_3.18.4 compile_wham_2.0.11)

SCRIPTS_NAMES_LIST=($@)

for SCRIPT_NAME in ${SCRIPTS_NAMES_LIST[@]}; do

echo -e "=========================================================="

echo -e "=== RUNNING ${SCRIPT_NAME}.sh "

echo -e "=== Pausing for 5 seconds... "

echo -e "=========================================================="

sleep 5

sudo -E env "PATH=${PATH}" "LIBRARY_PATH=${LIBRARY_PATH}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" bash "${SOURCE_PATH}"/${SCRIPT_NAME}.sh

done


echo -e "====================================================================="

echo -e "=== REMEMBER TO SOURCE YOUR BASHRC FOR CHANGES TO TAKE EFFECT: "

echo -e "=== source ~/.bashrc "
