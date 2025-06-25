#!/bin/bash

mkdir equil
mkdir equil/output_files_1

python3 ../indus_slab_position_GUA_1.py -f nvt_equil -p equil/output_files_1/ -s "asym_density_peak" -d 1,2.79,0.3 -N -1 -C 1,2.335
