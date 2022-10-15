#!/bin/bash
#SBATCH -J test7
#SBATCH -o test7_N4n4_TC.o%j
#SBATCH --ntasks-per-node=4 -N 4
#SBATCH -t 01:30:0
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
time mpirun ./test7 500000 500 10000

