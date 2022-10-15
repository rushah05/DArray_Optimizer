#!/bin/bash
#SBATCH -J barr_N2n2
#SBATCH -o svm_barr_N2n2.o%j
#SBATCH --ntasks-per-node=2 -N 2
#SBATCH -t 01:30:0
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
# module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"

time mpirun ./testbarrier "/project/pwu/dataset/ijcnn1" 10000 22 32 0.045 1 100

