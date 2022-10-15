#!/bin/bash 
#SBATCH -J svm 
#SBATCH -o svm_job.o%j 
#SBATCH -t 00:05:00 
#SBATCH -N 1 -n 4
#SBATCH --gres=gpu:4
#SBATCH --mem=150000

module load GCC intel
module load cmake
module load cudatoolkit/11.0
nvidia-smi

time mpirun ./testsvm
