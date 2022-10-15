#!/bin/bash
  
#SBATCH -N 4
#SBATCH -n 64
#SBATCH --distribution=block:block


module load GCC intel
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 mpirun ./perftest_cholesky -n 50000 
