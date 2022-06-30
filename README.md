# DArray: Distributed Memory Multi-dimensional Array

This project aims to provide a scalable universal dense and sparse
multi-dimensional array (vector, matrix, tensor) and
their operations (e.g. matrix/tensor decomposition)
as C++ headers. The dependencies are only a recent C++17
compiler, MPI, BLAS,
and LAPACK. [OpenBLAS] can satisfy the latter two
dependencies. 


The DArray is distributed along multiple MPI processes. 
The data distribution is n-d cyclic (e.g. on matrix, it's
usually 2d-cyclic similar to [Elemental]). 

## Current functionalities: 

- 2d cyclic DMatrix for distributed dense matrix
- General Linear algebra on matrix: 
    + BLAS3 Level: 
      - Triangular solve
      - Matrix-matrix multiplication
    + Cholesky factorization
      - Right-looking blocking
      - Recursive  
    + LU factorization
      - With partial pivoting
      - [WIP] Unpivoted
    + QR factorization (Householder)
      - [WIP] Tall Skinny
      - [WIP] Communication Avoiding QR with TSQR as panel
    + Eigenvalue Decomposition
      - [WIP] Cuppen's Divide and Conquer
      - [WIP] Two stage tridiagonalization 
      - [WIP] Polar decomposition based Eigenvalue decomposition
        + QHWD
        + Scaled Newton
- [WIP] Linear System/Least Square
  + Hybrid direct/iterative solver
- [WIP] Randomized Linear Algebra
    + Randomized Projection SVD/Low Rank Approximation
    + Randomized Block Lanczos for Eigen/Singular Value Decomposition
    
- [WIP] Tensor Decomposition
    + Tucker Decomposition for sparse tensor

## Features
+ C++ header only library: just include in your project
+ State of the art performance and scalability
+ Support for many recent algorithms (randomized, communication avoiding, matrix/tensor decomposition)
  and applications (optimization, machine learning)
+ Extensibility: the infrastructure provides critical data structure
    and functionality which can be easily plugged in applications. 
  
# Publications


# Related Projects

[Elemental] is a major inspiration for this project. We use the
same data distribution scheme (2d cyclic). Compare to Elemental,
this project can be only included 

[ScaLAPACK] is a dependable, mature, and fast/scalable numerical
linear algebra package. We use different data distribution scheme
(cyclic vs. block cyclic). 

[Dask] is a python library collection that "clusterize" common numerical
Python packages, including NumPy, pandas, and scikit-learn, to
cluster. It includes some functionality of matrix computations. 
The parallelism mechanism is DAG (directed acyclic graph) scheduling
of tasks. 

[LAPACK] is the most comprehensive, high quality matrix computation
routines that are performant, accurate, and reliable. It's a collection
of Fortran subroutines. 


[Elemental]: https://github.com/elemental/Elemental
[ScaLAPACK]: https://www.netlib.org/scalapack/
[Dask]: https://dask.org/
[LAPACK]: https://www.netlib.org/lapack/
[OpenBLAS]: https://github.com/xianyi/OpenBLAS

# Documentation

## Tests


## Performance Tests