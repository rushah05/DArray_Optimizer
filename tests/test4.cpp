#include "darray.h"
#include "lapack_like.h"
#include "cblas_like.h"

#include <iostream>
// test LU with partial pivoting correctness.
using namespace std;
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if (g.rank()==0) printf("#### LU test grid pxq %dx%d #### \n", p, q);
    g.barrier();

    int n = 10;
    int bs = 4;
    DArray::ArgsParser parser;
    parser.add_option(n, "matrix size n*n", "size-n", 'n', "n");
    parser.add_option(bs, "block size bs", "block-size", 'b', "bs");
    parser.parse(argc, argv);

    DArray::DMatrix<float> A(g, n, n);
    A.set_normal(0, 1);

    auto A0 = A.clone();
    auto A00 = A.clone();

    g.barrier();

//    auto LU = A.collect(0);

    g.barrier();

    DArray::Tracer::trace_on();
    {
        DArray::Tracer tracer("lu factorize",true);
        DArray::lu_factorize_partial_pivot(A, 0, 0, n, n, bs);
    }
    DArray::Tracer::trace_off();

    g.barrier();
    auto lA = A.collect(0);

    {
        DArray::Tracer tracer("trtrm",true);
        DArray::triangular_triangular_multiply(A, A, A0, 1.0f, 0.0f);
    }
    {
        DArray::Tracer tracer("permute P*(LU)",true);
        A0.permute_rows_inverse_ipiv(0, 0, n, n, A.ipiv().data(), A.ipiv().size());
    }
    auto fnA = A.fnorm();
    A00.substract(A0);
    auto fn = A00.fnorm();
    if(g.rank()==0) fmt::print("fnorm(A-P*L*U)/fnorm(A)={}\n", fn / fnA );
    MPI_Finalize();
}