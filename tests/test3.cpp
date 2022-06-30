#include "darray.h"
#include "lapack_like.h"
#include "cblas_like.h"

#include <iostream>

using namespace std;
int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);

    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if (g.rank()==0) printf("#### grid pxq %dx%d #### \n", p, q);
    if(p!=q) {
        fmt::print("p({}) must be equal to q({})\n",p, q);
    }
    g.barrier();

    int n = 10;
    DArray::DMatrix<float> A(g, n, n);
    DArray::LMatrix<float> m( n,n);
    m.set_function([](int i, int j) -> float {
        if (i >= j) return (1 + i + j);
        else return 0;
    });
    A.grid().barrier();


    if(A.grid().rank()==0) {
        m.print("matrix m");
    }
    DArray::LMatrix<float> C(A.dims()[0],A.dims()[1]);
    C.set_constant(0);
    DArray::multiply_matrix<float>("N", "T", 1.0, 1.0, m, m, C);
//    if(A.grid().rank()==0)
//        C.print("C=A*A");

    A.dereplicate_in_all(C, 0, 0, n, n);

    A.collect_and_print("matrix A=m*m'", 0);

    int i = 0, nn = 10;
//    DArray::cholesky_factorize(A, 'U', 10, 0);
    DArray::cholesky_factorize_recursive(A, 'U', nn, i, 4);

    A.collect_and_print(fmt::format("matrix A after cholesky of submatrix {}:{}x{}:{}",i, i,i+nn, i+nn ),0);

    auto mA = A.collect(0);

    MPI_Finalize();
}