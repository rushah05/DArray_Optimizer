#include "darray.h"
//#include "lapack_like.h"
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
    g.barrier();

    DArray::DMatrix<float> m(g, 9,9);
    m.set_function([](int i, int j) -> float {
        if (i >= j) return (1 + i + j);
        else return 0;
    });
    m.grid().barrier();

    auto A = m.collect(0);
    if(m.grid().rank()==0) {
        A.print("matrix A");
        DArray::LMatrix<float> C(A.dims()[0],A.dims()[1]);
        C.set_constant(0);
        DArray::multiply_matrix<float>("N", "T", 1.0, 1.0, A, A, C);
        C.print("C=A*A");
    }


    MPI_Finalize();
}