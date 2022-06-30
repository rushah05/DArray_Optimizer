#include "darray.h"
#include "blas_like.h"
#include "lapack_like.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q); // np = p*q
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if (g.rank()==0) printf("#### tsqr test grid pxq %dx%d #### \n", p, q);
    g.barrier();

    DArray::DMatrix<float> A(g, 10, 3);
    A.set_normal(0,1);

    A.collect_and_print("A");
    auto R = tall_skinny_qr_factorize(A, 0, 0, 10, 3);
    if (A.grid().rank() == 0) R.print("R");
    A.collect_and_print("Q");

    MPI_Finalize();
}
