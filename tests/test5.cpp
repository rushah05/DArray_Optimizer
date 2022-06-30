#include "darray.h"
#include "blas_like.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if (g.rank()==0) printf("#### transpose test grid pxq %dx%d #### \n", p, q);
    g.barrier();

    DArray::DMatrix<float> A(g, 10, 10);
    A.set_function([](int gi, int gj) ->float {
        return gi+gj/10.0;
    });

    auto At = A.transpose_and_replicate_in_rows(2,4,9,5);
//    A.collect_and_print("A");

    if(g.ranks()[1] == 0){
        g.for_each_process_in_column([&]() {
            fmt::print("P[{},{}]", g.ranks()[0], g.ranks()[1]);
            At.print("At");
        });
    }
//    At.print("At");

    A.collect_and_print("A");
    DArray::DMatrixDescriptor<float> A1{A, 5,0,10,5}, A2{A, 0,5,5, 10}, A3{A, 5, 5, 10, 10};
    DArray::matrix_multiply("T", "N", A2, A2, A3, -1.0f, 1.0f, 2);
    A.collect_and_print("A after multiplication");

//    A.set_normal(0, 1);
    A.set_function([](int i, int j){
        return 1+i+j;
    });
    A.collect_and_print("A before triangular solve");
    DArray::DMatrix<float> B(g, 10, 10);
    B.set_function([](int gi, int gj) ->float {
        return gi+gj/10.0;
    });
    DArray::DMatrixDescriptor<float> AA1{A, 0, 0, 5, 5}, B2{B, 0, 5, 5, 10};
    B.collect_and_print("B before triangular solve");
    DArray::triangular_solve("L", "U", "T", "N", 5, 5, 1.0f, A1, B2, 3);

    B.collect_and_print("B after triangular solve");


    MPI_Finalize();
}
