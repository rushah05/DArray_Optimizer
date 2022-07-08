#include "darray.h"
#include "darray_convex.h"
#include "cblas_like.h"
#include "read_input.h"
#include <random>

extern void cuda_init();
extern void cuda_finalize();

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q); // np = p*q
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if(g.rank()==0) printf("#### test grid pxq %dx%d #### \n", p, q);
    g.barrier();

    if(g.rank()==0) cuda_init();
    long long int gn=16;
    int d=4;
    int k=4;
    int qq=1;
    float gamma=0.0001f;

    DArray::DMatrix<float> X(g, d, gn);
    X.set_function([](int i, int j) -> float {
        return i+j*8; //ldx is 8
    });
    X.collect_and_print("X");
    // Y.set_function([](int i, int j) -> float {
    //     return (rand() > RAND_MAX/2) ? 1 : -1;
    // });
    // Y.collect_and_print("Y");
    g.barrier();


    printf("rank::%d, X[%d,%d]\n", g.rank(), X.dims()[0], X.dims()[1]);
    DArray::LMatrix<float> Xi = X.transpose_and_replicate_in_rows(0, 0, X.dims()[0], X.dims()[1]);
    DArray::LMatrix<float> Xj = X.replicate_in_columns(0, 0, X.dims()[0], X.dims()[1]);
    DArray::LMatrix<float> Xij_norm (Xi.dims()[0], Xj.dims()[1]);
    if(X.grid().rank() == 0){
        Xi.print("Xi");
        Xj.print("Xj");
    }
    g.barrier();

    

    // DArray::multiply_matrix<float>("N", "N", 2.0, 0.0, Xi, Xj, Xij);
    // if(X.grid().rank() == 0){
    //     Xij.print("Xij");
    // }

    if(g.rank()==0) cuda_finalize();
    MPI_Finalize();
}