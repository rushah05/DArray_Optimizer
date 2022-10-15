#include "darray.h"
#include "blas_like.h"

extern void cuda_init();
extern void cuda_finalize();
extern void tc_gemm(int rank, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    cuda_init();
    if (g.rank()==0) printf("#### test grid pxq %dx%d #### \n", p, q);
    g.barrier();
    DArray::ElapsedTimer timer;

    int n = atol(argv[1]);
    int k = atoi(argv[2]);
    int bs = atol(argv[3]);
    if (g.rank()==0) printf("#### Starting GEMM n=%d, k=%d, bs=%d  #### \n", n, k, bs);
    DArray::DMatrix<float> K(g, bs, bs);
    K.set_function([](int gi, int gj) ->float {
        return gi+gj+2/10.0;
    });
    DArray::DMatrixDescriptor<float> Kdesc{K, 0, 0, bs, bs};

    DArray::DMatrix<float> O(g, n, k);
    O.set_function([](int gi, int gj) ->float {
        return gi+gj+2/10.0;
    });
    DArray::DMatrixDescriptor<float> Odesc{O, 0, 0, n, k};

    DArray::DMatrix<float> A(g, n, k);
    A.set_constant(0.0);
    DArray::DMatrixDescriptor<float> Adesc{A, 0, 0, n, k};

    timer.start();
    auto Ki=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j2);
    auto Oj=Odesc.matrix.replicate_in_columns(Odesc.i1, Odesc.j1, bs, Odesc.j2);
    auto Aij=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, bs, Adesc.j2);
    // // if(g.rank() == 0) printf("Ki[%d,%d]%d,  Oj[%d,%d]%d,  Aij[%d,%d]%d \n", Ki.dims()[0], Ki.dims()[1], Ki.ld(),  Oj.dims()[0], Oj.dims()[1], Oj.ld(), Aij.dims()[0], Aij.dims()[1], Aij.ld());
    tc_gemm(g.rank(), Aij.dims()[0], Aij.dims()[1], Ki.dims()[1], Ki.data(), Ki.ld(), Oj.data(), Oj.ld(), Aij.data(), Aij.dims()[0]);
    // float one=1.0f, zero=0.0f;
    // sgemm_("N", "N", &Aij.dims()[0], &Aij.dims()[1], &Ki.dims()[1], &one, Ki.data(), &Ki.ld(), Oj.data(), &Oj.ld(), &zero, Aij.data(), &Aij.ld());
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: 1 block of K*O takes : {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);

    int chunks = (n+bs-1)/bs;
    if(g.rank() == 0) printf("in %d chunks. Progress: %d \n", chunks, chunks*chunks); 

    // timer.start();
    // for(int i=0; i<n; i+=bs){
    //     int ib=std::min(bs, n-i);
    //     auto AA=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i1+ib, Adesc.j2);

    //     for(int j=0; j<n; j+=bs){
    //         int jb=std::min(bs, n-j);
    //         //  if(g.rank()== 0) printf("[%d,%d] i+ib=%d j+jb=%d bs=%d\n", i, j, i+ib, j+jb, bs);
    //         auto Ki=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //         auto Oj=Odesc.matrix.replicate_in_columns(Odesc.i1, Odesc.j1, Odesc.i1+jb, Odesc.j2);
    //         DArray::LMatrix<float> Aij(Ki.dims()[0], Oj.dims()[1]);
    //         tc_gemm(g.rank(), Ki.dims()[0], Oj.dims()[1], Ki.dims()[1], Ki.data(), Ki.dims()[0], Oj.data(), Oj.dims()[0], Aij.data(), Aij.dims()[0]);
    //         DArray::copy_block(Aij.dims()[0], Aij.dims()[1], Aij.data(), Aij.ld(), AA.data(), AA.ld());
    //     }
    // }

    // ms = timer.elapsed();
    // if(g.rank()==0) fmt::print("P[{},{}]: Out-Of-Core K*O takes : {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);

    cuda_finalize();
    MPI_Finalize();
}

















// for(int kk=0; kk<n; kk+=bs){
//         int ke=std::min(bs, n-kk);
//         // printf("[0, %d], ke=%d\n", kk, ke);
//         auto Ki=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i2, Kdesc.j1+ke);
//         auto Oj=Odesc.matrix.replicate_in_columns(Odesc.i1+kk, Odesc.j1, Odesc.i1+kk+ke, Odesc.j2);
//         // DArray::LMatrix<float> Aij(Ki.dims()[0], Oj.dims()[1]);
//         // tc_gemm(g.rank(), Ki.dims()[0], Oj.dims()[1], Ki.dims()[1], Ki.data(), Ki.dims()[0], Oj.data(), Oj.dims()[0], Aij.data(), Aij.dims()[0]);

//         // auto AA=Adesc.matrix.local_view(Adesc.i1, Adesc.j1, Adesc.i2, Adesc.j2);
//         // DArray::copy_block(Aij.dims()[0], Aij.dims()[1], Aij.data(), Aij.ld(), AA.data(), AA.ld());
//     }