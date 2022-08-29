#include "darray.h"
#include "blas_like.h"

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q); // np = p*q
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    g.barrier();

    int n=32;
    int d=8;
    DArray::DMatrix<float> X(g, d, n);
    X.set_function([](int gi, int gj) ->float {
        return gi+gj/10.0;
    });

    // X.collect_and_print("X");

    float sone=1.0f;
    float szero=0.0f;
    int bs=16;

    for(int i=0, j=0; i<n && j<n; i+=bs, j+=bs) {
        int ib = std::min(bs, n-i);
        int jb = std::min(bs, n-j);

        auto Xi=X.replicate_in_all(0, i, d, i+ib).transpose();
        auto Xj=X.replicate_in_all(0, j, d, j+jb).transpose();
        DArray::LMatrix<float> K(ib, jb);
        if(g.rank()==0) printf("Xi[%d,%d] Xj[%d,%d] K[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], K.dims()[0], K.dims()[1]);
        sgemm_("N", "T", &ib, &jb, &d, &sone, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &szero, K.data(), &K.ld());
        
        if(g.rank()==0) Xi.print("Xi");
        if(g.rank()==0) Xj.print("Xj");
        if(g.rank()==0) K.print("K");
        return 0;
    }


    // X.collect_and_print("X");
    // auto Xi = X.transpose_and_replicate_in_rows(0, 0, d, n);
    // auto Xj = X.replicate_in_columns(0, 0, d, n);
    // if(g.rank()==0) Xi.print("Xi");
    // if(g.rank()==0) Xj.print("Xj");
    // // DArray::LMatrix<float> Kl(Xi.dims()[0], Xj.dims()[1]);
    // DArray::DMatrix<float> K(g, n, n);
    // auto Kl=K.replicate_in_all(0, 0, n, n);
    // float sone=1.0f, szero=0.0f;
    // if(g.rank()==0) printf("Xi[%d,%d] Xj[%d,%d] Kl[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Kl.dims()[0], Kl.dims()[1]);
    // sgemm_("N", "N", &Xi.dims()[0], &Xj.dims()[1], &Xi.dims()[1], &sone, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &szero, Kl.data(), &Kl.ld());
    // K.dereplicate_in_all(Kl, 0, 0, n, n);

    // K.collect_and_print("K");

    // DArray::DMatrix<float> K(g, n, n);
    // K.set_constant(0.0);
    // {
    //     auto Xi = X.transpose_and_replicate_in_rows(0, 0, d, n);
    //     auto Xj = X.replicate_in_columns(0, 0, d, n);
    //     // auto Xj=Xc.transpose();
    //     int lm=Xi.dims()[0];
    //     int ln=Xj.dims()[1];
    //     int ld=Xi.dims()[1];
    //     DArray::LMatrix<float> Kl(lm, ln);
    //     float sone=1.0f, szero=0.0f;
    //     sgemm_("N", "N", &lm, &ln, &ld, &sone, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &szero, Kl.data(), &Kl.ld());
    //     K.dereplicate_in_columns(Kl, 0, 0, lm, ln);
    //     K.dereplicate_in_rows(Kl, 0, 0, lm, ln);
    //     if(g.rank()==0) printf("Xi[%d,%d] Xj[%d,%d] Kl[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Kl.dims()[0], Kl.dims()[1]);
    // }
    // K.collect_and_print("K");

    // DArray::DMatrix<float> KK(g, n, n);
    // KK.set_constant(0.0);
    // {
    //     auto Xc=X.replicate_in_all(0, 0, d, n);
    //     auto Xi=Xc.transpose();
    //     auto Xj=Xc;
    //     auto Kl=KK.replicate_in_all(0, 0, n, n);
    //     float sone=1.0f, szero=0.0f;
    //     sgemm_("N", "N", &n, &n, &d, &sone, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &szero, Kl.data(), &Kl.ld());
    //     KK.dereplicate_in_all(Kl, 0, 0, n, n);
    //     if(g.rank()==0) printf("Xi[%d,%d] Xj[%d,%d] Kl[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Kl.dims()[0], Kl.dims()[1]);
    // }
    // KK.collect_and_print("K");


    
    MPI_Finalize();
}