#pragma once
#include "darray.h"
#include "blas_like.h"
#define NN 50000 // block size for out of core processing, 4096*12 due to the dimensions of K(NN, NN)

extern void rbf_kernel(int rank, int m, int n, int d, float *Xi, int ldxi, float *Xj, int ldxj, float *Yi, float *Yj, float *K, int ldk, float gamma);
extern void tc_gemm(int rank, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0);
extern void tc_sgemm(int rank, char transA, char transB, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0);

namespace DArray{

    template<typename T>
    void LRA(Grid g, int n, int d, int k, DMatrix<T> X, DMatrix<T>& Y, DMatrix<T>& O, DMatrix<T>& A, float gamma){
        int chunks = (n+NN-1)/NN;
		if(g.rank() == 0) printf("in %d chunks. Progress: %d \n", chunks, chunks*chunks); 
        DMatrix<T> K(g, NN, NN);
        K.set_normal(0, 1);
        T alpha=1, beta=0;

        for(int i=0; i<n; i+=NN){
            int rn=std::min(NN, n-i);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+rn);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+rn);
            auto Al=A.local_view(i, 0, i+rn, k);

            for(int j=0; j<n; j+=NN){
                int cn=std::min(NN, n-j);
                // if(g.rank() == 0) printf("[%d, %d], rn=%d, cn=%d \n", i, j, rn, cn);

                auto Xj=X.replicate_in_columns(0, j, d, j+cn).transpose();
                auto Yj=Y.replicate_in_columns(0, j, 1, j+cn).transpose();         
                auto Kl=K.local_view(0, 0, rn, cn);
                rbf_kernel(g.rank(), Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), Kl.data(), Kl.ld(), gamma);  
               
                auto Ki=K.replicate_in_rows(0, 0, rn, cn);
                // if(g.rank()==0) Ki.save_to_file("Ki0"+std::to_string(i)+std::to_string(j)+".csv");
                // if(g.rank()==1) Ki.save_to_file("Ki1"+std::to_string(i)+std::to_string(j)+".csv");
                // if(g.rank()==2) Ki.save_to_file("Ki2"+std::to_string(i)+std::to_string(j)+".csv");
                // if(g.rank()==3) Ki.save_to_file("Ki3"+std::to_string(i)+std::to_string(j)+".csv");
                auto Oj=O.replicate_in_columns(j, 0, j+cn, k);
                if(j>1) beta=1;
                tc_gemm(g.rank(), Ki.dims()[0], Oj.dims()[1], Ki.dims()[1], Ki.data(), Ki.ld(), Oj.data(), Oj.ld(), Al.data(), Al.ld(), alpha, beta);
                return;
            }
        }
    }


    template<typename T>
    void Knorm(Grid g, int n, int d, int k, DMatrix<T>& X, DMatrix<T>& Y, DMatrix<T>& Z, float gamma){
        DMatrix<T> K(g, NN, NN);
        double knorm=0.0, zknorm=0.0;

        for(int i=0; i<n; i+=NN){
            int rn=std::min(NN, n-i);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+rn);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+rn);
            auto Zi=Z.replicate_in_rows(i, 0, i+rn, k);

            for(int j=0; j<n; j+=NN){
                int cn=std::min(NN, n-j);
                // if(g.rank() == 0) printf("[%d, %d], rn=%d, cn=%d \n", i, j, rn, cn);
                auto Xj=X.replicate_in_columns(0, j, d, j+cn).transpose();
                auto Yj=Y.replicate_in_columns(0, j, 1, j+cn).transpose();
                auto Kl=K.local_view(0, 0, rn, cn);
                rbf_kernel(g.rank(), Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), Kl.data(), Kl.ld(), gamma);
                knorm+=K.fnorm();
                

                auto Zj=Z.replicate_in_rows(j, 0, j+cn, k).transpose();
                tc_gemm(g.rank(), Zi.dims()[0], Zj.dims()[1], Zi.dims()[1], Zi.data(), Zi.ld(), Zj.data(), Zj.ld(), Kl.data(), Kl.ld(), -1.0f, 1.0f);
                zknorm=K.fnorm();
            }
        }
        if(g.rank() == 0) printf("fnorm(K)=%f fnorm(K-U*U)=%f Residual fnorm=%f\n", knorm, zknorm, zknorm/knorm);
    }
}

























// #pragma once
// #include "darray.h"
// #include "blas_like.h"
// #define NN 50000 // block size for out of core processing, 4096*12 due to the dimensions of K(NN, NN)

// extern void rbf_kernel(int rank, int m, int n, int d, float *Xi, int ldxi, float *Xj, int ldxj, float *Yi, float *Yj, float *K, int ldk, float gamma);
// extern void tc_gemm(int rank, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0);
// extern void tc_sgemm(int rank, char transA, char transB, int m, int k, int n, float *A, int lda, float *BB, int ldb, float *C, int ldc, float alpha=1, float beta=0);

// namespace DArray{

//     template<typename T>
//     void LRA(Grid g, int n, int d, int k, DMatrix<T> X, DMatrix<T>& Y, DMatrix<T>& O, DMatrix<T>& A, float gamma){
//         int chunks = (n+NN-1)/NN;
// 		if(g.rank() == 0) printf("in %d chunks. Progress: %d \n", chunks, chunks*chunks); 
//         DMatrix<T> K(g, NN, NN);
//         T alpha=1, beta=0;

//         for(int i=0; i<n; i+=NN){
//             int rn=std::min(NN, n-i);
//             auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+rn);
//             auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+rn);
//             auto Al=A.local_view(i, 0, i+rn, k);

//             for(int j=0; j<n; j+=NN){
//                 int cn=std::min(NN, n-j);
//                 // if(g.rank() == 0) printf("[%d, %d], rn=%d, cn=%d \n", i, j, rn, cn);

//                 auto Xj=X.replicate_in_columns(0, j, d, j+cn).transpose();
//                 auto Yj=Y.replicate_in_columns(0, j, 1, j+cn).transpose();         
//                 auto Kl=K.local_view(0, 0, rn, cn);
//                 rbf_kernel(g.rank(), Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), Kl.data(), Kl.ld(), gamma);  
               
//                 auto Ki=K.replicate_in_rows(0, 0, rn, cn);
//                 auto Oj=O.replicate_in_columns(j, 0, j+cn, k);
//                 if(j>1) beta=1;
//                 tc_gemm(g.rank(), Ki.dims()[0], Oj.dims()[1], Ki.dims()[1], Ki.data(), Ki.ld(), Oj.data(), Oj.ld(), Al.data(), Al.ld(), alpha, beta);
//             }
//         }
//     }


//     template<typename T>
//     void Knorm(Grid g, int n, int d, int k, DMatrix<T>& X, DMatrix<T>& Y, DMatrix<T>& Z, float gamma){
//         DMatrix<T> K(g, NN, NN);
//         double knorm=0.0, zknorm=0.0;

//         for(int i=0; i<n; i+=NN){
//             int rn=std::min(NN, n-i);
//             auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+rn);
//             auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+rn);
//             auto Zi=Z.replicate_in_rows(i, 0, i+rn, k);

//             for(int j=0; j<n; j+=NN){
//                 int cn=std::min(NN, n-j);
//                 // if(g.rank() == 0) printf("[%d, %d], rn=%d, cn=%d \n", i, j, rn, cn);
//                 auto Xj=X.replicate_in_columns(0, j, d, j+cn).transpose();
//                 auto Yj=Y.replicate_in_columns(0, j, 1, j+cn).transpose();
//                 auto Kl=K.local_view(0, 0, rn, cn);
//                 rbf_kernel(g.rank(), Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Yi.data(), Yj.data(), Kl.data(), Kl.ld(), gamma);
//                 knorm+=K.fnorm();
                

//                 auto Zj=Z.replicate_in_rows(j, 0, j+cn, k).transpose();
//                 tc_gemm(g.rank(), Zi.dims()[0], Zj.dims()[1], Zi.dims()[1], Zi.data(), Zi.ld(), Zj.data(), Zj.ld(), Kl.data(), Kl.ld(), -1.0f, 1.0f);
//                 zknorm=K.fnorm();
//             }
//         }
//         if(g.rank() == 0) printf("fnorm(K)=%f fnorm(K-U*U)=%f Residual fnorm=%f\n", knorm, zknorm, zknorm/knorm);
//     }
// }