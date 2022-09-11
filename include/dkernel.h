#pragma once
#include "darray.h"
#include "blas_like.h"
#include "utils.h"
#include <math.h>

extern void TC_GEMM(int rk, int n, int k, int d, float alpha, float beta, float *fA, int lda, float *fB, int ldb, float *fC, int ldc);
extern void Kernel(int rk, int m, int n, float *Xi, float *Xj, float *Yi, float *Yj, float *Xij, int ldxij, float *K, int ldk, float gamma);

namespace DArray {

    template<typename T>
    void printMatrix(const std::string filepath, int m, int n, LMatrix<T>& A, int lda) {
        FILE *f = fopen(filepath.c_str(), "w");
        assert(f);

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                fprintf(f, "%f, ", A.data()[i+j*lda]);
            }
            fprintf(f, "\n");
        }

        fclose(f);
    }

    template<typename T>
    void rbf(int m, int n, T *Xi, T *Xj, T *Yi, T* Yj, T *Xij, int ldxij, T *K, int ldk, T gamma){
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                K[i+j*ldk] = Yi[i]*Yj[j]* exp(-gamma*(Xi[i]+Xj[j]-2*Xij[i+j*ldxij]));
            }
        }
    }

    template<typename T>
    void vecnorm(T *Z, int ldz, T *Zn, int n, int d){
        T sum=0.0;
        for(int i=0; i<n; ++i){
            sum=0.0;
            for(int j=0; j<d; ++j){
                // printf("i::%d, j::%d, ldz::%d\n", i, j, ldz);
                sum+=(Z[i+j*ldz]*Z[i+j*ldz]);
            }
            Zn[i]=sum;
        }
    }

    template<typename T>
    void LRA(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=16384){
        T alpha = 1, beta = 0, beta2 = 0;
        DMatrix<T> K(g, bs, bs);
        K.set_constant(0.0);
        DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
        DMatrixDescriptor<T> Adesc{A, 0, 0, n, k};
        DMatrixDescriptor<T> Odesc{O, 0, 0, n, k};

        for(int i=0; i<n; i+=bs){
            int ib = std::min(bs, n-i);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
            LMatrix<T> Xisqr(Xi.dims()[0], 1);
            vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
            auto Aik=Adesc.matrix.local_view(Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2);
            
            for(int j=0; j<n; j+=bs){
                int jb = std::min(bs, n-j);
                if(j>0) beta2=1; 
                auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
                auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
                LMatrix<T> Xjsqr(Xj.dims()[0], 1);
                vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

                LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
                auto Xjt=X.replicate_in_columns(0, j, d, j+jb);
                TC_GEMM(rk, Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], alpha, beta, Xi.data(), Xi.ld(), Xjt.data(), Xjt.ld(), Xij.data(), Xij.ld());
                auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);  
                Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0], gamma);   

                auto Kijk=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
                auto Ojk=Odesc.matrix.replicate_in_columns(Odesc.i1+j, Odesc.j1, Odesc.i1+j+jb, Odesc.j2);
                TC_GEMM(rk, Kijk.dims()[0], Ojk.dims()[1], Kijk.dims()[1], alpha, beta2, Kijk.data(), Kijk.ld(), Ojk.data(), Ojk.ld(), Aik.data(), Aik.dims()[0]);
            }
        }
    }

    template<typename T>
    void Knorm(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &Z, T gamma, int bs=16384){
        T alpha = 1, beta = 0, minus=-1;
        T fnK = 0, fnKZtZ = 0;

        DMatrix<T> K(g, bs, bs);
        K.set_constant(0.0);
        DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
        DMatrixDescriptor<T> Zdesc{Z, 0, 0, n, k};
        // if(rk == 0) printf("n::%d, d::%d, k::%d\n", n, d, k);

        for(int i=0; i<n; i+=bs){
            int ib = std::min(bs, n-i);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
            LMatrix<T> Xisqr(Xi.dims()[0], 1);
            vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
            auto Zi=Zdesc.matrix.replicate_in_rows(Zdesc.i1+i, Zdesc.j1, Zdesc.i1+i+ib, Zdesc.j2);
            
            for(int j=0; j<n; j+=bs){
                int jb = std::min(bs, n-j); 
                auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
                auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
                LMatrix<T> Xjsqr(Xj.dims()[0], 1);
                vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

                LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
                auto Xjt=X.replicate_in_columns(0, j, d, j+jb);
                TC_GEMM(rk, Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], alpha, beta, Xi.data(), Xi.ld(), Xjt.data(), Xjt.ld(), Xij.data(), Xij.ld());
                auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);  
                Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0], gamma); 
                fnK+=Kdesc.matrix.fnorm();

                auto Zj=Zdesc.matrix.replicate_in_rows(Zdesc.i1+j, Zdesc.j1, Zdesc.i1+j+jb, Zdesc.j2).transpose();
                // if(rk == 0) printf("Kij[%d,%d] Zi[%d,%d] Zj[%d,%d]\n", Kij.dims()[0], Kij.dims()[1], Zi.dims()[0], Zi.dims()[1], Zj.dims()[0], Zj.dims()[1]);
                TC_GEMM(rk, Zi.dims()[0], Zj.dims()[1], Zi.dims()[1], alpha, minus, Zi.data(), Zi.ld(), Zj.data(), Zj.ld(), Kij.data(), Kij.dims()[0]);
                fnKZtZ+=Kdesc.matrix.fnorm();
            }
        }
        if(rk==0) fmt::print("fnKZtZ::{}, fnK::{}, fnorm(K-ZtZ)/fnorm(K)={}\n", fnKZtZ, fnK, fnKZtZ/fnK);
    }


    // template<typename T>
    // void LRA(int rk, int p, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=16384){
    //     T alpha = 1, beta = 0, beta2 = 0;
    //     DMatrix<T> K(g, bs, bs);
    //     K.set_constant(0.0);
    //     DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
    //     DMatrixDescriptor<T> Adesc{A, 0, 0, n, k};
    //     DMatrixDescriptor<T> Odesc{O, 0, 0, n, k};
        

    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Aik=Adesc.matrix.local_view(Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2); 
            
    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta2=1; 

    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);
    //             LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
    //             sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
    //             // if(rk==0 && i==0) {
    //             //     Xi.save_to_file("XXXi0"+std::to_string(i)+".csv");
    //             //     Xj.save_to_file("XXXj0"+std::to_string(j)+".csv");
    //             //     Xij.save_to_file("XXXij0"+std::to_string(j)+".csv");
    //             // }
    //             auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
                
    //             // if(rk==0 && i==0) {
    //             // //     Xisqr.save_to_file("XXisqr0"+std::to_string(j)+".csv");
    //             // //     Xjsqr.save_to_file("XXjsqr0"+std::to_string(j)+".csv");
    //             // //     Xij.save_to_file("XXij0"+std::to_string(j)+".csv");
    //             // //     Yi.save_to_file("YYi0"+std::to_string(j)+".csv");
    //             // //     Yj.save_to_file("YYj0"+std::to_string(j)+".csv");
    //             //     Kij.save_to_file("KKKij0"+std::to_string(j)+".csv");
    //             // }
                

    //             auto Kijk=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             auto Ojk=Odesc.matrix.replicate_in_columns(Odesc.i1+j, Odesc.j1, Odesc.i1+j+jb, Odesc.j2);
    //             // if(rk == 0) printf("Kijk[%d,%d] Ojk[%d,%d] Aik[%d,%d]\n", Kijk.dims()[0], Kijk.dims()[1], Ojk.dims()[0], Ojk.dims()[1], Aik.dims()[0], Aik.dims()[1]);
    //             sgemm_("N", "N", &Kijk.dims()[0], &Aik.dims()[1], &Kijk.dims()[1], &alpha, Kijk.data(), &Kijk.ld(), Ojk.data(), &Ojk.ld(), &beta2, Aik.data(), &Aik.ld());
    //             // if(rk==0 && i==0) {
    //             //     Kijk.save_to_file("KKKijk0"+std::to_string(j)+".csv");
    //             //     Ojk.save_to_file("OOOjk0"+std::to_string(j)+".csv");
    //             //     Aik.save_to_file("AAAik0"+std::to_string(j)+".csv");
    //             // }
    //         }
    //     }
    // }
}





































































// template<typename T>
// void LRA(int rk, int p, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=16384){
//     T alpha = 1, beta = 0, beta2 = 0;
//     DMatrix<T> K(g, bs, bs);
//     K.set_constant(0.0);
//     DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
//     DMatrixDescriptor<T> Adesc{A, 0, 0, n, k};
//     DMatrixDescriptor<T> Odesc{O, 0, 0, n, k};
//     // if(rk == 0) printf("n::%d, d::%d, k::%d\n", n, d, k);

//     for(int i=0; i<n; i+=bs){
//         int ib = std::min(bs, n-i);
//         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
//         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
//         LMatrix<T> Xisqr(Xi.dims()[0], 1);
//         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
//         auto Aik=Adesc.matrix.replicate_in_rows(Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2);
        
//         for(int j=0; j<n; j+=bs){
//             int jb = std::min(bs, n-j);
//             if(j>0) beta2=1; 

//             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
//             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
//             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
//             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

//             LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
//             auto Xjt=X.replicate_in_columns(0, j, d, j+jb);
//             TC_GEMM(rk, Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], alpha, beta, Xi.data(), Xi.ld(), Xjt.data(), Xjt.ld(), Xij.data(), Xij.ld());
//             // sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
//             auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
//             // if(rk == 0) printf("Xi[%d,%d] Xjt[%d,%d] Xij[%d,%d], Kij[%d,%d]\n", Xi.dims()[0], Xi.dims()[1], Xjt.dims()[0], Xjt.dims()[1], Xij.dims()[0], Xij.dims()[1], Kij.dims()[0], Kij.dims()[1]);
//             rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
//             // Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.ld(), gamma);
            
//             auto Kijk=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
//             auto Ojk=Odesc.matrix.replicate_in_all(Odesc.i1+j, Odesc.j1, Odesc.i1+j+jb, Odesc.j2);
//             // if(rk == 0) printf("Kijk[%d,%d] Ojk[%d,%d] Aik[%d,%d]\n", Kijk.dims()[0], Kijk.dims()[1], Ojk.dims()[0], Ojk.dims()[1], Aik.dims()[0], Aik.dims()[1]);
//             TC_GEMM(rk, Kijk.dims()[0], Aik.dims()[1], Kijk.dims()[1], alpha, beta2, Kijk.data(), Kijk.ld(), Ojk.data(), Ojk.ld(), Aik.data(), Aik.ld());
//             // sgemm_("N", "N", &Kijk.dims()[0], &Aik.dims()[1], &Kijk.dims()[1], &alpha, Kijk.data(), &Kijk.ld(), Ojk.data(), &Ojk.ld(), &beta2, Aik.data(), &Aik.ld());
//         }
//         Adesc.matrix.dereplicate_in_rows(Aik, Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2);
//     }
// }







 // void Knorm(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &Z, T gamma, int bs=16384){
    //     T alpha = 1, beta = 0, beta2 = 0, minus=-1;
    //     T fnK = 0, fnKAtA = 0;
     
    //     DMatrix<T> K(g, bs, bs);
    //     K.set_constant(0.0);
    //     DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
    //     DMatrixDescriptor<T> Zdesc{Z, 0, 0, bs, k};
    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Zi=Zdesc.matrix.replicate_in_rows(Zdesc.i1, Zdesc.j1, Zdesc.i1+ib, Zdesc.j2);

    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta2=1; 

    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

    //             LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
    //             TC_GEMM(rk, 'N', 'T', Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], alpha, beta, Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Xij.data(), Xij.ld());
    //             // sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
    //             // if(rk==0 && i==0) Xij.save_to_file("Xij0"+std::to_string(j)+".csv");
    //             auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             int rk, int m, int n, float *Xi, float *Xj, float *Yi, float *Yj, float *K, int ldk, float gamma)
    //             Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
    //             // rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
    //             // if(rk==0 && i==0) Kij.save_to_file("Kij0"+std::to_string(j)+".csv");
    //             fnK+=K.fnorm();
                
    //             auto Zj=Zdesc.matrix.replicate_in_rows(Zdesc.i1, Zdesc.j1, Zdesc.i1+jb, Zdesc.j2).transpose();
    //             // if(rk == 0) printf("Zi[%d,%d], Zj[%d,%d], Kij[%d,%d]\n", Zi.dims()[0], Zi.dims()[1], Zj.dims()[0], Zj.dims()[1], Kij.dims()[0], Kij.dims()[1]); 
    //             // sgemm_("N", "T", &Zi.dims()[0], &Zj.dims()[0], &Zi.dims()[1], &alpha, Zi.data(), &Zi.ld(), Zj.data(), &Zj.ld(), &minus, Kij.data(), &Kij.ld()); 
    //             TC_GEMM(rk, 'N', 'T', Zi.dims()[0], Zj.dims()[0], Zi.dims()[1], minus, beta2, Zi.data(), Zi.ld(), Zj.data(), Zj.ld(), Kij.data(), Kij.ld());
    //             fnKAtA+=K.fnorm();
    //         }
    //     }
    //     if(rk==0) fmt::print("fnKAtA::{}, fnK::{}, fnorm(K-AtA)/fnorm(K)={}\n", fnKAtA, fnK, fnKAtA/fnK);
    // }








// template<typename T>
    // void LRA(int rk, int p, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=16384){
    //     T alpha = 1, beta = 0, beta2 = 0;
    //     DMatrix<T> K(g, bs, bs);
    //     K.set_constant(0.0);
    //     DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
    //     DMatrixDescriptor<T> Adesc{A, 0, 0, n, k};
    //     DMatrixDescriptor<T> Odesc{O, 0, 0, n, k};
        
    //     // auto Aik=Adesc.matrix.replicate_in_all(Adesc.i1+500, Adesc.j1, Adesc.i1+500+500, Adesc.j2);
    //     // if(rk == 0) printf("Aik[%d,%d,%d,%d] Aik.ld()::%d\n", Adesc.i1+500, Adesc.j1, Adesc.i1+500+500, Adesc.j2, Aik.ld());
    //     // auto Ojk=Odesc.matrix.replicate_in_all(Odesc.i1+500, Odesc.j1, Odesc.i1+500+500, Odesc.j2);
    //     // if(rk == 0) printf("Ojk[%d,%d,%d,%d] Ojk.ld()::%d\n", Odesc.i1+500, Odesc.j1, Odesc.i1+500+500, Odesc.j2, Ojk.ld());

    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Aik=Adesc.matrix.local_view(Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2);
    //         // if(rk == 0) printf("[%d] Aik[%d,%d,%d,%d]\n", i, i, Adesc.j1, i+ib, Adesc.j2);
            
    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta2=1; 

    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             // if(rk == 0) printf("ib::%d, jb::%d, Xi.dims()[0]::%d, Xj.dims()[0]::%d\n", ib, jb, Xi.dims()[0], Xj.dims()[0]);
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

    //             auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             auto Xjt = Xj.transpose();
    //             // if(rk == 0) printf("Kij[%d,%d] Kij.ld()::%d Xi[%d,%d] Xjt[%d,%d]\n", Kij.dims()[0], Kij.dims()[1], Kij.ld(), Xi.dims()[0], Xi.dims()[1], Xjt.dims()[0], Xjt.dims()[1]);
    //             TC_GEMM(rk, Xi.dims()[0], Xjt.dims()[1], Xi.dims()[1], alpha, beta, Xi.data(), Xi.ld(), Xjt.data(), Xjt.ld(), Kij.data(), Kij.ld());
    //             Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Kij.data(), Kij.ld(),  gamma);
    //             K.dereplicate_in_all(Kij, 0, 0, ib, jb);
    //             // if(rk==0 && i==0) Kij.save_to_file("Kij0"+std::to_string(j)+".csv");

    //             auto Kijk=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             auto Ojk=Odesc.matrix.replicate_in_columns(Odesc.i1+j, Odesc.j1, Odesc.i1+j+jb, Odesc.j2);
    //             // if(rk == 0) printf("Kijk[%d,%d] Ojk[%d,%d] Ojk.ld()::%d, Aik[%d,%d] Aik.ld()::%d\n", Kijk.dims()[0], Kijk.dims()[1], Ojk.dims()[0], Ojk.dims()[1], Ojk.ld(), Aik.dims()[0], Aik.dims()[1], Aik.ld());
    //             // if(rk==0 && i==0) {
    //             //     Kijk.save_to_file("Kijk0"+std::to_string(j)+".csv");
    //             //     Ojk.save_to_file("Ojk0"+std::to_string(j)+".csv");
    //             // }
    //             TC_GEMM(rk, Kijk.dims()[0], Aik.dims()[1], Kijk.dims()[1], alpha, beta2, Kijk.data(), Kijk.ld(), Ojk.data(), Ojk.ld(), Aik.data(), Aik.dims()[0]);
    //         }
    //         // if(rk==0 && i==0) Aik.save_to_file("Aik0"+std::to_string(i)+".csv");
    //     }
    // }