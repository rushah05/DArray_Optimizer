#pragma once
#include "darray.h"
#include "blas_like.h"
#include "utils.h"
#include <math.h>

namespace DArray {

    template<typename T>
    void printMatrix(const std::string filepath, int m, int n, DArray::LMatrix<T>& A, int lda) {
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
                K[i+j*ldk]=exp(-gamma*(Xi[i]-2*Xij[i+j*ldxij]+Xj[j]))*Yi[i]*Yj[j];
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


    // template<typename T>
    // void Knorm(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &Z, T gamma, int bs=8192){
    //     T alpha = 1, beta = 0, minus=-1;
    //     T fnorm = 0, fnorm_res = 0;
    //     T all_fnorm = 0, all_fnorm_res = 0;
    //     DMatrix<T> K(g, bs, bs);
        
    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Zi=Z.replicate_in_columns(i, 0, i+ib, k);

    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta=1; 
    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

    //             auto Kij=K.local_view(0, 0, ib, jb);
    //             sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Kij.data(), &Kij.dims()[0]);
    //             rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Kij.data(), Kij.dims()[0], gamma);
    //             // if(rk == 0) printf("[Rank::%d] X[%d,%d], Xl[%d,%d] Xi[%d,%d] Xj[%d,%d] Kij[%d,%d]\n", rk, X.dims()[0], X.dims()[1], X.ldims()[0], X.ldims()[1], Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Kij.dims()[0], Kij.dims()[1]);
    //             K.dereplicate_in_all(Kij, 0, 0, Kij.dims()[0], Kij.dims()[1]);
    //             auto Kij1=K.replicate_in_all(0, 0, ib, jb);
    //             for(int a=0; a<Kij1.dims()[0]; a++) {
    //                 for(int b=0; b<Kij1.dims()[1]; b++) {
    //                     fnorm += (Kij1.data()[a+b*Kij1.ld()] * Kij1.data()[a+b*Kij1.ld()]);
    //                 }
    //             }

    //             // auto Kij=K.local_view(0, 0, ib, jb);
    //             auto Zj=Z.transpose_and_replicate_in_rows(j, 0, j+jb, k);
    //             // if(rk == 0) printf(" Zi[%d,%d], Zj[%d,%d] Kij1[%d,%d]\n", Zi.dims()[0], Zi.dims()[1], Zj.dims()[0], Zj.dims()[1], Kij1.dims()[0], Kij1.dims()[1]);
    //             sgemm_("N", "N", &Zi.dims()[0], &Zj.dims()[0], &Zi.dims()[1], &minus, Zi.data(), &Zi.ld(), &Zj.data()[j], &Zj.ld(), &beta, Kij1.data(), &Kij1.ld());
    //             for(int a=0; a<Kij1.dims()[0]; a++) {
    //                 for(int b=0; b<Kij1.dims()[1]; b++) {
    //                     fnorm_res += (Kij1.data()[a+b*Kij1.ld()] * Kij1.data()[a+b*Kij1.ld()]);
    //                 }
    //             }
    //         }
    //     }
        
    //     MPI_Allreduce(&fnorm, &all_fnorm, 1, MPI_FLOAT, MPI_SUM, g.comm()); 
    //     MPI_Allreduce(&fnorm_res, &all_fnorm_res, 1, MPI_FLOAT, MPI_SUM, g.comm());        
    //     if(rk == 0) printf("all_form::%d, all_fnorm_res::%d, form::%d, fnorm_res::%d\n", sqrt(all_fnorm), sqrt(all_fnorm_res), sqrt(fnorm), sqrt(fnorm_res));
    // }


    template<typename T>
    void Knorm(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &Z, T gamma, int bs=8192){
        T alpha = 1, beta = 0, beta2 = 0, minus=-1;
        T fnorm = 0, fnorm_res = 0;
        T all_fnorm = 0, all_fnorm_res = 0;
     
        DMatrix<T> K(g, bs, bs);
        K.set_constant(0.0);
        DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
        DMatrixDescriptor<T> Zdesc{Z, 0, 0, Z.dims()[0], Z.dims()[1]};

        for(int i=0; i<n; i+=bs){
            int ib = std::min(bs, n-i);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
            LMatrix<T> Xisqr(Xi.dims()[0], 1);
            vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
            auto Zi=Z.replicate_in_columns(Zdesc.i1+i, Zdesc.j1, Zdesc.i1+i+ib, Zdesc.j2);
            
            for(int j=0; j<n; j+=bs){
                int jb = std::min(bs, n-j);
                if(j>0) beta2=1; 

                auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
                auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
                // if(rk==0 && i==0) Xi.save_to_file("Xi0"+std::to_string(i)+".csv");
                // if(rk==0 && i==0) Xj.save_to_file("Xj0"+std::to_string(j)+".csv");
                LMatrix<T> Xjsqr(Xj.dims()[0], 1);
                vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

                LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
                sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
                // if(rk==0 && i==0) Xij.save_to_file("Xij0"+std::to_string(j)+".csv");
                auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
                rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
                // if(rk==0 && i==0) Kij.save_to_file("Kij0"+std::to_string(j)+".csv");
                auto Kij1=Kdesc.matrix.replicate_in_all(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
                for(int a=0; a<Kij1.dims()[0]; a++) {
                    for(int b=0; b<Kij1.dims()[1]; b++) {
                        fnorm += (Kij1.data()[a+b*Kij1.ld()] * Kij1.data()[a+b*Kij1.ld()]);
                    }
                }

                auto Zj=Zdesc.matrix.transpose_and_replicate_in_rows(Zdesc.i1+j, Zdesc.j1, Zdesc.i1+j+jb, Zdesc.j2);
                 // if(rk == 0) printf(" Zi[%d,%d], Zj[%d,%d] Kij1[%d,%d]\n", Zi.dims()[0], Zi.dims()[1], Zj.dims()[0], Zj.dims()[1], Kij1.dims()[0], Kij1.dims()[1]);
                sgemm_("N", "N", &Zi.dims()[0], &Zj.dims()[0], &Zi.dims()[1], &minus, Zi.data(), &Zi.ld(), &Zj.data(), &Zj.ld(), &beta2, Kij1.data(), &Kij1.ld());
                for(int a=0; a<Kij1.dims()[0]; a++) {
                    for(int b=0; b<Kij1.dims()[1]; b++) {
                        fnorm_res += (Kij1.data()[a+b*Kij1.ld()] * Kij1.data()[a+b*Kij1.ld()]);
                    }
                }
               
            }
        }
        
        MPI_Allreduce(&fnorm, &all_fnorm, 1, MPI_FLOAT, MPI_SUM, g.comm()); 
        MPI_Allreduce(&fnorm_res, &all_fnorm_res, 1, MPI_FLOAT, MPI_SUM, g.comm());        
        if(rk == 0) printf("all_form::%d, all_fnorm_res::%d, form::%d, fnorm_res::%d\n", sqrt(all_fnorm), sqrt(all_fnorm_res), sqrt(fnorm), sqrt(fnorm_res));
    }


    template<typename T>
    void LRA(int rk, int p, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=8192){
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
                // if(rk==0 && i==0) Xi.save_to_file("Xi0"+std::to_string(i)+".csv");
                // if(rk==0 && i==0) Xj.save_to_file("Xj0"+std::to_string(j)+".csv");
                LMatrix<T> Xjsqr(Xj.dims()[0], 1);
                vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);

                LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
                sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
                // if(rk==0 && i==0) Xij.save_to_file("Xij0"+std::to_string(j)+".csv");
                auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
                rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
                // if(rk==0 && i==0) Kij.save_to_file("Kij0"+std::to_string(j)+".csv");

                auto Kijk=Kdesc.matrix.replicate_in_rows(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
                auto Ojk=Odesc.matrix.replicate_in_columns(Odesc.i1+j, Odesc.j1, Odesc.i1+j+jb, Odesc.j2);
                // if(rk == 0) printf("Kijk[%d,%d] Ojk[%d,%d] Aik[%d,%d]\n", Kijk.dims()[0], Kijk.dims()[1], Ojk.dims()[0], Ojk.dims()[1], Aik.dims()[0], Aik.dims()[1]);
                sgemm_("N", "N", &Kijk.dims()[0], &Aik.dims()[1], &Kijk.dims()[1], &alpha, Kijk.data(), &Kijk.ld(), Ojk.data(), &Ojk.ld(), &beta2, Aik.data(), &Aik.ld());
                // if(rk==0 && i==0) {
                //     Kijk.save_to_file("Kijk0"+std::to_string(j)+".csv");
                //     Ojk.save_to_file("Ojk0"+std::to_string(j)+".csv");
                // }
            }
            // if(rk==0 && i==0) Aik.save_to_file("Aik0"+std::to_string(i)+".csv");
        }
    }
}








// auto Kij1=K.replicate_in_all(0, 0, ib, jb);
//                 for(int a=0; a<Kij1.dims()[0]; a++) {
//                     for(int b=0; b<Kij1.dims()[1]; b++) {
//                         fnorm += (Kij1.data()[a+b*Kij1.ld()] * Kij1.data()[a+b*Kij1.ld()]);
//                     }
//                 }

                

    // template<typename T>
    // void LRA(int rk, int p, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=8192){
    //     T alpha = 1, beta = 0;
    //     DMatrix<T> K(g, bs, bs);
    //     DMatrixDescriptor<T> Adesc{A, 0, 0, n, k};
    //     LMatrix<T> Ajk(bs, k);

    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Aik=Adesc.matrix.local_view(Adesc.i1+i, Adesc.j1, Adesc.i1+i+ib, Adesc.j2); 
            
    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta=1; 

    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);
    //             auto Kij=K.local_view(0, 0, ib, jb);
    //             sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Kij.data(), &Kij.dims()[0]);
    //             // if (rk == 0) Kij.save_to_file("Kij00.csv");
    //             rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Kij.data(), Kij.dims()[0], gamma);
    //             // if(rk==1) Kij.save_to_file("KRBF01.csv");
    //     //         // if(rk == 0) printf("[Rank::%d] X[%d,%d], Xl[%d,%d] Xi[%d,%d] Xj[%d,%d] Kij[%d,%d]\n", rk, X.dims()[0], X.dims()[1], X.ldims()[0], X.ldims()[1], Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], Kij.dims()[0], Kij.dims()[1]);
    //             K.dereplicate_in_all(Kij, 0, 0, Kij.dims()[0], Kij.dims()[1]);

    //             auto Kijk=K.replicate_in_rows(0, 0, ib, jb);
    //             // if(rk==0 && i==0 && j==8192) Kijk.print("Kijk");
    //             auto Ojk=O.replicate_in_columns(j, 0, j+jb, k);
    //             // if(rk == 0) printf("Kijk[%d,%d] Ojk[%d,%d] Aik[%d,%d]\n", Kijk.dims()[0], Kijk.dims()[1], Ojk.dims()[0], Ojk.dims()[1], Aik.dims()[0], Aik.dims()[1]);
    //             sgemm_("N", "N", &Kijk.dims()[0], &Aik.dims()[1], &Kijk.dims()[1], &alpha, Kijk.data(), &Kijk.ld(), &Ojk.data()[j], &Ojk.ld(), &beta, Ajk.data(), &Ajk.ld());
    //             if(rk==0 && i==0){
    //                 printMatrix("Aik0"+std::to_string(j)+".csv", Aik.dims()[0], Aik.dims()[1], Aik, Aik.ld());
    //                 // Aik.save_to_file("Aik0"+std::to_string(j)+".csv");
    //             }
    //         }
    //         MPI_Reduce(kai, ka, (rsize * k), MPI_FLOAT, MPI_SUM, i, MPI_COMM_WORLD);
    //     }
    // }




