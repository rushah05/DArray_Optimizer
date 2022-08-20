#pragma once
#include "darray.h"
#include "blas_like.h"
#include "utils.h"
#include <math.h>

namespace DArray {

    template<typename T>
    void rbf(int m, int n, T *Xi, T *Xj, T *Yi, T* Yj, T *K, int ldk, T gamma){
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                K[i+j*ldk]=exp(-gamma*(Xi[i]-2*K[i+j*ldk]+Xj[j]))*Yi[i]*Yj[j];
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
    void LRA(int rk, int np, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=8192){
        float sone=1.0f;
        float szero=0.0f;
        for(int i=0, j=0; i<n && j<n; i+=bs, j+=bs) {
            int ib = std::min(bs, n-i);
            int jb = std::min(bs, n-j);
            // printf("[%d,%d], ib::%d, jb::%d\n", i, j, ib, jb);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
            auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
            auto Xj=X.transpose_and_replicate_in_rows(0, j, d, j+jb);
            auto Yj=Y.transpose_and_replicate_in_rows(0, j, 1, j+jb);
            auto Oj=O.replicate_in_rows(j, 0, j+jb, k);
            auto Ai=A.replicate_in_rows(i, 0, i+ib, k);
            int lm=Xi.dims()[0];
            int ln=Xj.dims()[0];
            int ld=Xi.dims()[1];
            int lk=Oj.dims()[1];
            // if(rk == 0) printf("lm::%d, ln::%d, ld::%d, lk=%d\n", lm, ln, ld, lk);
            LMatrix<T> Xisqr(lm, 1), Xjsqr(ln, 1), K(lm, ln);
            vecnorm<float>(Xi.data(), Xi.ld(), Xisqr.data(), lm, ld);
            vecnorm<float>(Xj.data(), Xj.ld(), Xjsqr.data(), ln, ld);
            // if(rk == 0) printf("Xi[%d,%d], Xj[%d,%d], K[%d,%d]\n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1], K.dims()[0], K.dims()[1]);
            sgemm_("N", "T", &lm, &ln, &ld, &sone, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &szero, K.data(), &K.ld());
            rbf<float>(lm, ln, Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), K.data(), K.ld(), gamma);
            // if(rk == 0) printf("K[%d,%d], Oj[%d,%d], Ai[%d,%d], lm::%d, ln::%d, lk::%d\n", K.dims()[0], K.dims()[1], Oj.dims()[0], Oj.dims()[1], Ai.dims()[0], Ai.dims()[1], lm, ln, lk);
            sgemm_("N", "N", &lm, &lk, &ln, &sone, K.data(), &K.ld(), Oj.data(), &Oj.ld(), &szero, Ai.data(), &Ai.ld()); 
            A.dereplicate_in_rows(Ai, i, 0, i+ib, k);
            // if(rk == 0) Ai.print("Ai");
        }
        
    }

} 