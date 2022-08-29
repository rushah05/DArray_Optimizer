#pragma once
#include "darray.h"
#include "cblas_like.h"
#include "blas_like.h"
#include "utils.h"
#include <math.h>

namespace DArray {

    template<typename T>
    void printMatrix(char *filename, int m, int n, T *A, int lda) {
        FILE *f = fopen(filename, "w");
        if (f == NULL) {
        printf("fault!\n");
        return;
        }

        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                fprintf(f, "%f, ", A[i+j*lda]);
            }
            fprintf(f, "\n");
        }

        fclose(f);
    }

    template<typename T>
    void RBF_Kernel(int rk, int np, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &O, DMatrix<T> &A, T gamma, int bs=128){
        T one = 1, zero = 0, minus= -1;

        for(int i=0, j=0; i<n && j<n; i+=bs, j+=bs) {
            int ib = std::min(bs, n-i);
            int jb = std::min(bs, n-j);
            DMatrix<T> Kij(X.grid(), ib, jb);
            Kij.set_constant(0.0);
            auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
            auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
        
            if(rk == 0 && i==0 && j==0) printMatrix("Xi.csv", Xi.dims()[0], Xi.dims()[1], Xi.data(), Xi.ld());
            if(rk == 0 && i==0 && j==0) printMatrix("Xj.csv", Xj.dims()[0], Xj.dims()[1], Xj.data(), Xj.ld());
            DMatrixDescriptor<T> Kij_desc{Kij, 0, 0, ib, jb};
            auto Kijl = Kij_desc.matrix.local_view(Kij_desc.i1, Kij_desc.j1, Kij_desc.i2, Kij_desc.j2);
            sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &one, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &zero, Kijl.data(), &Kijl.ld());
        }
    }


    template<typename T>
    T f(int rk, int n, int k, LMatrix<T> x, LMatrix<T> G){
        // f(x) = (x'*(G*(G'*x))) + sum(x)
        T sone = 1.0f;
        T szero = 0.0f;
        int one=1;
        LMatrix<T> Gx(k, 1);
        sgemm_("T", "T", &k, &one, &n, &sone, G.data(), &G.ld(), x.data(), &x.ld(), &szero, Gx.data(), &Gx.ld()); 
       
        LMatrix<T> GGx(n, 1);
        sgemm_("N", "N", &n, &one, &k, &sone, G.data(), &G.ld(), Gx.data(), &Gx.ld(), &szero, GGx.data(), &GGx.ld());
       
        T xGGx;
        if(rk==0) printf("x[%d,%d], GGx[%d,%d]\n", x.dims()[0], x.dims()[1], GGx.dims()[0], GGx.dims()[1]);
        sgemm_("N", "N", &one, &one, &n, &sone, x.data(), &x.ld(), GGx.data(), &GGx.ld(), &szero, &xGGx, &one);
        
        T xsum=0.0;
        for(int i=0; i<n; i++){
            xsum+=x.data()[i];
        }

        printf("xGGx::%f, xsum::%f\n", xGGx, xsum);
        return xGGx-xsum;
    }

}










 // if(rk == 0) printf("X[%d,%d], Xi[%d,%d], Xj[%d,%d], Kijl[%d,%d], \n", X.dims()[0], X.dims()[1], Xi.dims()[0], Xi.dims()[1],
            // Xj.dims()[0], Xj.dims()[1], Kijl.dims()[0], Kijl.dims()[1], Kijl.ld());

            // if(i==0 && j==0) Kij.collect_and_print("Kij");
            // auto Kijl=Kij.collect(0);
            // if(rk == 0 && i==0 && j==0) printMatrix("Kij.csv", Kijl.dims()[0], Kijl.dims()[1], Kijl.data(), Kijl.ld());