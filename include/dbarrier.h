#pragma once
#include "darray.h"
#include "cblas_like.h"
#include "blas_like.h"
#include "utils.h"
#include <math.h>

namespace DArray {

    template<typename T>
    T f(int rk, int n, int k, LMatrix<T> x, LMatrix<T> G){
        // f(x) = (x'*(G*(G'*x))) + sum(x)
        T done = 1, dzero = 0;
        int one=1;
        LMatrix<T> Gx(k, 1);
        // printf("G[%d,%d] x[%d,%d] Gd[%d,%d] \n", G.dims()[0], G.dims()[1], x.dims()[0], x.dims()[1], Gx.dims()[0], Gx.dims()[1]);
        dgemm_("T", "N", &k, &one, &n, &done, G.data(), &G.ld(), x.data(), &x.ld(), &dzero, Gx.data(), &Gx.ld()); 
       
        LMatrix<T> GGx(n, 1);
        dgemm_("N", "N", &n, &one, &k, &done, G.data(), &G.ld(), Gx.data(), &Gx.ld(), &dzero, GGx.data(), &GGx.ld());
       
        T xGGx = 0;
        // if(rk==0) printf("x[%d,%d], GGx[%d,%d]\n", x.dims()[0], x.dims()[1], GGx.dims()[0], GGx.dims()[1]);
        dgemm_("T", "N", &one, &one, &n, &done, x.data(), &x.ld(), GGx.data(), &GGx.ld(), &dzero, &xGGx, &one);
        
        T xsum = 0;
        for(int i=0; i<n; i++){
            xsum+=x.data()[i];
        }

        // if(rk == 0) printf("xGGx::%f, xsum::%f, fval::%f\n", xGGx, xsum, xGGx-xsum);
        return xGGx-xsum;
    }


    template<typename T>
    T g(int rk, int n, int k, LMatrix<T> x, LMatrix<T> G, double C, int t){
        T gval = 0;
        for(int i=0; i<n; ++i){
            if(x.data()[i]<0.0 || x.data()[i]>C){
                gval=-1;
                break;
            }
        }
        if(gval==0){
            T fval = f(rk, n, k, x, G);
            T xsum = 0;
            for(int i=0; i<n; i++){
                xsum+=(log(x.data()[i])+log(C-x.data()[i]))/t;
            }
            gval=fval-xsum;
        }
        return gval;
    }
}






