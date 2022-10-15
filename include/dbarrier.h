#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include "darray.h"
#include "cblas_like.h"

namespace DArray{

    template<typename T>
    T f(int rk, int n, int k, LMatrix<T>& x, LMatrix<T>& Z){
        // f(x) = 0.5*(x'*(Z*(Z'*x))) - sum(x)
        T done = 1, dzero = 0, dhalf=0.5;
        int one=1;

        LMatrix<T> Zx(k, 1);
        dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), x.data(), &x.ld(), &dzero, Zx.data(), &Zx.ld()); 

        LMatrix<T> ZZx(n, 1);
        dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Zx.data(), &Zx.ld(), &dzero, ZZx.data(), &ZZx.ld()); 
        
        T xZZx = 0;
        dgemm_("T", "N", &one, &one, &n, &dhalf, x.data(), &x.ld(), ZZx.data(), &ZZx.ld(), &dzero, &xZZx, &one);
        
        T xsum = 0;
        for(int i=0; i<n; i++){
            xsum+=x.data()[i];
        }

        // if(rk == 0) printf("xZZx::%f, xsum::%f, fval::%f\n", xZZx, xsum, xZZx-xsum);
        return xZZx-xsum;
    }

    template<typename T>
    T gg(int rk, int n, int k, LMatrix<T>& x, LMatrix<T>& Z, T C, T t){
        T lgsum=0.0;
        for(int i=0; i<n; ++i){
            lgsum+=(log(x.data()[i]) + log(C-x.data()[i]))/t;
        }
        T fval=f(rk, n, k, x, Z);
        return fval-lgsum;
    }

    template<typename T>
    void Grad_f(int n, int k, LMatrix<T>& Z, LMatrix<T>& a, LMatrix<T>& gradf){
        // Gradf=(G*a - 1.0);
        gradf.set_constant(1.0);
        T dminus=-1, done=1, dzero=0;
        int one=1;
        LMatrix<T> Za(k, 1);
        // printf(" n=%d, k=%d, ldz=%d, ldza=%d, lda=%d \n", n, k, Z.ld(), Za.ld(), a.ld());
        dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), a.data(), &a.ld(), &dzero, Za.data(), &Za.ld());
        dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Za.data(), &Za.ld(), &dminus, gradf.data(), &gradf.ld());
    }

    template<typename T>
    void Grad_Phi(int n, LMatrix<T>& a, T C, T t, LMatrix<T>& gradphi){
        //  GradPhi = (-1.0 ./a + 1.0 ./(C-a))/t;
        for(int i=0; i<n; ++i){
            gradphi.data()[i] = ((-1.0/a.data()[i]) + (1.0/(C-a.data()[i])))/t;
        }
    }

    template<typename T>
    void Grad(int n, LMatrix<T>& gradf, LMatrix<T>& gradphi, LMatrix<T>& grad){
        // Grad = Gradf +  GradPhi;
        for(int i=0; i<n; ++i){
            grad.data()[i] = (gradf.data()[i] + gradphi.data()[i]);
        }
    }

    template<typename T>
    void Hess_Phi(int n, LMatrix<T>& a, T C, T t, LMatrix<T>& D){
        // D = (1./(a.*a) + 1./((C-a).*(C-a)))./t; % Hessian of phi(a)
        for(int i=0; i<n; ++i){
            D.data()[i] = ((1.0/(a.data()[i]*a.data()[i])) + (1.0/((C-a.data()[i])*(C-a.data()[i]))))/t;
        }
    }

    template<typename T>
    LMatrix<T> pre_conjgrad(int rk, int n, int k, LMatrix<T>& Z, LMatrix<T>& M, LMatrix<T>& D, LMatrix<T>& b, LMatrix<T>& a){
        // r=b-(ZZ' + D + I)x , r is residual
        T done=1, dzero=0;
        int one=1;
        LMatrix<T> Zx(k, 1);
        dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), a.data(), &a.ld(), &dzero, Zx.data(), &Zx.ld()); 
        LMatrix<T> ZZx(n, 1);
        dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Zx.data(), &Zx.ld(), &dzero, ZZx.data(), &ZZx.ld()); 

        for(int i=0; i<n; ++i){
            ZZx.data()[i] += (D.data()[i]*a.data()[i]) + (n*a.data()[i]);
        }

        LMatrix<T> r(n,1), z(n,1), p(n,1), x(n,1);
        for(int i=0; i<n; ++i){
            r.data()[i] = b.data()[i] - ZZx.data()[i];
            z.data()[i] = r.data()[i]/M.data()[i];
            p.data()[i] = z.data()[i];
            x.data()[i] = a.data()[i];
        }
        
        T rsold=ddot_(&n, r.data(), &one, z.data(), &one);
        int iter=0;

        LMatrix<T> Zp(k, 1), ZZp(n, 1);
        T alpha =0.0, rsnew = 0.0;

        T normr=dnrm2_(&n, r.data(), &one);
        if(rk == 0) printf("norm(r)=%f \n", normr);

        while(normr > 1.0e-2){
            dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), p.data(), &p.ld(), &dzero, Zp.data(), &Zp.ld());
            dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Zp.data(), &Zp.ld(), &dzero, ZZp.data(), &ZZp.ld());
            for(int i=0; i<n; ++i){
                ZZp.data()[i] += (D.data()[i]*p.data()[i]) + (n*p.data()[i]);
            }            
            alpha = ddot_(&n, p.data(), &one, ZZp.data(), &one);
            alpha=rsold/alpha;

            for(int i=0; i<n; ++i){
                x.data()[i] = x.data()[i] + alpha * p.data()[i];
                r.data()[i] = r.data()[i] - alpha * ZZp.data()[i];
                z.data()[i]= r.data()[i]/M.data()[i];
            }

            rsnew=ddot_(&n, r.data(), &one, z.data(), &one);
            for(int i=0; i<n; ++i){
                p.data()[i] = z.data()[i] + (rsnew/rsold) * p.data()[i];
            }
            rsold=rsnew;
            normr=dnrm2_(&n, r.data(), &one);
            if(rk == 0) printf("%d, norm(r)::%f \n", iter, normr);
            iter+=1;
        }
        return x;
    }



    // template<typename T>
    // LMatrix<T> conjgrad(int n, int k, LMatrix<T>& Z, LMatrix<T>& M, LMatrix<T>& D, LMatrix<T>& b, LMatrix<T>& a, T t){
    //     // r=b-(Z*Z' + D)x , r is residual
    //     T done=1, dzero=0;
    //     int one=1;
    //     LMatrix<T> Zx(k, 1);
    //     dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), a.data(), &a.ld(), &dzero, Zx.data(), &Zx.ld()); 
    //     LMatrix<T> ZZx(n, 1);
    //     dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Zx.data(), &Zx.ld(), &dzero, ZZx.data(), &ZZx.ld()); 

    //     LMatrix<T> Dx(n, 1),  nIx(n, 1);
    //     for(int i=0; i<n; ++i){
    //         Dx.data()[i] = (D.data()[i]/t)*a.data()[i];
    //         nIx.data()[i] = (n * a.data()[i]);
    //     }

        
    //     LMatrix<T> r(n,1), p(n,1), x(n,1), z(n,1);
    //     for(int i=0; i<n; ++i){
    //         r.data()[i] = b.data()[i]- (ZZx.data()[i] + Dx[i] + nIx[i]);
    //         // z.data()[i] = r.data()[i];
    //     }

        
    //     // int info=-1;
    //     // dpotrf_("L", &n, M.data(), &M.ld(), &info);
    //     // if(info !=0) printf("info after dpotrf_ =%d \n", info);
    //     // info =-1;
    //     // dpotrs_("L", &n, &one, M.data(), &M.ld(), z.data(), &z.ld(), &info);
    //     // if(info !=0) printf("info after dpotrs_ =%d \n", info);
    //     for(int i=0; i<n; i++){
    //         z.data()[i] = (r.data()[i] * (1/M.data()[i]));
    //     }


    //     for(int i=0; i<n; ++i){
    //         p.data()[i] = z.data()[i];
    //         x.data()[i] = a.data()[i];
    //     }
        
    //     T rsold=ddot_(&n, r.data(), &one, z.data(), &one);
    //     int pi=1;

    //     LMatrix<T> Zp(k, 1), ZZp(n, 1);
    //     T alpha =0.0, rsnew = 0.0;

    //     T normr=dnrm2_(&n, r.data(), &one);
    //     printf("norm(r)=%f \n", normr);

    //     while(normr > 1.0e-2){
    //         dgemm_("T", "N", &k, &one, &n, &done, Z.data(), &Z.ld(), p.data(), &p.ld(), &dzero, Zp.data(), &Zp.ld());
    //         dgemm_("N", "N", &n, &one, &k, &done, Z.data(), &Z.ld(), Zp.data(), &Zp.ld(), &dzero, ZZp.data(), &ZZp.ld());
    //         alpha = ddot_(&n, p.data(), &one, ZZp.data(), &one);
    //         alpha=rsold/alpha;

    //         for(int i=0; i<n; ++i){
    //             x.data()[i] = x.data()[i] + alpha * p.data()[i];
    //             r.data()[i] = r.data()[i] - alpha * ZZp.data()[i];
    //             // z.data()[i] = r.data()[i];
    //         }
            
    //         // info=-1;
    //         // dpotrf_("L", &n, M.data(), &M.ld(), &info);
    //         // if(info !=0) printf("%d, info after dpotrf_ =%d \n", pi, info);
    //         // info =-1;
    //         // dpotrs_("L", &n, &one, M.data(), &M.ld(), z.data(), &z.ld(), &info);
    //         // if(info !=0) printf("%d, info after dpotrs_ =%d \n", pi, info);
    //         for(int i=0; i<n; i++){
    //             z.data()[i] = (r.data()[i] * (1/M.data()[i]));
    //         }

    //         rsnew=ddot_(&n, r.data(), &one, z.data(), &one);
    //         for(int i=0; i<n; ++i){
    //             p.data()[i] = z.data()[i] + ((rsnew/rsold) * p.data()[i]);
    //         }
    //         rsold=rsnew;
    //         normr=dnrm2_(&n, r.data(), &one);
    //         printf("%d, norm(r)::%f \n", pi, normr);
    //         pi+=1;
    //     }
    //     return x;
    // }































    template<typename T>
    T f(int rk, int n, LMatrix<T>& x, LMatrix<T>& G){
        // f(x) = 0.5*(x'*(G*x))) - sum(x)
        T done = 1, dzero = 0, dhalf=0.5;
        int one=1;

        LMatrix<T> Gx(n, 1);
        dgemm_("N", "N", &n, &one, &n, &done, G.data(), &G.ld(), x.data(), &x.ld(), &dzero, Gx.data(), &Gx.ld()); 
        
        T xGx = 0;
        dgemm_("T", "N", &one, &one, &n, &dhalf, x.data(), &x.ld(), Gx.data(), &Gx.ld(), &dzero, &xGx, &one);
        
        T xsum = 0;
        for(int i=0; i<n; i++){
            xsum+=x.data()[i];
        }

        // if(rk == 0) printf("xGx::%f, xsum::%f, fval::%f\n", xGx, xsum, xGx-xsum);
        return xGx-xsum;
    }

    template<typename T>
    T gg(int rk, int n, LMatrix<T>& x, LMatrix<T>& G, T C, T t){
        T lgsum=0.0;
        for(int i=0; i<n; ++i){
            lgsum+=(log(x.data()[i]) + log(C-x.data()[i]))/t;
        }
        T fval=f(rk, n, x, G);
        return fval-lgsum;
    }


    template<typename T>
    void Grad_f(int n, LMatrix<T>& G, LMatrix<T>& a, LMatrix<T>& gradf){
        // Gradf=(G*a - 1.0);
        T dminus=-1, done=1;
        int one=1;
        dgemm_("N", "N", &n, &one, &n, &done, G.data(), &G.ld(), a.data(), &a.ld(), &dminus, gradf.data(), &gradf.ld());
    }

    // template<typename T>
    // void Grad_Phi(int n, LMatrix<T>& a, T C, T t, LMatrix<T>& gradphi){
    //     //  GradPhi = (-1.0 ./a + 1.0 ./(C-a))/t;
    //     for(int i=0; i<n; ++i){
    //         gradphi.data()[i] = ((-1.0/a.data()[i]) + (1.0/(C-a.data()[i])))/t;
    //     }
    // }

    // template<typename T>
    // void Grad(int n, LMatrix<T>& gradf, LMatrix<T>& gradphi, LMatrix<T>& grad){
    //     // Grad = Gradf +  GradPhi;
    //     for(int i=0; i<n; ++i){
    //         grad.data()[i] = (gradf.data()[i] + gradphi.data()[i]);
    //     }
    // }

    // template<typename T>
    // void Hess_Phi(int n, LMatrix<T>& a, T C, T t, LMatrix<T>& D){
    //     // D = (1./(a.*a) + 1./((C-a).*(C-a)))./t; % Hessian of phi(a)
    //     for(int i=0; i<n; ++i){
    //         D.data()[i] = ((1.0/(a.data()[i]*a.data()[i])) + (1.0/((C-a.data()[i])*(C-a.data()[i]))))/t;
    //     }
    // }

    template<typename T>
    void createH(int n, LMatrix<T>& dK, LMatrix<T>& D, T t, LMatrix<T>& dH){
        // H = G+diag(D)/t;
        dH = dK;
        for(int i=0; i<n; ++i){
            dH.data()[i+i*dH.ld()] += (D.data()[i]/t);
        }
    }

    template<typename T>
    LMatrix<T> conjgrad(int n, LMatrix<T>& G, LMatrix<T>& M, LMatrix<T>& b, LMatrix<T>& a){
        // r=b-Gx , r is residual
        LMatrix<T> Gx(n, 1);
        T done=1, dzero=0;
        int one=1;
        dgemm_("N", "N", &n, &one, &n, &done, G.data(), &G.ld(), a.data(), &a.ld(), &dzero, Gx.data(), &Gx.ld()); 

        LMatrix<T> r(n,1), z(n,1), p(n,1), x(n,1);
        for(int i=0; i<n; ++i){
            r.data()[i] = b.data()[i]-Gx.data()[i];
            z.data()[i] = r.data()[i];
        }

        int info=-1;
        dpotrf_("L", &n, M.data(), &M.ld(), &info);
        if(info !=0) printf("info after dpotrf_ =%d \n", info);
        info =-1;
        dpotrs_("L", &n, &one, M.data(), &M.ld(), z.data(), &z.ld(), &info);
        if(info !=0) printf("info after dpotrs_ =%d \n", info);

        for(int i=0; i<n; ++i){
            p.data()[i] = z.data()[i];
            x.data()[i] = a.data()[i];
        }
        
        T rsold=ddot_(&n, r.data(), &one, z.data(), &one);
        int pi=1;
        LMatrix<T> Gp(n, 1);
        T alpha =0.0, rsnew = 0.0;

        T normr=dnrm2_(&n, r.data(), &one);
        printf("norm(r)=%f \n", normr);

        while(normr > 1.0e-2){
            dgemm_("N", "N", &n, &one, &n, &done, G.data(), &G.ld(), p.data(), &p.ld(), &dzero, Gp.data(), &Gp.ld());
            alpha = ddot_(&n, p.data(), &one, Gp.data(), &one);
            alpha=rsold/alpha;

            for(int i=0; i<n; ++i){
                x.data()[i] = x.data()[i] + alpha * p.data()[i];
                r.data()[i] = r.data()[i] - alpha * Gp.data()[i];
                z.data()[i]=r.data()[i];
            }
            
            info=-1;
            dpotrf_("L", &n, M.data(), &M.ld(), &info);
            if(info !=0) printf("%d, info after dpotrf_ =%d \n", pi, info);
            info =-1;
            dpotrs_("L", &n, &one, M.data(), &M.ld(), z.data(), &z.ld(), &info);
            if(info !=0) printf("%d, info after dpotrs_ =%d \n", pi, info);

            rsnew=ddot_(&n, r.data(), &one, z.data(), &one);
            for(int i=0; i<n; ++i){
                p.data()[i] = z.data()[i] + (rsnew/rsold) * p.data()[i];
            }
            rsold=rsnew;
            normr=dnrm2_(&n, r.data(), &one);
            printf("%d, norm(r)::%f \n", pi, normr);
            pi+=1;
        }
        return x;
    }


    template<typename T>
    void writemodel(const std::string filepath, int n, int d, int k, LMatrix<T>& a, T C, LMatrix<T>& dZ, LMatrix<T> dY, LMatrix<T> dX, T gamma, int minus, int plus){
        int nSV = 0, nBSV = 0;
        for( int i=0; i<n; i++ ){
            if( a.data()[i] > 1e-6 ) {
                nSV++;
                if( a.data()[i] < C-1e-6 ) {
                    nBSV++;
                }
            }
        }

        int *iSV = (int*) malloc(sizeof(int)*nSV);
        int *iBSV = (int*) malloc(sizeof(int)*nBSV);

        int svi = 0, bsvi = 0;
        for( int i=0; i<n; i++ ) {
            if( a.data()[i] > 1e-6 ) {
                iSV[svi++] = i;
                if( a.data()[i] < C-1e-6 ) {
                    iBSV[bsvi++] = i;
                }
            }
        }

        printf("#BSV %d, #SV %d\n", nBSV, nSV);
        double b = 0;
        double acc = 0;
        std::vector<double> bs(std::min(nBSV,50), 0);
        for (int j=0; j<std::min(nBSV,50); j++) {
            int jj = iBSV[j];
            double yj = dY[jj];
            for (int i=0; i<nSV; i++) {
                int ii = iSV[i];
                double sum = 0; 
                for (int kk=0; kk<k; kk++) {
                    sum += dZ[ii+kk*dZ.ld()] * dZ[ii+kk*dZ.ld()];
                }
                yj -= a[ii] * dY[jj] * sum; 
            }
            acc += yj;
            bs[j] = yj;
            // printf("y[%d]=%.3e\n", jj, yj);
        }
        b = acc/std::min(nBSV,50);
        double sumsq = 0;
        for( int j=0; j<bs.size(); j++ ) 
            sumsq += (bs[j]-b)*(bs[j]-b);
        printf("approx mean b=%.6e std b=%.6e, #samples=%d\n ", b, sqrt(sumsq/bs.size()), bs.size());


        // writing files in LIBSVM format
        FILE *f = fopen(filepath.c_str(), "w");
        assert(f);
        
        fprintf(f,"svm_type c_svc\n");
        fprintf(f,"kernel_type rbf\n");
        fprintf(f,"gamma %.7f\n", gamma);
        
        fprintf(f,"nr_class 2\n");
        fprintf(f,"total_sv %d\n", nSV);
        fprintf(f,"rho %f\n", -b);
        fprintf(f,"label %d %d\n", plus, minus);
        fprintf(f,"nr_sv %d %d\n", nBSV, nSV-nBSV);
        fprintf(f,"SV\n");
        for( int i=0; i<nSV; i++ ) {
            int j = iSV[i];
            fprintf(f, "%7f ", dY[j]*a[j]);
            for( int kk=0; kk<d; kk++ ) {
                if( dX[j*dX.ld()+kk]>0 || dX[j*dX.ld()+kk]<0) {
                    fprintf(f, "%d:%7f ", kk+1, dX[j*d+kk]);
                }
            }
            fprintf(f, "\n");
        }
        fclose(f);
    }

}































    // T f(int rk, int n, int k, LMatrix<T>& x, LMatrix<T>& G){
    //     // f(x) = (x'*(G*(G'*x))) + sum(x)
    //     T done = 1, dzero = 0;
    //     int one=1;
    //     LMatrix<T>& Gx(k, 1);
    //     // printf("G[%d,%d] x[%d,%d] Gd[%d,%d] \n", G.dims()[0], G.dims()[1], x.dims()[0], x.dims()[1], Gx.dims()[0], Gx.dims()[1]);
    //     sgemm_("T", "N", &k, &one, &n, &done, G.data(), &G.ld(), x.data(), &x.ld(), &dzero, Gx.data(), &Gx.ld()); 
       
    //     LMatrix<T>& GGx(n, 1);
    //     sgemm_("N", "N", &n, &one, &k, &done, G.data(), &G.ld(), Gx.data(), &Gx.ld(), &dzero, GGx.data(), &GGx.ld());
       
    //     T xGGx = 0;
    //     // if(rk==0) printf("x[%d,%d], GGx[%d,%d]\n", x.dims()[0], x.dims()[1], GGx.dims()[0], GGx.dims()[1]);
    //     sgemm_("T", "N", &one, &one, &n, &done, x.data(), &x.ld(), GGx.data(), &GGx.ld(), &dzero, &xGGx, &one);
        
    //     T xsum = 0;
    //     for(int i=0; i<n; i++){
    //         xsum+=x.data()[i];
    //     }

    //     // if(rk == 0) printf("xGGx::%f, xsum::%f, fval::%f\n", xGGx, xsum, xGGx-xsum);
    //     return xGGx-xsum;
    // }