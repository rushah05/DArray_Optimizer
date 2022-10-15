#include "darray.h"
#include "read_input.h"
#include "dkernel.h"
#include "lapack_like.h"
#include "dbarrier.h"

extern void cuda_init();
extern void cuda_finalize();

template<typename T>
T gaussrand(){
    static T v1, v2, s;
    static int phase = 0;
    T x;
    
    if(phase == 0) {
        do {
            T U1 = (T)rand() / RAND_MAX;
            T U2 = (T)rand() / RAND_MAX;
            
            v1 = 2 * U1 - 1;
            v2 = 2 * U2 - 1;
            s = v1 * v1 + v2 * v2;
        } while(s >= 1 || s == 0);
        
        x = v1 * sqrt(-2 * log(s) / s);
    } else
        x = v2 * sqrt(-2 * log(s) / s);
    
    phase = 1 - phase;
    return x;
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q);
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    cuda_init();
    DArray::ElapsedTimer timer;

    if(argc < 7){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(float)> <power refinement q(int)>\n");
        return 0;
    }

    char *filename = argv[1];
    long n = atol(argv[2]);
    long d = atol(argv[3]);
    int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/ 
    float gamma = strtod(argv[5], NULL);
    int qq = atoi(argv[6]);
    double C = strtod(argv[7], NULL);
    char *modelname = argv[8];

    float *x=(float*)malloc(sizeof(float)*d*n);
    float *y=(float*)malloc(sizeof(float)*n);
    for(int i=0; i<d; i++){
        for(int j=0; j<n; j++){
            x[i+j*d]=0.0;
        }
    }
    
    for(int i=0; i<n; ++i){
        y[i]=0.0;
    }

    timer.start();
    read_input_file<float>(filename, n, d, x, d, y);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: libsvm read takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    g.barrier();

    DArray::DMatrix<float> X(g, d, n);
    DArray::DMatrix<float> Y(g, 1, n);

    timer.start();
    X.set_value_from_matrix(x, d);
    Y.set_value_from_matrix(y, 1);
    ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: training and label data distribution across {} processes takes : {} (ms)\n",  g.ranks()[0], g.ranks()[1], np, ms);
    g.barrier();

    DArray::DMatrix<float> A(g, n, k);
    A.set_constant(0.0);
    {
        DArray::DMatrix<float> O(g, n, k);
        O.set_function([](int gi, int gj) ->float {
            return gaussrand<float>();
        });
        timer.start();
        DArray::LRA<float>(g, n, d, k, X, Y, O, A, gamma);
        ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: lra takes : {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    }
    // A.collect_and_print("A");
    g.barrier();

    // Distributed QR
    DArray::DMatrix<float> Q = A.clone();
    {
        timer.start();
        DArray::tall_skinny_qr_factorize(Q, 0, 0, n, qq*k);
        int ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    }
    // Q.collect_and_print("Q");
    g.barrier();


    // Distributed KQ
    DArray::DMatrix<float> KQ(g, n, qq*k);
    KQ.set_constant(0.0);
    {
        timer.start();
        DArray::LRA<float>(g, n, d, k, X, Y, Q, KQ, gamma);
        int ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: K*Q takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    }
    // KQ.collect_and_print("KQ");
    g.barrier();


    DArray::DMatrix<float> CC(g, qq*k, qq*k);
    CC.set_constant(0.0);
    {
        timer.start();
        auto Ql=Q.transpose_and_replicate_in_rows(0, 0, n , qq*k);
        auto KQl=KQ.replicate_in_columns( 0, 0, n, qq*k);
        auto Cl=CC.local_view(0, 0, qq*k, qq*k);
        // if(g.rank()==0) printf("Ql[%d,%d] %d, KQl[%d,%d] %d, Cl[%d, %d] %d \n", Ql.dims()[0], Ql.dims()[1], Ql.ld(), KQl.dims()[0], KQl.dims()[1], KQl.ld(), Cl.dims()[0], Cl.dims()[1], Cl.ld());
        tc_gemm(g.rank(), Cl.dims()[0], Cl.dims()[1], Ql.dims()[1], Ql.data(), Ql.ld(), KQl.data(), KQl.ld(), Cl.data(), Cl.ld(), 1.0f, 0.0f);
        int ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: C=QT*KQ takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    }
    g.barrier();
    // CC.collect_and_print("CC");

    DArray::DMatrix<float> U(g, k, k);
    U.set_constant(0.0);
    {
        timer.start();
        auto Ul = U.replicate_in_all(0, 0, k, k);
        auto Cl = CC.replicate_in_all(0, 0, k, k);
        auto Ud = DArray::svd<float>(g.rank(), Cl, 0, 0, k, k);

        for(int i=0; i<k; i++){
            for(int j=0; j<k; j++){
                if(i==j) Ul.data()[i+j*Ul.ld()] = Ud.data()[i];
                else Ul.data()[i+j*Ul.ld()] = 0.0;
            }
        }
        int ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: SVD takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
        U.dereplicate_in_all(Ul, 0, 0, k, k);
    }
    // U.collect_and_print("U");
    g.barrier();


    DArray::DMatrix<float> Z(g, n, k);
    Z.set_constant(0.0);
    {
        timer.start();
        auto Ql=Q.replicate_in_rows(0, 0, n, qq*k);
        auto Ul=U.replicate_in_columns(0, 0, qq*k, k);
        auto Zl=Z.local_view(0, 0, n, k);
        tc_gemm(g.rank(), Ql.dims()[0], Ul.dims()[1], Ql.dims()[1], Ql.data(), Ql.ld(), Ul.data(), Ul.ld(), Zl.data(), Zl.ld(), 1.0f, 0.0f);
        int ms = timer.elapsed();
        if(g.rank()==0) fmt::print("P[{},{}]: Creating Low Rank Approximator Z takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    }
    // Z.collect_and_print("Z");
    g.barrier();

    Knorm(g, n, d, k, X, Y, Z, gamma);


    auto ZZ = Z.replicate_in_all(0, 0, n, k);
    auto YY = Y.replicate_in_all(0, 0, 1, n);
    auto XX = X.replicate_in_all(0, 0, d, n);

    DArray::LMatrix<double> dZ(n, k);
    DArray::LMatrix<double> dY(n, 1);
    DArray::LMatrix<double> dX(d, n);
    dZ.float_to_double(ZZ.data(), ZZ.ld());
    dY.float_to_double(YY.data(), YY.ld());
    dX.float_to_double(XX.data(), XX.ld());

    // Initial guess of a
    DArray::LMatrix<double> a(n, 1);
    int plus=0, minus=0;
    for(int i=0; i<n; ++i){
        if(dY.data()[i] == 1) plus+=1;
        else if(dY.data()[i] == -1) minus+=1;
        else assert(false);
    }

    double mval=0, pval=0;
    if(plus>minus){
        mval=0.9*C;
        pval=mval*minus/plus;
    }else{
        pval=0.9*C;
        mval=pval*plus/minus;
    }

    for(int i=0; i<n; ++i){
        if(dY.data()[i] == 1) a.data()[i] = pval;
        else a.data()[i] = mval;
    }

    if(g.rank() == 0) printf("plus=%d, minus=%d, mval=%f, pval=%f\n", plus, minus, mval, pval);

    for(int i=0; i<n; ++i){
        if(a.data()[i] < 0 || a.data()[i] > C)
            if(g.rank() == 0) printf("Initial guess Inequality violation! offending: i=%d, a[i]=%f \n", i, a.data()[i]);
    }

    double Yta=0.0;
    for(int i=0; i<n; ++i){
        Yta+=(dY.data()[i]*a.data()[i]);
    }
    if(g.rank() == 0) printf("Initial feasibility:: dY'*a=%f\n", Yta);

    // Starting Barrier Method
    double t=1.0, alpha=0.1, beta=0.5;
    int mu=20, bi=1, one=1;
    int rk=g.rank();
    DArray::LMatrix<double> b(n,1), M(n, 1);
    b.set_Identity();
    for(int i=0; i<n; ++i){
        double val=0.0;
        for(int j=0; j<k; ++j){
            val+=(dZ.data()[i+j*dZ.ld()]* dZ.data()[i+j*dZ.ld()]);
        }
        M.data()[i]=val;
    }
    // if(g.rank() == 0){ M.save_to_file("M.csv"); }

    // Start barrier outer iteration
    while(true){
        if(g.rank() == 0) printf("Barrier iteration bi=%d, suboptimality gap (n/t)=%f, n=%d, t=%f, f(a)=%f\n", bi, (n/t), n, t, f(rk, n, k, a, dZ));
        int ni=1;

        // Start Newton inner iteration
        while(true){
            DArray::LMatrix<double> gradf(n,1);
            Grad_f(n, k, dZ, a, gradf);
            DArray::LMatrix<double> gradphi(n,1);
            Grad_Phi(n, a, C, t, gradphi);
            DArray::LMatrix<double> grad(n,1);
            Grad(n, gradf, gradphi, grad);
            DArray::LMatrix<double> D(n,1);
            Hess_Phi(n, a, C, t, D);
            int nn=n;
            int imax = idamax_(&nn, D.data(), &one);
		    int imin = idamin_(&nn, D.data(), &one);
            if( D.data()[imax]/D.data()[imin] > 1e18 ) {
			    if(g.rank() == 0) printf("D is too ill-conditioned %.3e! Terminating.\n", D[imax]/D[imin]);
			    break;
		    }       
            auto s = pre_conjgrad(rk, nn, k, dZ, M, D, b, a);

            double nd=0.0;
            for(int i=0; i<n; ++i){ 
                nd+=(-1.0*grad.data()[i])*s.data()[i];
            }
            nd = sqrt(nd);           
            if(g.rank() == 0) printf("Newton iteration ni::%d, nd::%f Decrement ::%f, g(a)::%f \n", ni, nd, (nd*nd/2), gg(rk, n, k, a, dZ, C, t));

            double lastg=gg(rk, n, k, a, dZ, C, t);
            double tt=1.0; //Newton Step Size
            while(true){
                DArray::LMatrix<double> gval(n, 1);
                for(int i=0; i<n; ++i){
                    gval.data()[i]=a.data()[i]+(tt*s.data()[i]);
                }
                double last_val=0.0;
                for(int i=0; i<n; ++i){
                    last_val+=(grad.data()[i]*s.data()[i]);
                }
                last_val*=(alpha*tt);
                if(gg(rk, n, k, gval, dZ, C, t) <= gg(rk, n, k, a, dZ, C, t)+last_val){
                    break;
                }
                tt=beta*tt;
            }

            if(g.rank() == 0) printf("Newton Step size::%f \n", tt);
            for(int i=0; i<n; ++i){
                a.data()[i] += (tt*s.data()[i]);
            }

            if(nd*nd/2 < 1.0e-6){
                if(g.rank() == 0) printf("Newton converged in ni::%d iterations; decrement ::%f; g(a)::%f\n", ni, (nd*nd/2), gg(rk, n, k, a, dZ, C, t));
                break;
            }


            if(ni>1){
                if((lastg-gg(rk, n, k, a, dZ, C, t)) < (1.0e-9*abs(gg(rk, n, k, a, dZ, C, t)))){
                    if(g.rank() == 0) printf("Newton slow progress! Decrement ::%f; g(a)::%f; f(a)::%f \n", (nd*nd/2), gg(rk, n, k, a, dZ, C, t), f(rk, n, k, a, dZ));
                    break;
                }
            }

            ni+=1;
        }
        // End Newton inner iteration


        if(2*n/t < abs(f(rk, n, k, a, dZ)) * 1e-5){ 
            //f(a) need to converge to 0.1% around f*
            if(g.rank() == 0) printf("Barrier converged in bi::%d iterations; suboptimality gap 2n/t::%f, n=%d, t=%f \n", bi, (2*n/t), n, t);
            break;
        }
        t=mu*t;
        bi+=1;
    }
    // End barrier outer iteration
    double dgamma=gamma;
    if(g.rank() == 0) DArray::writemodel(modelname, n, d, k, a, C, dZ, dY, dX, dgamma, minus, plus);

    cuda_finalize();
    MPI_Finalize();
    return 0;
}