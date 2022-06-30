#include "darray.h"
#include "blas_like.h"
#include "lapack_like.h"


void set_normal(int m, int n, float *A, float mean, float stddev) {
            std::mt19937 generator(1);
            std::normal_distribution<float> distribution(mean, stddev);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    A[i+j*m] = distribution(generator);
                }
            }
        }

extern void test_cuda();
extern void init();
extern void finalize();
extern void TCGEMM(int m, int n, int k, float alpha, float beta, float *hA, float *hB, float *hC);


int main(int argc, char** argv) 
{
    test_cuda();
    init();
    
    MPI_Init(&argc, &argv);
    int np, p, q;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    DArray::Grid::np_to_pq(np, p, q); // np = p*q
    DArray::Grid g(MPI_COMM_WORLD, p, q);
    if (g.rank()==0) printf("#### tsqr test grid pxq %dx%d #### \n", p, q);
    g.barrier();
    int m = 16;
    int n = 4;
    int b = 4;
    int qq = 1;
    DArray::LMatrix<float> Omega(n, b);
    DArray::DMatrix<float> A(g, m, n);
    DArray::DMatrix<float> K(g, m, b);
    DArray::DMatrix<float> QTA(g, n, n);
    A.set_normal_seed(0,1,10);
    auto Q = A.clone();
    
    //Omega.set_normal(0,1);
    set_normal(n,b,Omega.data(), 0,1);
    A.collect_and_print("A");
    //Q.collect_and_print("Q");
    
    if (A.grid().rank() == 0)
        Omega.print("Omega");

    float sone = 1.0;
    float szero = 0.0;

    if(qq==1)
    {
        //perform a A*B^T GEMM
        auto AL = A.replicate_in_rows(0,0,m,n);
        DArray::LMatrix<float> tmp(AL.dims()[0], b);
        if (A.grid().rank() == 0)
            AL.print("AL");
        //multiply_matrix("N", "N", 1.0f, 0.0f, AL, Omega, tmp);
        //host2device(AL.dims()[0], Omega.dims()[1], AL.dims()[1], AL.data(), Omega.data());
        TCGEMM(AL.dims()[0], Omega.dims()[1], AL.dims()[1], 1.0f, 0.0f, AL.data(), Omega.data(), tmp.data());
        if (A.grid().rank() == 0)
            tmp.print("tmp");
        A.dereplicate_in_rows(tmp,0,0,m,n);
        A.collect_and_print("A");
    }
    else
    {
        //perform a set of GEMMs, needs disscussion

    }
    auto R = tall_skinny_qr_factorize(Q, 0, 0, m, n);
    if (A.grid().rank() == 0) R.print("R");
    Q.collect_and_print("Q");

    //perform a GEMM Q^T*A
    {
        auto QT = Q.transpose_and_replicate_in_rows(0, 0, m, n);
        if(A.grid().rank() == 0) QT.print("QT");
        auto AL = A.replicate_in_columns(0,0,m,n);
        DArray::LMatrix<float> tmp(QT.dims()[0],QT.dims()[0]);
        //multiply_matrix("N","N",1.0f, 0.0f, QT, AL, tmp);
        TCGEMM(QT.dims()[0], AL.dims()[1], QT.dims()[1], 1.0f, 0.0f, QT.data(), AL.data(), tmp.data());
        if(A.grid().rank() == 0) tmp.print("tmp");
        for(int i = 0; i<tmp.dims()[0]; i++)
        {
            for(int j = 0; j<tmp.dims()[1];j++)
            {
                QTA.data()[i*tmp.dims()[0]+j] = tmp.data()[i*tmp.dims()[0]+j];
            }
        }
        QTA.collect_and_print("QTA");
    }

    auto QTAL = QTA.replicate_in_all(0,0,n,n);
    if(A.grid().rank() == 0) QTAL.print("QTAL");

    DArray::LMatrix<float> S(n,1);
    DArray::LMatrix<float> U(n,n);
    DArray::LMatrix<float> VT(n,n);

    int info, lwork = -1;
    float tmp;
    //perform a svd on Q^T*A
    sgesvd_("A", "A", &n, &n, QTAL.data(), &n, S.data(), U.data(), &n, VT.data(), &n, &tmp, &lwork, &info);

    lwork = tmp;

    DArray::LMatrix<float> work(lwork, 1);

    sgesvd_("A", "A", &n, &n, QTAL.data(), &n, S.data(), U.data(), &n, VT.data(), &n, work.data(), &lwork, &info);
    
    if(A.grid().rank() == 0)
    {
        U.print("U");
        S.print("S");
        VT.print("VT");
    }
    finalize();
    MPI_Finalize();
}