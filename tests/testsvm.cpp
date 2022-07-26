#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"

extern void cuda_init();
extern void cuda_finalize();
// extern void lra(int rank, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* K, int ldk, float gamma);
extern void lra(int rank, int gn, int ln, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float gamma, float* Omega, int ldo);
extern void SGEQRF(int *m, int *n, float *Q, int *ldq, float *tau, float *work, int *lwork, int *info);

template<typename T>
DArray::LMatrix<T> QR(DArray::DMatrix<T> A, int i1, int j1, int i2, int j2) {
  // 1. replicate in rows: rows distributed, col replicated
  auto LA = A.replicate_in_rows(i1,j1,i2,j2);
  int m = LA.dims()[0], n = LA.dims()[1];
  assert( n == j2 - j1);
  assert( m > n);

  // 2. local QR (BLAS)
  std::vector<T> tau(std::min(m,n));
  int lwork = -1, info ;
  T tmp;
  SGEQRF( &m, &n, LA.data(), &LA.ld(), tau.data(), &tmp, &lwork, &info);
}

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();
  cuda_init();
  DArray::ElapsedTimer timer;

  long long int gn=49980;
	int d=22;
	int k=64;
	int qq=1;
	float gamma=0.0001f;
	char *filename="/project/pwu/dataset/ijcnn1";

  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, gn::%d, d::%d, k::%d) \n", p, q, gn, d, k);
  float *x=(float*)malloc(sizeof(float)*d*gn);
  float *y=(float*)malloc(sizeof(float)*gn);
  for(int i=0; i<d; i++){
    for(int j=0; j<gn; j++){
      x[i+j*d]=0.0;
    }
  }
  for(int i=0; i<gn; ++i){
    y[i]=0.0;
  }
  read_input_file(filename, gn, d, k, x, d, y);
  if(g.rank()==0) printf("Finished Reading input \n");
  g.barrier();

  DArray::DMatrix<float> X(g, d, gn);
  DArray::DMatrix<float> Y(g, 1, gn);
  DArray::DMatrix<float> Omega(g, gn, k);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  Omega.set_normal_seed(0, 1, 1);
  // X.collect_and_print("X");
  // Y.collect_and_print("Y");
  // Omega.collect_and_print("Omega");
  // if(g.rank()==0) printf("Elemental Distribution Done \n");
  // g.barrier();

  // Distributed Low-Rank Approximated Kernel
  DArray::DMatrix<float> KO(g, gn, k);
  KO.set_zeros();
  {
    DArray::LMatrix<float> Xi = X.transpose_and_replicate_in_rows(0, 0, d, gn);
    DArray::LMatrix<float> Xj = X.replicate_in_all(0, 0, d, gn);
    DArray::LMatrix<float> Yi = Y.transpose_and_replicate_in_rows(0, 0, 1, gn);
    DArray::LMatrix<float> Yj = Y.replicate_in_all(0, 0, 1, gn);
    DArray::LMatrix<float> XjT = Xj.transpose();
    DArray::LMatrix<float> YjT = Yj.transpose();

    g.barrier();
    // if(g.rank()==0) Xi.print("Xi");
    // if(g.rank()==0) Xj.print("Xj");
    // if(g.rank()==0) Yi.print("Yi");
    // if(g.rank()==0) Yj.print("Yj");
    

    DArray::LMatrix<float> buff = KO.replicate_in_rows(0, 0, gn, k);
    DArray::LMatrix<float> Omegal = Omega.replicate_in_columns(0, 0, gn, k);
    buff.set_zeros();
    // if(g.rank()==0) Omegal.print("Omegal");
    // if(g.rank()==0) buff.print("buff");
    if(g.rank()==0) printf("Xi[%d,%d], Xj[%d,%d], XjT[%d,%d], \nOmega_l[%d,%d] \nbuff[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1],
    XjT.dims()[0], XjT.dims()[1], Omegal.dims()[0], Omegal.dims()[1], buff.dims()[0], buff.dims()[1]);

    int ln = Xi.dims()[0];
    int gd = Xi.dims()[1];
    int lk = Omegal.dims()[1];
    timer.start();
    lra(g.rank(), gn, ln, gd, lk, Xi.data(), Xi.dims()[0], XjT.data(), XjT.dims()[0], Yi.data(), YjT.data(), buff.data(), buff.dims()[0], gamma, Omegal.data(), Omegal.dims()[0]);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    KO.dereplicate_in_all(buff, 0, 0, buff.dims()[0], buff.dims()[1]);
  }

  // KO.collect_and_print("KO");
  auto fnorm = KO.fnorm();
  if(g.rank()==0) fmt::print("norm(K)={}\n", fnorm);

  // Distributed QR
  DArray::DMatrix<float> Q = KO.clone();
  DArray::LMatrix<float> R = QR(Q, 0, 0, gn, k);
  if(g.rank()==0) printf("Finished QR\n");


  cuda_finalize();
  MPI_Finalize();
}



