#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"

extern void cuda_init();
extern void cuda_finalize();
extern void LRA(int rank, int gn, int ln, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float gamma, float* Omega, int ldo);
extern void SGEQRF(int m, int n, float *Q, int ldq, int *info);
extern void SORMQR(int m, int n, float *Q, int ldq, float *RQ, int ldrq, int *info);

template<typename T>
DArray::LMatrix<T> QR(DArray::DMatrix<T> A, int i1, int j1, int i2, int j2) {
  // 1. replicate in rows: rows distributed, col replicated
  auto LA = A.replicate_in_rows(i1,j1,i2,j2);
  int m = LA.dims()[0], n = LA.dims()[1];
  assert( n == j2 - j1);
  assert( m > n);
  
  // 2. local QR (BLAS)
  int info=-1;
  SGEQRF( m, n, LA.data(), LA.ld(), &info);
  assert(info == 0);

  // 3. stack Rs into a big tall R, size (n*np) * n, e.g. 10,000 * 100
  // FIXME: chnage datatype according to typename T.
  int np = A.grid().dims()[0];
  std::vector<T> recv(np*n*n);
  {
      DArray::LMatrix<T> R(n, n);
      auto err = MPI_Allgather(R.data(), n * n, MPI_FLOAT, recv.data(), n * n, MPI_FLOAT, A.grid().comms()[0]);
      assert(err == MPI_SUCCESS);
  }

  DArray::LMatrix<T> Rs(np*n, n);
  int ldRs = Rs.ld();
  for(int b=0; b<np; b++) {
      T* recvblock = &recv.data()[n*n*b];
      for(int j=0; j<n; j++) {
          for (int i=0; i<n; i++)
              Rs.data()[b*n+i+j*ldRs] = (j<i)? 0 : recvblock[i + j*n];
      }
  }

  // 4. qr the stacked R
  DArray::LMatrix<T> Q(np*n, n);
  int ldq = Q.ld();
  for (int i=0; i<ldq; i++) {
      for (int j = 0; j < n; j++)
          Q.data()[i + j * ldq] = (i == j) ? 1.0 : 0;
  }
  {
    int m = np*n;
    int info=-1;
    SGEQRF( m, n, Rs.data(), Rs.ld(), &info);
    assert(info == 0);

    info=-1;
    SORMQR(m, n, Rs.data(), Rs.ld(), Q.data(), Q.ld(), &info);
    assert(info == 0);
  }

  // 5. post-multiply the Q of stacked R
  // LA Q has size m*n; R Q has size n*n
  // LA Q * R Q -> m*n
  // (I - 2*tau*H[1]*H[1]') m*m apply to n*n -> m*n
  DArray::LMatrix<T> RQ(m, n);
  auto pi = A.grid().ranks()[0];
  int ldRQ = RQ.ld(), ldQ = Q.ld();
  for(int j=0; j<n; j++) {
    for(int i=0; i<n; i++) {
        RQ.data()[i+j*ldRQ] = Q.data()[n*pi + i + j*ldQ];
    }
    for(int i=n; i<m; i++)
        RQ.data()[i+j*ldRQ] = 0;
  }

  info=0;
  SORMQR(m, n, LA.data(), LA.ld(), RQ.data(), ldRQ, &info);
  assert(info == 0);

  // 6. write Q back to distributed A
  auto grid = A.grid();
  // grid.print(RQ, "RQ");
  A.dereplicate_in_rows(RQ, i1, j1, i2, j2);

  DArray::LMatrix<T> R(n,n);
  DArray::copy_block(n, n, Rs.data(), Rs.ld(), R.data(), R.ld());
  return R;
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

  long long int gn=5000000;
	int d=18;
	int k=256;
	int qq=1;
	float gamma=0.0001f;
	char *filename="/project/pwu/dataset/SUSY";


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
  if(g.rank()==0) printf("Elemental Distribution Done \n");
  g.barrier();

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
    // if(g.rank()==0) printf("Xi[%d,%d], Xj[%d,%d], XjT[%d,%d], \nOmega_l[%d,%d] \nbuff[%d,%d] \n", Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1],
    // XjT.dims()[0], XjT.dims()[1], Omegal.dims()[0], Omegal.dims()[1], buff.dims()[0], buff.dims()[1]);

    int ln = Xi.dims()[0];
    int gd = Xi.dims()[1];
    int lk = Omegal.dims()[1];
    timer.start();
    LRA(g.rank(), gn, ln, gd, lk, Xi.data(), Xi.ld(), XjT.data(), XjT.ld(), Yi.data(), YjT.data(), buff.data(), buff.ld(), gamma, Omegal.data(), Omegal.ld());
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: LRA takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);
    KO.dereplicate_in_all(buff, 0, 0, buff.dims()[0], buff.dims()[1]);
  }

  // // KO.collect_and_print("KO");
  // auto fnorm = KO.fnorm();
  // if(g.rank()==0) fmt::print("norm(K)={}\n", fnorm);

  // Distributed QR
  DArray::DMatrix<float> Q = KO.clone();
  timer.start();
  DArray::LMatrix<float> R = QR(Q, 0, 0, gn, k);
  int ms = timer.elapsed();
  if(g.rank()==0) fmt::print("P[{},{}]: QR takes: {}(ms)\n",  g.ranks()[0], g.ranks()[1], ms);

  cuda_finalize();
  MPI_Finalize();
}



