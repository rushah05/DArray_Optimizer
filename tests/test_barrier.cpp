#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"

extern void cuda_init();
extern void cuda_finalize();
extern void logbarrier(double* K, int ldk, int m, int n);

void rbf(int m, int n, double *Xi, double *Xj, double *Yi, double* Yj, double *Xij, int ldxij, double *K, int ldk, double gamma){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            K[i+j*ldk]=exp(-gamma*(Xi[i]-2*Xij[i+j*ldxij]+Xj[j]))*Yi[i]*Yj[j];
        }
    }
}

void vecnorm(double *Z, int ldz, double *Zn, int n, int d){
    double sum=0.0;
    for(int i=0; i<n; ++i){
        sum=0.0;
        for(int j=0; j<d; ++j){
            sum+=(Z[i+j*ldz]*Z[i+j*ldz]);
        }
        Zn[i]=sum;
    }
}


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();
  // cuda_init();
  DArray::ElapsedTimer timer;

  if(argc < 9){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(float)> <power refinement q(int)>\n");
      return 0;
  }

  char *filename = argv[1];
  // long long int n = atoi(argv[2]);
  int n = atoi(argv[2]);
  int d = atoi(argv[3]);
  int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
  int bs = atoi(argv[5]); 
  double gamma = strtod(argv[6], NULL);
  int qq = atoi(argv[7]);
  double C = strtod(argv[8], NULL);

  if(qq < 1) qq=1;

  if(bs <= 0) bs = 128;

  if(qq*k > n/2) {
    if(g.rank()==0) printf(" The kernel needs to be tall and skinny\n", p, q, n, d, k);
    return 0;
  }

  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, n::%d, d::%d, k::%d) \n", p, q, n, d, k);
  double *x=(double*)malloc(sizeof(double)*d*n);
  double *y=(double*)malloc(sizeof(double)*n);
  for(int i=0; i<d; i++){
    for(int j=0; j<n; j++){
      x[i+j*d]=0.0;
    }
  }
  for(int i=0; i<n; ++i){
    y[i]=0.0;
  }
  
  if(g.rank() == 0) {
    timer.start();
    read_input_file(g.rank(), filename, n, d, x, d, y);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  MPI_Bcast(x, d*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  g.barrier();

  DArray::DMatrix<double> X(g, d, n);
  DArray::DMatrix<double> Y(g, 1, n);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  free(x);
  free(y);
  g.barrier();

  
  DArray::DMatrix<double> K(g, n, n);
  {
    auto Xi=X.transpose_and_replicate_in_rows(0, 0, d, n);
    auto Yi=Y.transpose_and_replicate_in_rows(0, 0, 1, n);
    auto Xj=X.replicate_in_columns(0, 0, d, n).transpose();
    auto Yj=Y.replicate_in_columns(0, 0, 1, n).transpose();
    DArray::LMatrix<double> Xij(n, n);
    double alpha=1.0, beta=0.0;
    dgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
    DArray::LMatrix<double> Xisqr(n,1), Xjsqr(n,1);
    vecnorm(Xi.data(), Xi.ld(), Xisqr.data(), n, d);
    vecnorm(Xj.data(), Xj.ld(), Xjsqr.data(), n, d);
    auto Ki = K.replicate_in_all(0, 0, n, n);
    rbf(n, n, Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Ki.data(), Ki.ld(), gamma);
    K.dereplicate_in_all(Ki, 0, 0, n, n);
  }

  auto Kl=K.collect(0);
  if(g.rank() == 0){
    logbarrier(Kl.data(), Kl.ld(), n, n);
  } 
  


  MPI_Finalize();
 }