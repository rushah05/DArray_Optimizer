#include "darray.h"
#include "darray_convex.h"
#include "read_input.h"

extern void cuda_init();
extern void cuda_finalize();
extern void lra(int rank, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float gamma, float* Omega, int ldo);

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();

  cuda_init();
  long long int gn=5000000;
	int d=18;
	int k=256;
	int qq=1;
	float gamma=0.0001f;
	char *filename="/project/pwu/dataset/SUSY";
  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, gn::%d, d::%d, k::%d)\n", p, q, gn, d, k);
  float *x=(float*)malloc(sizeof(float)*d*gn);
  float *y=(float*)malloc(sizeof(float)*gn);
  read_input_file(filename, gn, d, k, x, d, y);
  if(g.rank()==0) printf("Finished Reading input \n");
  g.barrier();

  DArray::DMatrix<float> X(g, d, gn);
  DArray::DMatrix<float> Y(g, 1, gn);
  DArray::LMatrix<float> Omega(X.ldims()[0], k);
  if(g.rank()==0) printf("X[%d,%d], Xlocal[%d,%d] \n", X.dims()[0], X.dims()[1], X.ldims()[0], X.ldims()[1]);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_vector(y);
  Omega.set_normal_seed(0, 1, 1);
  // X.collect_and_print("X");
  // Y.collect_and_print("Y");
  // Omega.collect_and_print("Omega");
  g.barrier();
  if(g.rank()==0) printf("Elemental Distribution Done \n");

  // printf("rank::%d, X[%d,%d], Y[%d,%d], Omega[%d,%d]\n", g.rank(), X.dims()[0], X.dims()[1], Y.dims()[0], Y.dims()[1], Omega.dims()[0], Omega.dims()[1]);
  DArray::LMatrix<float> Xi = X.transpose_and_replicate_in_rows(0, 0, X.dims()[0], X.dims()[1]);
  DArray::LMatrix<float> Xj = X.replicate_in_columns(0, 0, X.dims()[0], X.dims()[1]);
  DArray::LMatrix<float> Yi = Y.transpose_and_replicate_in_rows(0, 0, Y.dims()[0], Y.dims()[1]);
  DArray::LMatrix<float> Yj = Y.replicate_in_columns(0, 0, Y.dims()[0], Y.dims()[1]);
  
  if(g.rank()==0) {
    printf("Xi,Xj,Yi, and Yj  row and column replicated \n");
    // printf("X[%d,%d], Xi[%d,%d], Xj[%d,%d]\n", X.dims()[0], X.dims()[1], Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1]);
  }
  
  g.barrier();

  DArray::LMatrix<float> buff(Xi.dims()[1], k);
  // X is d*n
  lra(g.rank(), X.ldims()[1], X.ldims()[0], Omega.dims()[1], Xi.data(), Xi.dims()[0], Xj.data(), Xj.dims()[0], Yi.data(), Yj.data(), buff.data(), buff.dims()[0], gamma, Omega.data(), Omega.dims()[0]);
  if(g.rank()==0) printf("LRA done \n");

  cuda_finalize();
  MPI_Finalize();
}
