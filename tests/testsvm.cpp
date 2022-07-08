#include "darray.h"
#include "darray_convex.h"
#include "read_input.h"

extern void cuda_init();
extern void cuda_finalize();
extern void lra(int m, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float* Omega, int ldo);

int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  if(g.rank()==0) printf("#### test grid pxq %dx%d #### \n", p, q);
  g.barrier();

  if(g.rank()==0) cuda_init();
  long long int gn=680;
	int d=10;
	int k=4;
	int qq=1;
	float gamma=0.0001f;
	char *filename="/project/pwu/dataset/breast-cancer_scale";
  float *x=(float*)malloc(sizeof(float)*d*gn);
  int ldx=d;
  float *y=(float*)malloc(sizeof(float)*1*gn);
  read_input_file(filename, gn, d, k, x, ldx, y);
  g.barrier();

  DArray::DMatrix<float> X(g, d, gn);
  DArray::DMatrix<float> Y(g, 1, gn);
  DArray::DMatrix<float> Omega(g, gn, k);
  DArray::DMatrix<float> KOmega(g, gn, k);
  X.set_value_function(x,ldx);
  // X.collect_and_print("X");
  Y.set_value_function(y,1);
  // Y.collect_and_print("Y");
  Omega.set_normal_seed(0, 1, 1);
  // Omega.collect_and_print("Omega");
  g.barrier();

  // printf("rank::%d, X[%d,%d], Y[%d,%d], Omega[%d,%d]\n", g.rank(), X.dims()[0], X.dims()[1], Y.dims()[0], Y.dims()[1], Omega.dims()[0], Omega.dims()[1]);
  DArray::LMatrix<float> Xi = X.transpose_and_replicate_in_rows(0, 0, X.dims()[0], X.dims()[1]);
  DArray::LMatrix<float> Xj = X.replicate_in_columns(0, 0, X.dims()[0], X.dims()[1]);
  DArray::LMatrix<float> Yi = Y.transpose_and_replicate_in_rows(0, 0, Y.dims()[0], Y.dims()[1]);
  DArray::LMatrix<float> Yj = Y.replicate_in_columns(0, 0, Y.dims()[0], Y.dims()[1]);
  printf("X[%d,%d], Xi[%d,%d], Xj[%d,%d]\n", X.dims()[0], X.dims()[1], Xi.dims()[0], Xi.dims()[1], Xj.dims()[0], Xj.dims()[1]);
  g.barrier();

  lra(X.dims()[1], X.dims()[1], X.dims()[0], Omega.dims()[1], Xi.data(), Xi.dims()[0], Xj.data(), Xj.dims()[0], Yi.data(), Yj.data(), KOmega.data(), KOmega.dims()[0], Omega.data(), Omega.dims()[0]);
      // (int m, int n, int d, int k, float* Xi, int ldxi, float* Xj, int ldxj, float* Yi, float* Yj, float* KOmega, int ldk, float* Omega, int ldo)

  if(g.rank()==0) cuda_finalize();
  MPI_Finalize();
}
