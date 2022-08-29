#include "darray.h"
#include "read_input.h"
#include "lapack_like.h"
#include "dbarrier.h"


int main(int argc, char** argv){
  MPI_Init(&argc, &argv);
  int np, p, q;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  DArray::Grid::np_to_pq(np, p, q); // np = p*q
  DArray::Grid g(MPI_COMM_WORLD, p, q);
  g.barrier();
  // cuda_init();
  DArray::ElapsedTimer timer;

  if(argc < 8){
      if(g.rank() == 0) printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(double)> <power refinement q(int)>\n");
      return 0;
  }

  char *filename = argv[1];
  long long int n = atoi(argv[2]);
  // int n = atoi(argv[2]);
  int d = atoi(argv[3]);
  int k = atoi(argv[4]); /*no of columns of the tall matrix A, B and Omega*/
  double gamma = strtod(argv[5], NULL);
  int qq = atoi(argv[6]);
  double C = strtod(argv[7], NULL);

  if(qq < 1){
      qq=1;
  }

  if(g.rank()==0) printf("#### SVM Kernel Training (Grid::%dx%d, n::%d, d::%d, k::%d) \n", p, q, n, d, k);
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
  
  if(g.rank() == 0) {
    timer.start();
    read_input_file(g.rank(), filename, n, d, x, d, y);
    int ms = timer.elapsed();
    if(g.rank()==0) fmt::print("P[{},{}]: Reading input from file takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);
  }
  MPI_Bcast(x, d*n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  g.barrier();

  DArray::DMatrix<float> X(g, d, n);
  DArray::DMatrix<float> Y(g, 1, n);
  X.set_value_from_matrix(x, d);
  Y.set_value_from_matrix(y, 1);
  free(x);
  free(y);
  g.barrier();

  DArray::DMatrix<float> A(g, n, k);
  {
    DArray::DMatrix<float> O(g, n, k);
    O.set_normal_seed(0, 1, 1);
    O.collect(0);
    if(g.rank() == 0) DArray::printMatrix("O.csv", n, k, O.data(), O.lld());
    DArray::RBF_Kernel<float>(g.rank(), np, n, d, k, X, Y, O, A, gamma);
  }


  MPI_Finalize();
}