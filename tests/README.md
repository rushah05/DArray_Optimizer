export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/apps/cuda-11.6/lib64/"
module load GCC intel
module load cmake
module load cudatoolkit/11.6
cmake ..
cmake --build .



module load GCC intel
module load cmake
module load cudatoolkit/11.6
nvcc -c darray_cuda.cu -o darray_cuda.o -gencode arch=compute_70,code=sm_70 















echo "*********************************"
echo "*********************************"
echo "COVTYPE n=580864 d=54 k=160 q=1 (N 7 n 324)"
time mpirun -np 324 ./testbarrier "/project/pwu/dataset/covtype" 580864 54 128 6480 0.01 1 100


# echo "*********************************"
# echo "*********************************"
# echo "IJCNN1 n=49990 d=22 k=256 q=1 (N 1 n 16)"
# time mpirun -np 16 ./testbarrier "/project/pwu/dataset/ijcnn1" 49990 22 256 8192 0.05 1 100













# # # # echo "*********************************"
# # # # echo "*********************************"
# # echo "ijcnn1 n=49990 d=22 k=256 qq=1 (N 1 np 4)"
# # time mpirun -np 4 ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 256 0.05 1 100

# # # # echo "*********************************"
# # # # echo "*********************************"
# # echo "ijcnn1 n=49990 d=22 k=256 qq=1 (N 1 np 9)"
# # time mpirun -np 9 ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 256 0.05 1 100

# # # # echo "*********************************"
# # # # echo "*********************************"
# # echo "ijcnn1 n=49990 d=22 k=256 qq=1 (N 1 np 16)"
# # time mpirun -np 16 ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 256 0.05 1 100

# # # echo "*********************************"
# # # echo "*********************************"
# echo "ijcnn1 n=49990 d=22 k=256 qq=1 (N 1 np 25)"
# time mpirun -np 25 ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 256 0.05 1 100

# echo "*********************************"
# echo "*********************************"
# echo "covtype n=580864 d=54 k=128 qq=2 (N 2 np 64)"
# time mpirun -np 64 ./testsvm "/project/pwu/dataset/covtype" 580864 54 128 0.01 1 100


# # echo "SUSY n=5000000 d=18 k=250 qq=2 (N 6 np 256)"
# # time mpirun -np 256 ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 256 0.05 1 100









# # echo "================================================================================================="
# # echo "COVTYPE n=580864 d=54 k=128 qq=2"
# # echo "================================================================================================="

# # echo "*********************************"
# # echo "*********************************"
# # echo "N 1 n 1"
# # time mpirun -np 1 ./testsvm "/project/pwu/dataset/covtype" 580864 54 128 0.01 2 100

# # echo "*********************************"
# # echo "*********************************"
# # echo "N 1 n 4"
# # time mpirun -np 4 ./testsvm "/project/pwu/dataset/covtype" 580864 54 128 0.01 2 100

# # echo "*********************************"
# # echo "*********************************"
# # echo "N 1 n 16"
# # time mpirun -np 16 ./testsvm "/project/pwu/dataset/covtype" 580864 54 128 0.01 2 100


# # # echo "================================================================================================="
# # # echo "SUSY n=5000000 d=18 k=250 qq=2 "
# # # echo "================================================================================================="
# # # echo "N 1 n 1"
# # # time mpirun -np 1 ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 250 0.05 2 100

# # # echo "*********************************"
# # # echo "*********************************"
# # # echo "N 1 n 4"
# # # time mpirun -np 4 ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 250 0.05 2 100


# # # echo "*********************************"
# # # echo "*********************************"
# # # echo "N 1 n 16"
# # # time mpirun -np 16 ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 250 0.05 2 100

# # # echo "*********************************"
# # # echo "*********************************"
# # # echo "N 1 n 25"
# # # time mpirun -np 25 ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 250 0.05 2 100





# # echo "================================================================================================="
# # echo "HIGGS n=11000000 d=28 k=500 qq=2"
# # echo "================================================================================================="


# # # echo "*********************************"
# # # echo "*********************************"
# # # echo "N 1 n 25"
# # time mpirun -np 25 ./testsvm "/project/pwu/dataset/HIGGS" 11000000 28 500 0.05 2 100
















// template<typename T>
// void LRA(int rk, int n, int d, int k, T *X, T *Y, T *O, T *A, T gamma, int bs=8192){
//     T alpha = 1, beta = 0, beta2 = 0;
    
// }


// void LRA(int rk, int n, int d, int k, float *X, int ldx, float *Y, float *O, int ldo, float *A, int lda, float gamma, int bs=8192){
//     float sone = 1, szero = 0, beta2 = 0;
//     float *dX, *dY, *dXij, *dK, *dXisqr, *dXjsqr, *dO;
//     // __half *hX, *hY;
//     gpuErrchk(cudaMalloc(&dX, sizeof(float)*n*d));
//     gpuErrchk(cudaMalloc(&dY, sizeof(float)*n));
//     gpuErrchk(cudaMalloc(&dO, sizeof(float)*n*k));
//     gpuErrchk(cudaMalloc(&dXisqr, sizeof(float)*bs));
//     gpuErrchk(cudaMalloc(&dXjsqr, sizeof(float)*bs));
//     gpuErrchk(cudaMalloc(&dXij, sizeof(float)*bs*bs));
//     gpuErrchk(cudaMalloc(&dK, sizeof(float)*bs*bs));
//     gpuErrchk(cudaMalloc(&dA, sizeof(float)*n*k));
// 	gpuErrchk(cudaMemcpy(dX, X, sizeof(float)*n*d, cudaMemcpyHostToDevice));
// 	gpuErrchk(cudaMemcpy(dY, Y, sizeof(float)*n, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaMemcpy(dO, O, sizeof(float)*n*k, cudaMemcpyHostToDevice));
// 	// hX = half_mem;
// 	// hY = half_mem+sizeof(__half)*n;

//     // printMatrixDeviceBlock("X.csv", n, d, dX, n);
//     // printMatrixDeviceBlock("Y.csv", n, 1, dY, n);

//     for(int i=0; i<n; i+=bs){
//         int ib = std::min(bs, n-i);
//         vecnorm<<<(ib+63)/64, 64>>>(&dX[i], ldx, dXisqr, ib, d);
//         for(int j=0; j<n; j+=bs){
//             int jb = std::min(bs, n-j);
//             if(j > 1) beta2 = 1;
//             vecnorm<<<(jb+63)/64, 64>>>(&dX[j], ldx, dXisqr, jb, d);
//             // printf("ib::%d, jb::%d ldx::%d &dX[i]::%f, &dX[j]::%f, dXij::%f \n", ib, jb, ldx, &dX[i], &dX[j], &dXij[0]);
//             stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, ib, jb, d, &sone, &dX[i], CUDA_R_32F, ldx,
//                             &dX[j], CUDA_R_32F, ldx, &szero, dXij, CUDA_R_32F, ib,
//                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
//             if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
//             dim3 numBlocks((ib+31)/32, (jb+31)/32);
//             dim3 threadsPerBlock(32,32);
//             rbf<<<numBlocks, threadsPerBlock>>>(ib, jb, dK, ib, dXisqr, dXjsqr, dXij, ib, gamma, &dY[i], &dY[j]);
//             // if(rk==0 && i==0)printMatrixDeviceBlock("Kij0"+std::to_string(j)+".csv", ib, jb, dK, ib);

//             stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ib, k, jb, &sone, dK, CUDA_R_32F, ib,
//                             &dO[j], CUDA_R_32F, ldo, &beta2, &dA[i], CUDA_R_32F, lda,
//                             CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);	
//             if (stat != CUBLAS_STATUS_SUCCESS) printf ("cublasGemmEx failed %s\n", __LINE__);
//         }
//     }
//     gpuErrchk(cudaMemcpy(A, dA, sizeof(float)*n*k, cudaMemcpyDeviceToHost));
//     // if(rk == 0) printMatrixDeviceBlock("A.csv", n, k, dA, lda);
//     cudaDeviceSynchronize();
//     cudaFree(dX);
//     cudaFree(dY);
//     cudaFree(dXij);
//     cudaFree(dK);
//     cudaFree(dO);
//     cudaFree(dA);
//     cudaFree(dXisqr);
//     cudaFree(dXjsqr);
// }


// void host2device(int n, int k, int d, float *hA, float *hB){
// 	dA = mem;
// 	dB = mem+sizeof(float)*d*k;
// 	dC = mem+sizeof(float)*n*d+sizeof(float)*d*k;
// 	cudaMemcpy(dA, hA, sizeof(float)*n*d, cudaMemcpyHostToDevice);
// 	cudaMemcpy(dB, hB, sizeof(float)*d*k, cudaMemcpyHostToDevice);
// 	halfA = half_mem;
// 	halfB = half_mem+sizeof(__half)*d*k;
// }

// void device2host(int n, int k, float *hC){
// 	cudaMemcpy(hC, dC, sizeof(float)*n*k, cudaMemcpyDeviceToHost);
// }



    // template<typename T>
    // void Knorm(int rk, int np, Grid g, int n, int d, int k, DMatrix<T> &X, DMatrix<T> &Y, DMatrix<T> &Z, T gamma, int bs=8192){
    //     T alpha = 1, beta = 0, beta2 = 0, minus=-1;
    //     T fnK = 0, fnKAtA = 0;
     
    //     DMatrix<T> K(g, bs, bs);
    //     K.set_constant(0.0);
    //     DMatrixDescriptor<T> Kdesc{K, 0, 0, bs, bs};
    //     DMatrixDescriptor<T> Zdesc{Z, 0, 0, bs, k};

    //     for(int i=0; i<n; i+=bs){
    //         int ib = std::min(bs, n-i);
    //         auto Xi=X.transpose_and_replicate_in_rows(0, i, d, i+ib);
    //         auto Yi=Y.transpose_and_replicate_in_rows(0, i, 1, i+ib);
    //         LMatrix<T> Xisqr(Xi.dims()[0], 1);
    //         vecnorm<T>(Xi.data(), Xi.ld(), Xisqr.data(), Xi.dims()[0], d);
    //         auto Zi=Zdesc.matrix.replicate_in_rows(Zdesc.i1, Zdesc.j1, Zdesc.i1+ib, Zdesc.j2);

    //         for(int j=0; j<n; j+=bs){
    //             int jb = std::min(bs, n-j);
    //             if(j>0) beta2=1; 

    //             auto Xj=X.replicate_in_columns(0, j, d, j+jb).transpose();
    //             auto Yj=Y.replicate_in_columns(0, j, 1, j+jb).transpose();
    //             LMatrix<T> Xjsqr(Xj.dims()[0], 1);
    //             vecnorm<T>(Xj.data(), Xj.ld(), Xjsqr.data(), Xj.dims()[0], d);
                
    //             LMatrix<T> Xij(Xi.dims()[0], Xj.dims()[0]);
    //             // sgemm_("N", "T", &Xi.dims()[0], &Xj.dims()[0], &Xi.dims()[1], &alpha, Xi.data(), &Xi.ld(), Xj.data(), &Xj.ld(), &beta, Xij.data(), &Xij.ld());
    //             TC_GEMM(rk, 'N', 'T', Xi.dims()[0], Xj.dims()[0], Xi.dims()[1], alpha, beta,  Xi.data(), Xi.ld(), Xj.data(), Xj.ld(), Xij.data(), Xij.ld());
    //             auto Kij=Kdesc.matrix.local_view(Kdesc.i1, Kdesc.j1, Kdesc.i1+ib, Kdesc.j1+jb);
    //             // rbf<T>(Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
    //             Kernel(rk, Xi.dims()[0], Xj.dims()[0], Xisqr.data(), Xjsqr.data(), Yi.data(), Yj.data(), Xij.data(), Xij.ld(), Kij.data(), Kij.dims()[0],  gamma);
    //             // if(rk==0 && i==0) Kij.save_to_file("Kij0"+std::to_string(j)+".csv");
    //             fnK+=K.fnorm();
                
    //             auto Zj=Zdesc.matrix.replicate_in_rows(Zdesc.i1, Zdesc.j1, Zdesc.i1+jb, Zdesc.j2).transpose();
    //             // if(rk == 0) printf("Zi[%d,%d], Zj[%d,%d], Kij[%d,%d]\n", Zi.dims()[0], Zi.dims()[1], Zj.dims()[0], Zj.dims()[1], Kij.dims()[0], Kij.dims()[1]); 
    //             // sgemm_("N", "T", &Zi.dims()[0], &Zj.dims()[0], &Zi.dims()[1], &alpha, Zi.data(), &Zi.ld(), Zj.data(), &Zj.ld(), &minus, Kij.data(), &Kij.ld()); 
    //             TC_GEMM(rk, 'N', 'T', Zi.dims()[0], Zj.dims()[0], Zi.dims()[1], alpha, minus, Zi.data(), Zi.ld(), Zj.data(), Zj.ld(), Kij.data(), Kij.ld());
    //             fnKAtA+=K.fnorm();
    //         }
    //     }
    //     if(rk==0) fmt::print("fnKAtA::{}, fnK::{}, fnorm(K-AtA)/fnorm(K)={}\n", fnKAtA, fnK, fnKAtA/fnK);
    // }


    // <template<typename T>








//   DArray::DMatrix<float> A(g, n, k);
//   A.set_constant(0.0);
//   {
//     DArray::DMatrix<float> O(g, n, k);
//     for(int i=0; i<n; i++){
//       for(int j=0; j<k; j++){
//         O.at(i,j)=gaussrand<float>();
//       }
//     } 

//     DArray::LRA<float>(g.rank(), np, g, n, d, k, X, Y, O, A, gamma, bs);
//   }


// //   // PHASE 2
// //   timer.start();
// //   auto Yl=Y.replicate_in_all(0, 0, 1, n).transpose();
// //   auto Al=A.collect(0);

// //   if(g.rank() == 0){
// //     DArray::LMatrix<double> Ad(g, n, k);
// //     for(int i=0; i<n; ++i){
// //       for(int j=0; j<k; ++j){
// //         Ad.at(i,j)=Al.at(i,j);
// //       }
// //     }

// //     DArray::DMatrix<double> Yd(g, 1, n);{
// //       for(int i=0; i<1; ++i){
// //         for(int j=0; j<n; ++j){
// //           Yd.at(i,j)=Yl.at(i,j);
// //         }
// //       }
// //     }

// //     int plus=0, minus=0, t_plus=0, t_minus=0;
// //     double mval=0, pval=0, t_mval=0, t_pval=0;
// //     for(int i=0; i<yl.dims()[0]; ++i){
// //       if(yl[i]==1.0) plus+=1; 
// //       else minus+=1.0;
// //     }
    
// //     if(plus>minus){
// //       mval=0.9*C;
// //       pval=mval*minus/plus;
// //     }else{
// //       pval=0.9*C;
// //       mval=pval*plus/minus;
// //     }
// //     if(g.rank() == 0) fmt::print("plus:{}, minus:{}, pval:{} mval:{} \n", plus, minus, pval, mval);
    
// //   }

// //   // DArray::DMatrix<double> Ad(g, n, k);{
// //   //   for(int i=0; i<n; ++i){
// //   //     for(int j=0; j<k; ++j){
// //   //       Ad.at(i,j)=A.at(i,j);
// //   //     }
// //   //   }
// //   // }

// //   // // initial guess of a (all zeros??)
// //   // int plus=0, minus=0, t_plus=0, t_minus=0;
// //   // double mval=0, pval=0, t_mval=0, t_pval=0;
// //   // auto yl=Y.replicate_in_all(0, 0, 1, n).transpose();
// //   // {
// //   //   for(int i=0; i<yl.dims()[0]; ++i){
// //   //     if(yl[i]==1.0) plus+=1; 
// //   //     else minus+=1.0;
// //   //   }
    
// //   //   if(plus>minus){
// //   //     mval=0.9*C;
// //   //     pval=mval*minus/plus;
// //   //   }else{
// //   //     pval=0.9*C;
// //   //     mval=pval*plus/minus;
// //   //   }
// //   //   if(g.rank() == 0) fmt::print("plus:{}, minus:{}, pval:{} mval:{} \n", plus, minus, pval, mval);
// //   // }

// //   DArray::DMatrix<double> a(g, n, 1);
// //   auto al=a.replicate_in_all(0, 0, n, 1);
// //   // DArray::DMatrixDescriptor<double> adesc{a, 0, 0, n, 1};
// //   {
// //     // auto al=adesc.matrix.replicate_in_all(adesc.i1, adesc.j1, adesc.i2, adesc.j2);
// //     for(int i=0; i<al.dims()[0]; ++i){
// //       if(yl[i]==1.0) al.data()[i]=pval; 
// //       else if(yl[i]==-1.0) al.data()[i]=mval;
// //       else {
// //         assert(false);
// //         return 0;
// //       }
// //     }
// //     a.dereplicate_in_all(al, 0, 0, n, 1);
// //   }
// //   // a.collect_and_print("a");
// //   g.barrier();

// //   //  feasibility:
// //   for(int i=0; i<n; ++i){
// //       if (al[i]<0 || al[i]>C) fmt::print("Initial guess Inequality violation! offending a[i]: i={}, a[i]={} \n",i, al[i]);
// //   }
  
// //   double fone=1.0f, fzero=0.0f, Ydta=0.0f;
// //   int one=1;
// //   DArray::DMatrix<double> Yd(g, 1, n);{
// //     for(int i=0; i<1; ++i){
// //       for(int j=0; j<n; ++j){
// //         Yd.at(i,j)=Y.at(i,j);
// //       }
// //     }
// //   }
// //   auto ydl=Yd.replicate_in_all(0, 0, 1, n).transpose();
// //   // Yd.collect_and_print("Yd");
// //   dgemm_("T", "N", &one, &one, &n, &fone, ydl.data(), &ydl.ld(), al.data(), &al.ld(), &fzero, &Ydta, &one);  
// //   if(g.rank() == 0) fmt::print("Initial feasibility: Yd'*a={}\n",Ydta);

// //   // Initial values 
// //   int t=1, mu=20, bi=1;
// //   double alpha=0.1, beta=0.5;
// //   auto Al=Ad.replicate_in_all(0, 0, n, k);
// //   double done=1, dzero=0;
// //   // auto fval = DArray::f<double>(g.rank(), n, k, al, Al);

// //   while(true){
// //     if(g.rank()==0) fmt::print("Barrier iteration bi::{}; suboptimality gap n/t={}, n={}, t={}, f(a)={} \n",bi, (n/t), n, t, DArray::f<double>(g.rank(), n, k, al, Al));
// //     int ni=1;  // newton iteration

// //     // while(true){
// //     for(int i=0; i<10; ++i){

// //       //Gradf
// //       DArray::LMatrix<double> Gta(k, 1);
// //       dgemm_("T", "N", &k, &one, &n, &done, Al.data(), &Al.ld(), al.data(), &al.ld(), &dzero, Gta.data(), &Gta.ld()); 
// //       DArray::LMatrix<double> Gradf(n, 1);   
// //       dgemm_("N", "N", &n, &one, &k, &done, Al.data(), &Al.ld(), Gta.data(), &Gta.ld(), &dzero, Gradf.data(), &Gradf.ld()); 
// //       for(int i=0; i<n; ++i){
// //         Gradf.data()[i]=Gradf.data()[i]-1.0;
// //       }
      
// //       //GradPhi
// //       DArray::LMatrix<double> GradPhi(n, 1); 
// //       for(int i=0; i<n; ++i){
// //         GradPhi.data()[i] = ((-1.0/al.data()[i]) + (1.0/(C-al.data()[i])))/t;
// //       }


// //       // Grad
// //       DArray::LMatrix<double> Grad(n, 1); 
// //       for(int i=0; i<n; ++i){
// //         Grad.data()[i] = Gradf.data()[i]+GradPhi.data()[i];
// //       }

// //       // Hessian
// //       DArray::LMatrix<double> D(n, 1); 
// //       for(int i=0; i<n; ++i){
// //         D.data()[i] = (1.0/(al.data()[i]*al.data()[i]))+(1.0/((C-al.data()[i])*(C-al.data()[i])));
// //       }

// //       DArray::LMatrix<double> s(n, 1); 
// //       {
        
// //       }



// //     }

// //     if ((2*n/t) < (abs(DArray::f<double>(g.rank(), n, k, al, Al)) * 1.0e-5)) { // f(a) need to converge to 0.1% around f*
// //       if(g.rank()==0) fmt::print("Barrier converged in {} iterations; suboptimality gap 2n/t={}, n={}, t={} \n", bi, 2*n/t, n, t);
// //       break;
// //     }
// //     t=mu*t;
// //     bi+=1;
// //   }
// //   int ms = timer.elapsed();
// //   if(g.rank()==0) fmt::print("P[{},{}]: Phase 2 takes: {} (ms)\n",  g.ranks()[0], g.ranks()[1], ms);


//