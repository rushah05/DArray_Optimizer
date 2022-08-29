cmake_minimum_required(VERSION 3.16)
project(DArray LANGUAGES "CXX")
set(CMAKE_CXX_STANDARD 17)
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDA REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(include)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70")

set (PROFILER_LIBRARIES -ldl)
set (COMMON_HEADERS include/lapack_like.h include/blas_like.h include/cblas_like.h include/utils.h)

add_executable(test1 tests/test1.cpp ${COMMON_HEADERS})
target_link_libraries(test1 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}  ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test2 tests/test2.cpp ${COMMON_HEADERS})
target_link_libraries(test2 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test3 tests/test3.cpp ${COMMON_HEADERS})
target_link_libraries(test3 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test4 tests/test4.cpp ${COMMON_HEADERS})
target_link_libraries(test4 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test5 tests/test5.cpp ${COMMON_HEADERS})
target_link_libraries(test5 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test6 tests/test6.cpp ${COMMON_HEADERS})
target_link_libraries(test6 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(testbarrier tests/test_barrier.cpp ${COMMON_HEADERS} include/read_input.h include/dbarrier.h)
target_link_libraries(testbarrier PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

set_source_files_properties(tests/test_cuda.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(include/darray_cuda.cu PROPERTIES LANGUAGE CUDA)

add_executable(testsvm tests/testsvm.cpp ${COMMON_HEADERS} include/read_input.h include/dkernel.h include/dbarrier.h)
target_link_libraries(testsvm PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES} -MT -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)

#add_executable(testsvm tests/testsvm.cpp ${COMMON_HEADERS} include/read_input.h include/darray_cuda.o)
#target_link_libraries(testsvm PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES} cudart cublas cusolver cusparse -MT -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)

add_executable(perftest_cholesky perftests/perftest_cholesky.cpp ${COMMON_HEADERS})
target_link_libraries(perftest_cholesky PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES} -Wl,--export-dynamic)

add_executable(perftest_lu perftests/perftest_lu.cpp ${COMMON_HEADERS})
target_link_libraries(perftest_lu PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})
