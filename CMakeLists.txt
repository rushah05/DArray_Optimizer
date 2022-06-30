cmake_minimum_required(VERSION 3.16)
project(DArray LANGUAGES "CXX")
set(CMAKE_CXX_STANDARD 17)
find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(include)


find_package(BLAS)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set (PROFILER_LIBRARIES -ldl)

add_executable(test1 tests/test1.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h include/utils.h)
target_link_libraries(test1 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}  ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test2 tests/test2.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(test2 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test3 tests/test3.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(test3 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test4 tests/test4.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(test4 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test5 tests/test5.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(test5 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(test6 tests/test6.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(test6 PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})

add_executable(perftest_cholesky perftests/perftest_cholesky.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(perftest_cholesky PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES} -Wl,--export-dynamic)

add_executable(perftest_lu perftests/perftest_lu.cpp include/lapack_like.h include/blas_like.h include/cblas_like.h)
target_link_libraries(perftest_lu PRIVATE ${MPI_C_LIBRARIES} ${BLAS_LIBRARIES}   ${MPI_LIBRARIES} ${PROFILER_LIBRARIES})