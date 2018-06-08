/* Block-Sparse-Matrix-Lib, version 1.0. A block sparse matrix library.
 * Copyright (C) Emanuel H. Rubensson <emanuelrubensson@gmail.com>,
 *               Elias Rudberg <eliasrudberg@gmail.com>, and
 *               Anastasia Kruchinina <anastasia.kruchinina@it.uu.se>.
 * 
 * Distribution without copyright owners' explicit consent prohibited.
 * 
 * This source code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <iostream>
#include <cmath>
#include "hierarchical_block_sparse_lib.h"
#include "test_utils.h"
#include <sys/time.h>

static double get_wall_seconds() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) != 0)
    throw std::runtime_error("Error in cht::get_wall_seconds(), in gettimeofday().");
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

template<typename MatrixType>
static int test_creation(int N, int blockSize, double nonzeroFraction, int nMat, int nRep) {
    typename MatrixType::Params param;
    param.blocksize = blockSize;
    MatrixType M1;

    M1.set_params(param);
    if(!M1.empty())
    throw std::runtime_error("Error: M.empty() gave wrong result.");
    M1.resize(3, 7,3,7);
    
    if(M1.empty())
    throw std::runtime_error("Error: M.empty() gave wrong result.");
    M1.clear();
    if(!M1.empty())
    throw std::runtime_error("Error: M.empty() gave wrong result.");
    M1.resize(3, 7,3,7);

    if(M1.get_frob_squared() != 0)
    throw std::runtime_error("Error: M.get_frob_squared() != 0 for newly created matrix.");

    double refValue1 = 7.7;
    double refValue2 = 1.1;

    int NN = N / blockSize;

    // Allocate nMat matrices and do some operations with them.
    std::vector<MatrixType*> M_vec_1(nMat);
    for(int i = 0; i < nMat; i++) {
        
        M_vec_1[i] = new MatrixType();
        M_vec_1[i]->set_params(param);
        M_vec_1[i]->resize(N, N,N,N);
        
        // Test assign_from_vectors()
        {
            int nValues = (int)((double)NN*NN*nonzeroFraction);
            std::vector<int> rows(nValues);
            std::vector<int> cols(nValues);
            std::vector<double> values(nValues);
            int count = 0;
            for(int ii = 0; ii < NN; ii++) {
                for(int jj = 0; jj < NN; jj++) {
                    rows[count] = ii*blockSize;
                    cols[count] = jj*blockSize;
                    values[count] = 1.1 + 0.001*ii;
                    count++;
                    if(count == nValues)
                    break;
                }
                
                if(count == nValues)
                break;
            }
            
            assert(count == nValues);
            
            M_vec_1[i]->assign_from_vectors(rows, cols, values);
            
            MatrixType C;
            
            C.copy(*M_vec_1[i]);
            verify_that_matrices_are_equal(C, *M_vec_1[i]);
            
        }
    }
    
    // Now delete odd-numbered matrices (trying to create a fragmented memory situation).
    for(int i = 1; i < nMat; i+=2)
        delete M_vec_1[i];
    
    int nMatHalf = nMat / 2;
    
    // Now repeatedly create nMatHalf matrices and do some operations on them.
    for(int i = 0; i < nRep; i++) {

        std::vector<MatrixType*> M_vec_2(nMatHalf);
        for(int i = 0; i < nMatHalf; i++) {

            M_vec_2[i] = new MatrixType();
            M_vec_2[i]->set_params(param);
            M_vec_2[i]->resize(N, N,N,N);

              // Test assign_from_vectors()
            {
                int nValues = (int)((double)NN*NN*nonzeroFraction);
                std::vector<int> rows(nValues);
                std::vector<int> cols(nValues);
                std::vector<double> values(nValues);
                int count = 0;
                for(int ii = 0; ii < NN; ii++) {
                  for(int jj = 0; jj < NN; jj++) {
                    rows[count] = ii*blockSize;
                    cols[count] = jj*blockSize;
                    values[count] = 1.1 + 0.001*ii;
                    count++;
                    if(count == nValues)
                      break;
                  }
                  if(count == nValues)
                    break;
                }   
                assert(count == nValues);
                M_vec_2[i]->assign_from_vectors(rows, cols, values);
            }

            // Test copy()
            MatrixType C;
            C.copy(*M_vec_2[i]);
            verify_that_matrices_are_equal(C, *M_vec_2[i]);

        }

        for(int i = 0; i < nMatHalf; i++)
          delete M_vec_2[i];
    }
    
    // Now delete even-numbered-numbered matrices (trying to create a fragmented memory situation).
    for(int i = 0; i < nMat; i+=2)
        delete M_vec_1[i];

    std::cout << "Matrix test doing many allocations finished OK." << std::endl;

    return 0;
}


int main(int argc, const char* argv[]) {
  if(argc != 6) {
    std::cerr << "Please give 5 args: N blockSize nMat nRep nonzeroFraction" << std::endl;
    return -1;
  }
  int N = atoi(argv[1]);
  int blockSize = atoi(argv[2]);
  int nMat = atoi(argv[3]);
  int nRep = atoi(argv[4]);
  double nonzeroFraction = atof(argv[5]);
  std::cout << "N = " << N << std::endl;
  std::cout << "blockSize = " << blockSize << std::endl;
  std::cout << "nMat = " << nMat << std::endl;
  std::cout << "nRep = " << nRep << std::endl;
  std::cout << "nonzeroFraction = " << nonzeroFraction << std::endl;
  if(N < 7) {
    std::cerr << "Error: N must be 7 or greater." << std::endl;
    return -1;
  }
  if(nonzeroFraction < 0 || nonzeroFraction > 1) {
    std::cerr << "Error: nonzeroFraction must be between 0 and 1." << std::endl;
    return -1;
  }
  double nElementsPerMatrix = N*N*nonzeroFraction;
  double nBytesPerMatrix = nElementsPerMatrix * sizeof(double);
  double nBytesForAllMatrices = nBytesPerMatrix * nMat;
  double memUsageForAllMatricesInGB = nBytesForAllMatrices / ( 1000*1000*1000 );
  std::cout << "Expected mem usage (considering only matrix elements): " << memUsageForAllMatricesInGB << " GB" << std::endl;
  double startTime = get_wall_seconds();
  int resultCode = test_creation<hbsm::HierarchicalBlockSparseMatrix<double> >(N, blockSize, nonzeroFraction, nMat, nRep);
  double secondsTaken = get_wall_seconds() - startTime;
  printf("secondsTaken = %12.3f wall seconds\n", secondsTaken);
  return resultCode;
}
