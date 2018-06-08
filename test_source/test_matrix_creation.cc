/* Hierarchical-Block-Sparce-Lib, version 1.0. A hierarchical block sparse matrix library..
 * Copyright (C) Anton Artemov anton.artemov@it.uu.se.
 *
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



template<typename MatrixType>
static int test_creation() {
	typename MatrixType::Params param;
#if 1  
	param.blocksize = 4; // Only for block sparse matrix
#endif

	MatrixType M;

	M.set_params(param);
	
	if(!M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.resize(4, 4);
    
    if(M.get_n_rows() != 4 || M.get_n_cols() != 4 )
        throw std::runtime_error("Error: M.get_n_rows() or get_n_cols() gave wrong result.");
    
	if(M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.clear();
	if(!M.empty())
		throw std::runtime_error("Error: M.empty() or clear() gave wrong result.");

	M.resize(7, 16);
    
    if(M.get_n_rows() != 7 || M.get_n_cols() != 16 )
        throw std::runtime_error("Error: M.get_n_rows() or get_n_cols() gave wrong result.");
    
    
	if(M.get_frob_squared() != 0)
		throw std::runtime_error("Error: M.get_frob_squared() != 0 for newly created matrix.");

    M.clear();
    M.resize(32, 32);
    
    if(M.get_n_rows() != 32|| M.get_n_cols() != 32 )
    throw std::runtime_error("Error: M.get_n_rows() or get_n_cols() gave wrong result.");
    

	double refValue1 = 7.7;
    double refValue2 = 1.1;

	  // Test assign_from_vectors()
	{
		int nValues = 2;
		std::vector<int> rows(nValues);
		std::vector<int> cols(nValues);
		std::vector<double> values(nValues);
		rows  [0] = 0;
		cols  [0] = 0;
		values[0] = refValue1;
		rows  [1] = 6;
		cols  [1] = 7;
		values[1] = refValue2;
		M.assign_from_vectors(rows, cols, values);
        
        if(!M.children_exist())   
            throw std::runtime_error("Error: M.chilren_exist() gave wrong result.");
        
		double expected_frob_sq = refValue1*refValue1 + refValue2*refValue2;
		
		if(fabs(M.get_frob_squared() - expected_frob_sq) > 1e-7)
			throw std::runtime_error("Error: M.get_frob_squared() gave wrong result.");
            
        if(M.get_nnz() != nValues)
            throw std::runtime_error("Error: M.get_nnz() gave wrong result.");
            
       
	}

	  // Test get_values()
	{
		int nValues2 = 3;
		std::vector<int> rows(nValues2), rows_nnz;
		std::vector<int> cols(nValues2), cols_nnz;
		rows  [0] = 0;
		cols  [0] = 0;
		rows  [1] = 6;
		cols  [1] = 7;
		rows  [2] = 3;
		cols  [2] = 3;
		std::vector<double> values, values_nnz;
		
		M.get_values(rows, cols, values);
		
		if(std::fabs(values[0] - refValue1) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[1] - refValue2) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[2] - 0) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
            
        M.get_all_values(rows_nnz, cols_nnz, values_nnz);
        
        if(rows_nnz.size() != 2)
            throw std::runtime_error("Error: wrong result from get_all_values().");
     
        assert(rows_nnz[0] == 0 && cols_nnz[0] == 0 && values_nnz[0] == refValue1);
        assert(rows_nnz[1] == 6 && cols_nnz[1] == 7 && values_nnz[1] == refValue2);
	}
	
    
	size_t size = M.get_size();
    if(size != 516) // two children each 5*4 + 4*8 + 16*8 = 180, plus 3 higher levels 52 bytes each
        throw std::runtime_error("Error: wrong result from get_size().");
	
	std::vector<char> buf(size);
	M.write_to_buffer(&buf[0], size);
	
	MatrixType B;
	B.assign_from_buffer(&buf[0], size);
    
    verify_that_matrices_are_equal<MatrixType>(M,B);

    MatrixType C;
    C.copy(M);

    verify_that_matrices_are_equal<MatrixType>(M,C);
    verify_that_matrices_are_equal<MatrixType>(M,C);
    verify_that_matrices_are_equal<MatrixType>(B,C);
    
    C.clear();
    M.clear();
    B.clear();

	return 0;
}


int main() {  

	return test_creation<hbsm::HierarchicalBlockSparseMatrix<double> >();
  

}
