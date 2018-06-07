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
	param.blocksize = 1; // Only for block sparse matrix
#endif

	MatrixType M;
	
	M.set_params(param);
	
	if(!M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.resize(3, 7);
	
	if(M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.clear();
	if(!M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.resize(3, 7);
	if(M.get_frob_squared() != 0)
		throw std::runtime_error("Error: M.get_frob_squared() != 0 for newly created matrix.");

	double refValue1 = 7.7;
    double refValue2 = 1.1;

	  // Test assign_from_vectors()
	{
		int nValues = 2;
		std::vector<int> rows(nValues);
		std::vector<int> cols(nValues);
		std::vector<double> values(nValues);
		rows  [0] = 2;
		cols  [0] = 6;
		values[0] = refValue1;
		rows  [1] = 1;
		cols  [1] = 3;
		values[1] = refValue2;
		M.assign_from_vectors(rows, cols, values);
		double expected_frob_sq = refValue1*refValue1 + refValue2*refValue2;
		
		if(fabs(M.get_frob_squared() - expected_frob_sq) > 1e-7)
			throw std::runtime_error("Error: M.get_frob_squared() gave wrong result.");
	}

	  // Test get_values()
	{
		int nValues2 = 3;
		std::vector<int> rows(nValues2);
		std::vector<int> cols(nValues2);
		rows  [0] = 2;
		cols  [0] = 6;
		rows  [1] = 1;
		cols  [1] = 3;
		rows  [2] = 0;
		cols  [2] = 5;
		std::vector<double> values;
		
		M.get_values(rows, cols, values);
		
		if(std::fabs(values[0] - refValue1) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[1] - refValue2) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[2] - 0) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
	}
	
	// Test write_to_buffer()
	size_t size = M.get_size();
	std::vector<char> buf(size);
	M.write_to_buffer(&buf[0], size);

	return 0;
}


int main() {  

	return test_creation<hbsm::HierarchicalBlockSparseMatrix<double> >();
  
	
}
