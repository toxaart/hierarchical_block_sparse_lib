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
	
	if(M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.clear();
	if(!M.empty())
		throw std::runtime_error("Error: M.empty() gave wrong result.");

	M.resize(16, 16);
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
		rows  [0] = 0;
		cols  [0] = 0;
		values[0] = refValue1;
		rows  [1] = 1;
		cols  [1] = 1;
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
		rows  [0] = 0;
		cols  [0] = 0;
		rows  [1] = 1;
		cols  [1] = 1;
		rows  [2] = 3;
		cols  [2] = 3;
		std::vector<double> values;
		
		M.get_values(rows, cols, values);
		
		if(std::fabs(values[0] - refValue1) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[1] - refValue2) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
		
		if(std::fabs(values[2] - 0) > 1e-7)
			throw std::runtime_error("Error: wrong result from get_values().");
	}
	

	size_t size = M.get_size();
	std::cout << "C size before clearing " << size <<std::endl; 
	
/*
	M.clear();
	
	size = M.get_size();
    std::cout << "C size after clearing " << size <<std::endl; */
	
	std::vector<char> buf(size);
	M.write_to_buffer(&buf[0], size);
	
	
	MatrixType B;
	B.assign_from_buffer(&buf[0], size);
	
	std::cout << "|M|^2 = " << M.get_frob_squared() << std::endl;
	std::cout << "|B|^2 = " << B.get_frob_squared() << std::endl;
	
	{
		int nValues3 = 3;
		std::vector<int> rows(nValues3);
		std::vector<int> cols(nValues3);
		rows  [0] = 0;
		cols  [0] = 0;
		rows  [1] = 1;
		cols  [1] = 1;
		rows  [2] = 3;
		cols  [2] = 3;
		std::vector<double> values_M, values_B;
	
		M.get_values(rows, cols, values_M);
		B.get_values(rows, cols, values_B);
		
		for(int i = 0; i < nValues3; ++i){
			std::cout << values_M[i] << " " << values_B[i] << std::endl;
		}
	
	}

	return 0;
}


int main() {  

	return test_creation<hbsm::HierarchicalBlockSparseMatrix<double> >();
  
	
}
