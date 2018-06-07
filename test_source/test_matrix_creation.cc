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


static int test_creation() {


  return 0;
}


int main() {  
	
  typedef double real;	
  
  hbsm::HierarchicalBlockSparseMatrix<real> *A = new hbsm::HierarchicalBlockSparseMatrix<real>();	
  
  hbsm::HierarchicalBlockSparseMatrix<real>::Params params;
  params.blocksize = 4;
  
  A->set_params(params);
  
  std::cout << "before resize " << A->get_size() << std::endl;
  std::cout << "sizeof(HierarchicalBlockSparseMatrix<Treal>*) is " << sizeof(hbsm::HierarchicalBlockSparseMatrix<real>*) << std::endl;
  std::cout << "sizeof(int) is " << sizeof(int) << std::endl; 
  std::cout << "sizeof(real) is " << sizeof(real) << std::endl; 
  std::cout << "sizeof(size_t) is " << sizeof(size_t) << std::endl;
  
  
    
  A->resize(4,4);
  std::cout << " after resize: size of A is " << A->get_size() << std::endl;

/*
  std::vector<int> rows, cols;
  std::vector<real> vals;
  
  rows.push_back(0);
  cols.push_back(0);
  vals.push_back(1.0);
  
  rows.push_back(1);
  cols.push_back(1);
  vals.push_back(1.0);
  
  rows.push_back(3);
  cols.push_back(0);
  vals.push_back(2.0);

  
  A->assign_from_vectors(rows,cols,vals);
  */
  
  std::cout << "size of A is " << A->get_size() << std::endl;

  
  
  size_t size_of_A = A->get_size();
  std::vector<char> buf(size_of_A);
  A->write_to_buffer(&buf[0],size_of_A);
  
  
  hbsm::HierarchicalBlockSparseMatrix<real> B;
  B.assign_from_buffer(&buf[0],size_of_A);
  
  std::cout << "B has size " << B.get_size() << std::endl;
  
  delete A;
  
 
	
  return 0;
}
