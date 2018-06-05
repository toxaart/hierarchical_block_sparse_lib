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
  
  A->resize(25,15);
  
  std::vector<int> rows, cols;
  std::vector<real> vals;
  
  rows.push_back(0);
  cols.push_back(0);
  vals.push_back(1.0);
  
  rows.push_back(1);
  cols.push_back(1);
  vals.push_back(1.0);
  
  rows.push_back(7);
  cols.push_back(7);
  vals.push_back(2.0);
  
  rows.push_back(14);
  cols.push_back(14);
  vals.push_back(2.0);
  
  A->assign_from_vectors_general(rows,cols,vals,false,false);
  
  delete A;
  
 
	
  return 0;
}
