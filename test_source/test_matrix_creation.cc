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
  
  A->resize(32,32);
  
  A->clear();
  
  A->resize(16,16);
  
  delete A;
  
 
	
  return 0;
}
