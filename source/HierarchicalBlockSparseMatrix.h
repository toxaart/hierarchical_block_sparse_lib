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
#ifndef HBSM_HIERARCHICAL_BLOCK_SPARSE_MATRIX_HEADER
#define HBSM_HIERARCHICAL_BLOCK_SPARSE_MATRIX_HEADER
#include <vector>
#include <list>
//#include <forward_list>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#if BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaManager.h"
#include "CudaSyncPtr.h"
#endif
#include "gblas.h"

// Use namespace hbsm: "hierarchical block sparse matrix library".
namespace hbsm {

	// Matrix class template for hiearachical sparse matrices. Stores blocks only
	// at the lowest level of hiearchy, otherwise keeps pointers to lower level matrices (always 4).

	template<class Treal>
		class HierarchicalBlockSparseMatrix{
	public:
			typedef Treal real;
	private:
			int nRows; // number of rows on the current level
			int nCols; // number of cols on the current level
			int blocksize; // size of blocks at the lowest level (blocksize x blocksize)
			HierarchicalBlockSparseMatrix* children[4]; // array of pointers to the next level.
			/*		child0 | child2
			 * 		--------------		
			 *      child1 | child3
			 * 
			 * */
			
			std::vector<real> submatrix; // actual data is here if lowest level, otherwise is empty.
				
			inline bool is_power_of_2(int n){
				return n && !(n & (n - 1));
			}
			
			bool lowest_level() const {
				return (nRows == blocksize) && (nCols == blocksize) && !children_exist();
			}
						
	public:
			struct Params {
			  int blocksize;
			};
		
			HierarchicalBlockSparseMatrix():nRows(0), nCols(0), blocksize(-1){
				children[0] = NULL;
				children[1] = NULL;
				children[2] = NULL;
				children[3] = NULL;
				submatrix.clear();
			}
			
			
			~HierarchicalBlockSparseMatrix(){
				if(children[0] != NULL) delete children[0];
				if(children[1] != NULL) delete children[1];
				if(children[2] != NULL) delete children[2];
				if(children[3] != NULL) delete children[3];
				submatrix.clear();
			}

			int get_n_rows() const { return nRows; }
			int get_n_cols() const { return nCols; }
			void set_params(Params const & param); 
			Params get_params() const;
			bool children_exist() const; 
			bool empty() const;
			void resize(int nRows_, int nCols_);
			void clear();
    };
	
	
	template<class Treal> 
		bool HierarchicalBlockSparseMatrix<Treal>::children_exist() const  {
			return (children[0] != NULL || children[1] != NULL || children[2] != NULL || children[3] != NULL);
		}
		
	template<class Treal> 
		bool HierarchicalBlockSparseMatrix<Treal>::empty() const {
			if(nRows == 0 && nCols == 0 && submatrix.empty() && !children_exist())
			  return true;
			else
			  return false;
		}		
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrix<Treal>::set_params(Params const & param) {
			if ( !empty() )
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::set_params: Matrix must be empty when setting params.");
			blocksize = param.blocksize;
		}	
  
	template<class Treal>
		typename HierarchicalBlockSparseMatrix<Treal>::Params HierarchicalBlockSparseMatrix<Treal>::get_params() const {
			Params p;
			p.blocksize = blocksize;
			return p;
		}
	
	template<class Treal> 
		void HierarchicalBlockSparseMatrix<Treal>::resize(int nRows_, int nCols_) {
		assert(blocksize > 0);
		
		// lowest level
		// FIXME is it the right way to check if lowest level reached?
		if(nRows_ == blocksize && nCols_ == blocksize){
			nRows = nRows_;
			nCols = nCols_;
			submatrix.clear();
			return;
		}
	
		// actually, matrix is always kept square, with padding with zeros where necessary
		// the "virtual dimension" can be computed as blocksize * 2^P, 
		// where P is such that blocksize * 2^(P-1) <= max(nRows_, nCols_) <= blocksize * 2^P 
		
		int maxdim = ((nRows_ > nCols_) ? nRows_ : nCols_);
		
		//how many times block covers maxdim
		int n_covers = maxdim / blocksize;
		if(maxdim % blocksize != 0) n_covers += 1;
		
		// now we need to find P such that 2^(P-1) <= n_covers <= 2^P
		int P = 1;
		int two_to_power_P = 2;
		while(n_covers >  two_to_power_P){
			two_to_power_P *= 2;
			P += 1;
		}
		
		int virtual_size = blocksize * two_to_power_P;
		
		/*
		std::cout << "maxdim / blocksize = " << maxdim / blocksize << std::endl;
		std::cout << "maxdim % blocksize = " << maxdim % blocksize << std::endl;
		std::cout << "n_covers = " << n_covers << std::endl;
		std::cout << "P = " << P << std::endl;
		std::cout << "virtual size is " << virtual_size << std::endl;
		*/
		
		nRows = virtual_size;
		nCols = virtual_size;
	
		
		for(int i = 0; i < 4; ++i){
			if(children[i] != NULL) delete children[i];
			children[i] = new HierarchicalBlockSparseMatrix<Treal>();			
 			children[i]->set_params(get_params());
			children[i]->resize(nRows / 2, nCols / 2);
		}
	}
	
	template<class Treal> 
		void HierarchicalBlockSparseMatrix<Treal>::clear() {
			if(lowest_level()){
				submatrix.clear();
				return;
			}
			nCols = 0;
			nRows = 0;
			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL){
					children[i]->clear();
					delete children[i];
				}
			}	
		}
  
} /* end namespace hbsm */

#endif
