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
#ifndef HBSM_HIERARCHICAL_BLOCK_SPARSE_MATRIX_SMART_POINTERS_HEADER
#define HBSM_HIERARCHICAL_BLOCK_SPARSE_MATRIX_SMART_POINTERS_HEADER
#include <vector>
#include <list>
#include <iterator>
//#include <forward_list>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "gblas.h"
#include <random>
#include <algorithm>
#include <memory>
#if BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaManager.h"
#include "CudaSyncPtr.h"
#endif

// Use namespace hbsm: "hierarchical block sparse matrix library".
namespace hbsm {

	// Matrix class template for hiearachical sparse matrices. Stores blocks only
	// at the lowest level of hiearchy, otherwise keeps pointers to lower level matrices (always 4).

	template<class Treal>
		class HierarchicalBlockSparseMatrixSmartPointers: std::enable_shared_from_this<HierarchicalBlockSparseMatrixSmartPointers<Treal> >{
	public:
			typedef Treal real;
	private:
			int nRows; // number of rows on the current level
			int nCols; // number of cols on the current level
			int nRows_orig; // before 'virtual size' has been computed
			int nCols_orig; // before 'virtual size' has been computed
			int blocksize; // size of blocks at the lowest level (blocksize x blocksize)
			std::shared_ptr<HierarchicalBlockSparseMatrixSmartPointers<real> > children[4]; // array of pointers to the next level.
			HierarchicalBlockSparseMatrixSmartPointers<real>* parent; // way to go to top level;
			/*		child0 | child2
			 * 		--------------		
			 *      child1 | child3
			 * 
			 * */
	
			std::vector<Treal> submatrix; // actual data is here if lowest level, otherwise is empty.

			
			struct Transpose{
				  const char              *bt;
				  bool                     t;
				  Transpose(const char *bt, bool t) : bt(bt), t(t) {}
				  static Transpose        N() { return Transpose("N", false); }
				  static Transpose        T() { return Transpose("T", true); }
			};
			
				
				
	public:
			struct Params {
			  int blocksize;
			};
		
			HierarchicalBlockSparseMatrixSmartPointers():nRows(0), nCols(0),nRows_orig(0), nCols_orig(0), blocksize(-1){}
						
			~HierarchicalBlockSparseMatrixSmartPointers(){
				submatrix.clear();
			}
			
			
			int get_n_rows() const; //returns n_rows in ORIGINAL matrix, can be called from any level still gives results for original matrix
			int get_n_cols() const; //returns n_cols in ORIGINAL matrix, can be called from any level still gives results for original matrix
			void set_params(Params const & param); 
			Params get_params() const;
			bool children_exist() const; 
			bool empty() const;
			void resize(int nRows_, int nCols_, size_t* no_of_resizes = NULL);

								
    };
	
	template<class Treal> 
		bool HierarchicalBlockSparseMatrixSmartPointers<Treal>::empty() const {	           
			if(nRows == 0 && nCols == 0 && submatrix.empty() && !children_exist())
			  return true;
			else
			  return false;
		}		
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::set_params(Params const & param) {
			if ( !empty() )
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::set_params: Matrix must be empty when setting params.");
			blocksize = param.blocksize;
		}	
  
	template<class Treal>
		typename HierarchicalBlockSparseMatrixSmartPointers<Treal>::Params HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_params() const {
			Params p;
			p.blocksize = blocksize;
			return p;
		}
		
	template<class Treal> 
		bool HierarchicalBlockSparseMatrixSmartPointers<Treal>::children_exist() const  {
			return (children[0] != NULL || children[1] != NULL || children[2] != NULL || children[3] != NULL);
		}

	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::resize(int nRows_, int nCols_, size_t* no_of_resizes) {
		assert(blocksize > 0);
		
		nRows_orig = nRows_; // only top level contains true size, rest have duplicates of nRows and nCols
		nCols_orig = nCols_;
				
		// lowest level	
		if(nRows_ <= blocksize && nCols_ <= blocksize){
			nRows = blocksize;
			nCols = blocksize;
			submatrix.resize(blocksize*blocksize);
			if(no_of_resizes != NULL) (*no_of_resizes)++;
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
		
		
		nRows = virtual_size;
		nCols = virtual_size;
	
	} 

	template<class Treal> 
		int HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_n_rows() const  {
			if(parent == NULL)
				return nRows_orig;
			else{
				const HierarchicalBlockSparseMatrixSmartPointers<Treal> *tmp = this;
				while(tmp->parent != NULL){
					tmp = tmp->parent;
                  } 
				return tmp->nRows_orig;
			}					
		}
		
	template<class Treal> 
		int HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_n_cols() const  {
			if(parent == NULL)
				return nCols_orig;
			else{
				const HierarchicalBlockSparseMatrixSmartPointers<Treal> *tmp = this;
				while(tmp->parent != NULL){
					tmp = tmp->parent;
                  } 
				return tmp->nCols_orig;
			}		
		}	
      
} /* end namespace hbsm */

#endif