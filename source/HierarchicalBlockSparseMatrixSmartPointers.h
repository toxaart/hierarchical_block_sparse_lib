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
		class HierarchicalBlockSparseMatrixSmartPointers{
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
			
			bool lowest_level() const {
				return (nRows == blocksize) && (nCols == blocksize) && !children_exist();
			}	
			
			Treal get_single_value(int row, int col) const;
			
			// computes the distance in the hierarchy from top level to current. 
			int get_level() const; 
            
            // function returns code consisting of digits 0-3, which indicated the path from root to particular submatrix if read from left to right
            std::string get_position_code() const;
            
            // functions returning top left corner of any matrix in hierarchy
            int get_first_col_position() const;
            int get_first_row_position() const;
            
            bool on_right_boundary() const; // for any level!
            bool on_bottom_boundary() const;
				
	public:
			struct Params {
			  int blocksize;
			};
		
			HierarchicalBlockSparseMatrixSmartPointers():nRows(0), nCols(0),nRows_orig(0), nCols_orig(0), blocksize(-1){
				parent = NULL;
			}
						
			~HierarchicalBlockSparseMatrixSmartPointers(){
				parent = NULL;
				if(submatrix.size() > 0) submatrix.clear();
			}
			
			
			int get_n_rows() const; //returns n_rows in ORIGINAL matrix, can be called from any level still gives results for original matrix
			int get_n_cols() const; //returns n_cols in ORIGINAL matrix, can be called from any level still gives results for original matrix
			void set_params(Params const & param); 
			Params get_params() const;
			bool children_exist() const; 
			bool empty() const;
			void resize(int nRows_, int nCols_, size_t* no_of_resizes = NULL);
			void clear();
			void assign_from_vectors_general(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values,
				     bool useMax,
					 bool boundaries_checked);					 
			void assign_from_vectors(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values);
			void assign_from_vectors_max(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values,
					 bool use_max);	
			Treal get_frob_squared() const;
			
			size_t get_nnz() const;

			// get values with indices kept in vectors rows and cols			  
			void get_values(const std::vector<int> & rows, 
						  const std::vector<int> & cols,
						  std::vector<Treal> & values) const;	

			void get_all_values(std::vector<int> & rows,
						  std::vector<int> & cols,
						  std::vector<Treal> & values) const;
								
			size_t get_size() const;

			void write_to_buffer ( char * dataBuffer, size_t const bufferSize ) const;
			void assign_from_buffer ( char * dataBuffer, size_t const bufferSize );


			void copy(const HierarchicalBlockSparseMatrixSmartPointers<Treal> & other, size_t* no_of_resizes = NULL);

			void add_scaled_identity( HierarchicalBlockSparseMatrixSmartPointers<Treal> const & other, Treal alpha);
    };
	
			
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
		bool HierarchicalBlockSparseMatrixSmartPointers<Treal>::empty() const {	           
			if(nRows == 0 && nCols == 0 && submatrix.empty() && !children_exist())
			  return true;
			else
			  return false;
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
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::clear() {
			
			if(empty()){
				return;
			}
			
			if(lowest_level()){              
				submatrix.clear();
				nRows = 0;
				nCols = 0;
				if(children_exist()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::clear: children exist on leaf level.");
				}
				return;
			}
			
			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL){
					children[i]->clear(); // smart pointer, no need to call delete explicitly!
					children[i] = NULL;
				}
			}
			nRows = 0;
			nCols = 0;
		}
      
	template<class Treal> 
		Treal HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_frob_squared() const  {
			
			if(empty()) 
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_frob_squared: empty matrix occured.");
			
		    if(lowest_level()){
				
				Treal frob_norm_squared = 0.0;		
				for(int i = 0; i < submatrix.size(); ++i){
					frob_norm_squared += submatrix[i] * submatrix[i];
				}
				return frob_norm_squared;
			}
			else{
				
				Treal frob_norm_squared = 0.0;
				
				for(int i = 0; i < 4; ++i){
					if(children[i] != NULL) frob_norm_squared += children[i]->get_frob_squared();
				}
				
				return frob_norm_squared;
			}
			
		}
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors_general(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values,
				     bool useMax,
					 bool boundaries_checked) {
						 
						 
			
			assert(blocksize > 0);
			if(rows.size() != values.size() || cols.size() != values.size())
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: bad sizes.");
			
			if(rows.size() == 0) return;
		
			// this will be done only at the top level, later it is not necessary
			if(!boundaries_checked){
				for(int i = 0; i < values.size(); ++i){
					if( rows[i] < 0 ||  rows[i] > nRows_orig-1 || cols[i] < 0 || cols[i] > nCols_orig - 1) 
						throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: index outside matrix boundaries.");
						
				}
			}
			
			if(lowest_level()){
                
				for(int i = 0; i < nCols*nRows; ++i){
					submatrix[i] = 0.0;
				}
                
                int max_row_num = nRows;
                if(on_bottom_boundary()) max_row_num = get_n_rows() % blocksize;
				//std::cout << "max_row_num " << max_row_num << std::endl;
                
                int max_col_num = nCols;
                if(on_right_boundary()) max_col_num = get_n_cols() % blocksize;
				//std::cout << "max_col_num " << max_col_num << std::endl;
                
                if(get_first_row_position() >= get_n_rows() || get_first_col_position() >= get_n_cols()){ // block to skip
					//std::cout << "assign_from_vectors: block skipped at leaf level " << std::endl;
                    return;
                }
				
				//std::cout << values.size() << std::endl;
                    
				for(size_t i = 0; i < values.size(); ++i){
					
                    int row = rows[i];
					int col = cols[i];
                    
                    // if boundary block, do not write elements outside big matrix
                    if(row >= max_row_num || col >= max_col_num){
						//std::cout << "elements outside" <<std::endl;
						continue;
					}
                    
					Treal val = values[i];
					
                    if(useMax){
                        submatrix[col * nRows + row] = (val > submatrix[col * nRows + row]) ? val : submatrix[col * nRows + row];
                    }
                    else{
                        submatrix[col * nRows + row] += val; // Note: we use += here, so if an element appears several times, the result becomes the sum of those contributions. 
                    }

				}

				return;
			}
			
			// now we have to split arrays into 4 parts and recursively apply the same procedure on children
			
			
			std::vector<int> rows0, cols0, rows1, cols1, rows2, cols2, rows3, cols3;
            std::vector<Treal> vals0,vals1,vals2,vals3;
            
            rows0.reserve(rows.size());
            cols0.reserve(rows.size());
            vals0.reserve(rows.size());
            
            rows1.reserve(rows.size());
            cols1.reserve(rows.size());
            vals1.reserve(rows.size());
            
            rows2.reserve(rows.size());
            cols2.reserve(rows.size());
            vals2.reserve(rows.size());
            
            rows3.reserve(rows.size());
            cols3.reserve(rows.size());
            vals3.reserve(rows.size());
			
			// the offset, when providing parts of vectors to the next level their "coordinates" will be shifted
			int offset = nRows/2;
			
			for(size_t i = 0; i < values.size(); ++i){
											
				int row = rows[i];
				int col = cols[i];
				Treal val = values[i];
			
				// this part goes to child0
				if(row < offset && col < offset){
					rows0.push_back(row);
					cols0.push_back(col);
					vals0.push_back(val);
				}
				
				// this part goes to child1
				if(offset <= row && row < nRows && col < offset){
					rows1.push_back(row-offset);
					cols1.push_back(col);
					vals1.push_back(val);
				}
				
				// this part goes to child2
				if(row < offset && offset <= col && col < nCols){
					rows2.push_back(row);
					cols2.push_back(col-offset);
					vals2.push_back(val);
				}
				
				// this part goes to child3
				if(offset <= row && row < nRows && offset <= col && col < nCols){
					rows3.push_back(row-offset);
					cols3.push_back(col-offset);
					vals3.push_back(val);
				}
				
			}


			if(vals0.size() > 0){
				if(children[0] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors: non-null child0 matrix occured.");
				}
				children[0] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();			
			    children[0]->set_params(get_params());
				children[0]->resize(nRows / 2, nCols / 2);
				children[0]->parent = this;
                children[0]->assign_from_vectors_general(rows0, cols0, vals0, useMax,true);
			}
			
			if(vals1.size() > 0){
				if(children[1] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors: non-null child1 matrix occured.");
				}
				children[1] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();			
			    children[1]->set_params(get_params());
				children[1]->resize(nRows / 2, nCols / 2);
				children[1]->parent = this;
                children[1]->assign_from_vectors_general(rows1, cols1, vals1, useMax,true);
			}
			
			if(vals2.size() > 0){
				if(children[2] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors: non-null child2 matrix occured.");
				}
				children[2] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();			
			    children[2]->set_params(get_params());
				children[2]->resize(nRows / 2, nCols / 2);
				children[2]->parent = this;
                children[2]->assign_from_vectors_general(rows2, cols2, vals2, useMax,true);
			}
			
			if(vals3.size() > 0){
				if(children[3] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors: non-null child3 matrix occured.");
				}
				children[3] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();	
			    children[3]->set_params(get_params());
				children[3]->resize(nRows / 2, nCols / 2);
				children[3]->parent = this;
                children[3]->assign_from_vectors_general(rows3, cols3, vals3, useMax,true);
			}		
			
		}
		
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values) {
			 return HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors_general(rows, cols, values, false, false);
		}
  
  	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors_max(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values,
					 bool use_max) {
			 return HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_vectors_general(rows, cols, values, use_max, false);
		}	
	
	 template<class Treal> 
		Treal HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_single_value(int row, int col) const {
			
			
			if ( empty() )
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_single_value: empty matrix occured.");
		
		
			if(!(0 <= row && row < nRows_orig && 0 <= col && col < nCols_orig ))
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_single_value: bad index at highest level.");
			
				
		
		
			if(!(0 <= row && row < nRows && 0 <= col && col < nCols ) ){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_single_value: bad index.");
			}
			
			if(lowest_level()){
				return submatrix[col*nRows + row];
			}
			
			
			int offset = nRows/2;
						
			// this part goes to child0
			if(row < offset && col < offset){
				if(children[0] == NULL) return 0.0;
				else return children[0]->get_single_value(row,col);
			}
			
			// this part goes to child1
			if(offset <= row && row < nRows && col < offset){
				if(children[1] == NULL) return 0.0;
				else return children[1]->get_single_value(row-offset,col);
			}
			
			// this part goes to child2
			if(row < offset && offset <= col && col < nCols){
				if(children[2] == NULL) return 0.0;
				else return children[2]->get_single_value(row,col-offset);
			}
			
			// this part goes to child3
			if(offset <= row && row < nRows && offset <= col && col < nCols){
				if(children[3] == NULL) return 0.0;
				else return children[3]->get_single_value(row-offset,col-offset);
			}
				
		}	

	template<class Treal> 
		std::string HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_position_code() const  {

            HierarchicalBlockSparseMatrixSmartPointers<Treal> *tmp = parent;
			
            if(parent != NULL){
                std::string code;
                if(this == parent->children[0].get()) code = "0";
                if(this == parent->children[1].get()) code = "1";
                if(this == parent->children[2].get()) code = "2";
                if(this == parent->children[3].get()) code = "3";
                return parent->get_position_code() + code;
            }
            else{
                return "";
            }
					
		}
		
	template<class Treal> 
		int HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_first_col_position() const  {
    
            if(parent == NULL) return 0; // top level
            else{
              
               std::string position_code_str = get_position_code();
               int x0 = 0;
               int two_to_power = 1;
               
                for(int i = position_code_str.length()-1; i >= 0; --i){
                    char K = position_code_str[i];
                    if(K == '2' || K == '3'){
                        x0 += two_to_power;
                    }
                    two_to_power *= 2;
                }
                
                x0 *= nCols;
                return x0;
            }
            
		}
        
    template<class Treal> 
		int HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_first_row_position() const  {
            
            if(parent == NULL) return 0; // top level
            else{
              
               std::string position_code_str = get_position_code();
               int y0 = 0;
               int two_to_power = 1;
               
                for(int i = position_code_str.length()-1; i >= 0; --i){
                    char K = position_code_str[i];
                    if(K == '1' || K == '3'){
                        y0 += two_to_power;
                    }
                    two_to_power *= 2;
                }
                
                y0 *= nRows;
                return y0;
            }
            
		}

	template<class Treal> 
		bool HierarchicalBlockSparseMatrixSmartPointers<Treal>::on_right_boundary() const  {
            if((get_first_col_position() + nCols > get_n_cols()) && (get_first_col_position() < get_n_cols())) return true;
            else return false;
		} 

	template<class Treal> 
		bool HierarchicalBlockSparseMatrixSmartPointers<Treal>::on_bottom_boundary() const  {
            if((get_first_row_position() + nRows > get_n_rows()) && (get_first_row_position()< get_n_rows())) return true;
            else return false;
		}   

	template<class Treal> 
		size_t HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_nnz() const  {
			if(empty()) 
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_nnz: empty matrix occured.");
			
		    if(lowest_level()){
				
				int nnz = 0;			
				for(int i = 0; i < submatrix.size(); ++i){
					if(fabs(submatrix[i]) > 0.0) nnz += 1; 
				}
				
				return nnz;
			}
			else{
				
				int nnz = 0.0;
				
				for(int i = 0; i < 4; ++i){
					if(children[i] != NULL) nnz += children[i]->get_nnz();
				}
				
				return nnz;
			}
			
		}
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_values(const std::vector<int> & rows,
						  const std::vector<int> & cols,
						  std::vector<Treal> & values) const  {
			
			if(empty()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_values: empty matrix occured.");
			}						  
							  
			if(rows.size() != cols.size()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_values: bad sizes.");
			}		

			if(rows.size() == 0) return;
			
			values.resize(rows.size());
							
			for(int i = 0; i < rows.size(); ++i){
				values[i] = get_single_value(rows[i],cols[i]);
			}
		}
		
	template<class Treal> 
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_all_values(std::vector<int> & rows,
						  std::vector<int> & cols,
						  std::vector<Treal> & values) const  {
			
							  
			if(!rows.empty()) rows.clear();
			if(!cols.empty()) cols.clear();
			if(!values.empty()) values.clear();		

            rows.reserve(nRows * nCols);
            cols.reserve(nRows * nCols);
            values.reserve(nRows * nCols);
			  
			if(empty()){;
				return;
			}				  
							  
			if(lowest_level()){      				
				for(int i = 0; i < submatrix.size(); ++i){
                  
					if(fabs(submatrix[i]) > 0.0){
						
						int col = i / nRows;
						int row = i % nRows;
						rows.push_back(row);
						cols.push_back(col);
						values.push_back(submatrix[i]);
						
					}
				}
	
				return;
			}
			
			// here we have to merge vectors obtained from children, takin into account offset of indices
			int offset = nRows/2;
			
			std::vector<int> rows0,rows1,rows2,rows3;
			std::vector<int> cols0,cols1,cols2,cols3;
			std::vector<Treal> vals0,vals1,vals2,vals3;
			
			
			if(children[0] != NULL){
				children[0]->get_all_values(rows0,cols0,vals0);
				assert(rows0.size() == cols0.size() && rows0.size() == vals0.size());
				// child0 - no offset applied
				for(int i = 0; i < rows0.size(); ++i){
					rows.push_back(rows0[i]);
					cols.push_back(cols0[i]);
					values.push_back(vals0[i]);
				}				
			}
			
			if(children[1] != NULL){
				children[1]->get_all_values(rows1,cols1,vals1);
				assert(rows1.size() == cols1.size() && rows1.size() == vals1.size());
				// child1 - offset applied to row number
				for(int i = 0; i < rows1.size(); ++i){
					rows.push_back(rows1[i]+offset);
					cols.push_back(cols1[i]);
					values.push_back(vals1[i]);
				}		
			}
			
			if(children[2] != NULL){
				children[2]->get_all_values(rows2,cols2,vals2);
				assert(rows2.size() == cols2.size() && rows2.size() == vals2.size());
				// child2 - offset applied to col number
				for(int i = 0; i < rows2.size(); ++i){
					rows.push_back(rows2[i]);
					cols.push_back(cols2[i]+offset);
					values.push_back(vals2[i]);
				}
			}
			
			
			if(children[3] != NULL){
				children[3]->get_all_values(rows3,cols3,vals3);
				assert(rows3.size() == cols3.size() && rows3.size() == vals3.size());
				// child3 - offset applied to both col and row number
				for(int i = 0; i < rows3.size(); ++i){
					rows.push_back(rows3[i]+offset);
					cols.push_back(cols3[i]+offset);
					values.push_back(vals3[i]);
				}
			}

		}	

	template<class Treal>
		size_t HierarchicalBlockSparseMatrixSmartPointers<Treal>::get_size() const  {
			if(empty()) return 5 * sizeof(int) + 4 * sizeof(size_t);

			if(lowest_level()){
				return 5 * sizeof(int) + 4 * sizeof(size_t) + submatrix.size() * sizeof(Treal);
			}

			size_t totalsize = 5 * sizeof(int) + 4 * sizeof(size_t); // keep sizes of children in a row!


			if(children[0] != NULL){
				totalsize += children[0]->get_size();
			}


			if(children[1] != NULL) {
				totalsize += children[1]->get_size();
			}


			if(children[2] != NULL){
				totalsize += children[2]->get_size();
			}


			if(children[3] != NULL){
				totalsize += children[3]->get_size();
			}


			return totalsize;

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::write_to_buffer(char * dataBuffer, size_t const bufferSize) const  {
			if(bufferSize < get_size())
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers<Treal>::write_to_buffer(): buffer too small.");

			size_t size_of_matrix_pointer = sizeof(HierarchicalBlockSparseMatrixSmartPointers<Treal>*);

			char* p = dataBuffer;

			if(empty()){

				std::cout << "empty case in write_to_buffer, buffer size " << bufferSize << std::endl;
				int n_bytes_written = 0;

				memcpy(p, &nRows, sizeof(int));
				p += sizeof(int);
				n_bytes_written += sizeof(int);

				memcpy(p, &nCols, sizeof(int));
				p += sizeof(int);
				n_bytes_written += sizeof(int);

				memcpy(p, &nRows_orig, sizeof(int));
				p += sizeof(int);
				n_bytes_written += sizeof(int);

				memcpy(p, &nCols_orig, sizeof(int));
				p += sizeof(int);
				n_bytes_written += sizeof(int);

				memcpy(p, &blocksize, sizeof(int));
				p += sizeof(int);
				n_bytes_written += sizeof(int);

				//no children - write 4 zeros!
				size_t zero_size = 0;
				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);
				n_bytes_written += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);
				n_bytes_written += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);
				n_bytes_written += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);
				n_bytes_written += sizeof(size_t);

				std::cout << "n_bytes_written " << n_bytes_written <<std::endl;

				return;
			}


			if(lowest_level()){

				memcpy(p, &nRows, sizeof(int));
				p += sizeof(int);

				memcpy(p, &nCols, sizeof(int));
				p += sizeof(int);

				memcpy(p, &nRows_orig, sizeof(int));
				p += sizeof(int);

				memcpy(p, &nCols_orig, sizeof(int));
				p += sizeof(int);

				memcpy(p, &blocksize, sizeof(int));
				p += sizeof(int);

				//no children - write 4 zeros!
				size_t zero_size = 0;
				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);

				memcpy(p, &zero_size, sizeof(size_t));
				p += sizeof(size_t);

				memcpy(p, &submatrix[0], submatrix.size() * sizeof(Treal));
				p += submatrix.size() * sizeof(Treal);

				return;
			}

			size_t size_child0 = 0, size_child1 = 0, size_child2 = 0, size_child3 = 0;
			std::vector<char> buf_child0, buf_child1, buf_child2, buf_child3;

			if(children[0] != NULL){ // if child do not exist, its size will be zero!
				size_child0 = children[0]->get_size();
				buf_child0.resize(size_child0);
				children[0]->write_to_buffer(&buf_child0[0],size_child0);
			}

			if(children[1] != NULL){
				size_child1 = children[1]->get_size();
				buf_child1.resize(size_child1);
				children[1]->write_to_buffer(&buf_child1[0],size_child1);
			}

			if(children[2] != NULL){
				size_child2 = children[2]->get_size();
				buf_child2.resize(size_child2);
				children[2]->write_to_buffer(&buf_child2[0],size_child2);
			}

			if(children[3] != NULL){
				size_child3 = children[3]->get_size();
				buf_child3.resize(size_child3);
				children[3]->write_to_buffer(&buf_child3[0],size_child3);
			}


			memcpy(p, &nRows, sizeof(int));
			p += sizeof(int);

			memcpy(p, &nCols, sizeof(int));
			p += sizeof(int);

			memcpy(p, &nRows_orig, sizeof(int));
			p += sizeof(int);

			memcpy(p, &nCols_orig, sizeof(int));
			p += sizeof(int);

			memcpy(p, &blocksize, sizeof(int));
			p += sizeof(int);

			// write sizes of children even if they are zeros!
			memcpy(p, &size_child0, sizeof(size_t));
			p += sizeof(size_t);

			memcpy(p, &size_child1, sizeof(size_t));
			p += sizeof(size_t);

			memcpy(p, &size_child2, sizeof(size_t));
			p += sizeof(size_t);

			memcpy(p, &size_child3, sizeof(size_t));
			p += sizeof(size_t);


		   if(size_child0 > 0){ // if child size is > 0, write size and the child itself
				memcpy(p, &buf_child0[0],  size_child0);
				p += size_child0;
			}

			if(size_child1 > 0){
				memcpy(p, &buf_child1[0],  size_child1);
				p += size_child1;
			}

			if(size_child2 > 0){
				memcpy(p, &buf_child2[0],  size_child2);
				p += size_child2;
			}

			if(size_child3 > 0){
				memcpy(p, &buf_child3[0],  size_child3);
				p += size_child3;
			}

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::assign_from_buffer(char * dataBuffer, size_t const bufferSize) {

			const char *p = dataBuffer;

			int n_bytes_left_to_read = bufferSize;

			if (bufferSize < 5 * sizeof(int) + 4 * sizeof(size_t))
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::assign_from_buffer(): buffer too small.");


			memcpy(&nRows, p, sizeof(int));
			p += sizeof(int);
			n_bytes_left_to_read -= sizeof(int);

			memcpy(&nCols, p, sizeof(int));
			p += sizeof(int);
			n_bytes_left_to_read -= sizeof(int);

			memcpy(&nRows_orig, p, sizeof(int));
			p += sizeof(int);
			n_bytes_left_to_read -= sizeof(int);

			memcpy(&nCols_orig, p, sizeof(int));
			p += sizeof(int);
			n_bytes_left_to_read -= sizeof(int);

			memcpy(&blocksize, p, sizeof(int));
			p += sizeof(int);
			n_bytes_left_to_read -= sizeof(int);

			size_t child0_size, child1_size, child2_size, child3_size;

			memcpy(&child0_size, p, sizeof(size_t));
			p += sizeof(size_t);
			n_bytes_left_to_read -= sizeof(size_t);

			memcpy(&child1_size, p, sizeof(size_t));
			p += sizeof(size_t);
			n_bytes_left_to_read -= sizeof(size_t);

			memcpy(&child2_size, p, sizeof(size_t));
			p += sizeof(size_t);
			n_bytes_left_to_read -= sizeof(size_t);

			memcpy(&child3_size, p, sizeof(size_t));
			p += sizeof(size_t);
			n_bytes_left_to_read -= sizeof(size_t);

			this->resize(nRows_orig, nCols_orig);

			//check if buffer ended, if so, that was an empty matrix
			if(n_bytes_left_to_read == 0){
				std::cout << "That was an empty matrix" << std::endl;
				return;
			}

			if(child0_size > 0){
				if(children[0] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::assign_from_buffer(): non-null child exist!");

				char child0_buf[child0_size];
				memcpy(&child0_buf[0], p, child0_size);
				p += child0_size;
				n_bytes_left_to_read -= child0_size;

				children[0] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[0]->assign_from_buffer(&child0_buf[0], child0_size);
				children[0]->parent = this;

			}

			if(child1_size > 0){
				if(children[1] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::assign_from_buffer(): non-null child exist!");

				char child1_buf[child1_size];
				memcpy(&child1_buf[0], p, child1_size);
				p += child1_size;
				n_bytes_left_to_read -= child1_size;

				children[1] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[1]->assign_from_buffer(&child1_buf[0], child1_size);
				children[1]->parent = this;

			}


			if(child2_size > 0){
				if(children[2] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::assign_from_buffer(): non-null child exist!");

				char child2_buf[child2_size];
				memcpy(&child2_buf[0], p, child2_size);
				p += child2_size;
				n_bytes_left_to_read -= child2_size;

				children[2] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[2]->assign_from_buffer(&child2_buf[0], child2_size);
				children[2]->parent = this;

			}


			if(child3_size > 0){
				if(children[3] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::assign_from_buffer(): non-null child exist!");

				char child3_buf[child3_size];
				memcpy(&child3_buf[0], p, child3_size);
				p += child3_size;
				n_bytes_left_to_read -= child3_size;

				children[3] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[3]->assign_from_buffer(&child3_buf[0], child3_size);
				children[3]->parent = this;

			}

			// at this point, if n_bytes_left_to_read is 0, then we are done, if not, then ot was a leaf matrix!
			if(n_bytes_left_to_read == 0){
				return;
			}
			else{

				//std::cout << "Leaf matrix read " << std::endl;
				assert(n_bytes_left_to_read == nRows * nCols * sizeof(Treal));
				submatrix.resize(nRows * nCols);

				memcpy(&submatrix[0], p, nRows * nCols * sizeof(Treal));
				p += nRows * nCols * sizeof(Treal);

			}
		}

	template<class Treal>
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::copy(const HierarchicalBlockSparseMatrixSmartPointers<Treal> &other, size_t* no_of_resizes) {

			if(this == &other) return; // no need to copy itself to itself

			if(other.empty()){
				this->clear();
				return;
			}

			set_params(other.get_params());
			resize(other.nRows_orig,other.nCols_orig, no_of_resizes);

			if(other.lowest_level()){
				memcpy(&submatrix[0], &(other.submatrix[0]), sizeof(Treal) * other.submatrix.size());
				return;
			}

			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::copy(): non-null child exist!");

				if(other.children[i] != NULL){
					children[i] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
					children[i]->copy(*other.children[i], no_of_resizes);
					children[i]->parent = this;
				}


			}


		}

	template<class Treal>
		void HierarchicalBlockSparseMatrixSmartPointers<Treal>::add_scaled_identity(const HierarchicalBlockSparseMatrixSmartPointers<Treal> & other, Treal alpha) {

			if(other.empty()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::add_scaled_identity(): empty matrix as input!");
			}

			if(other.nRows != other.nCols)
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrixSmartPointers::add_scaled_identity(): non-square matrix as input!");

			this->clear();

			this->set_params(other.get_params());

			this->resize(other.nRows_orig,other.nCols_orig);

			if(other.lowest_level()){
				memcpy(&submatrix[0], &(other.submatrix[0]), sizeof(Treal) * other.submatrix.size());
				for(int i = 0; i < nRows; ++i){
					submatrix[i*nRows + i] += alpha;
				}

				return;
			}

			//non-leaf case
			if(other.children[1] != NULL){
				children[1] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[1]->parent = this;
				children[1]->copy(*(other.children[1]));
			}

			if(other.children[2] != NULL){
				children[2] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[2]->parent = this;
				children[2]->copy(*(other.children[2]));
			}


			if(other.children[0] != NULL){

				if(children[0] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child0 occured");

				children[0] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();

				children[0]->parent = this;

				children[0]->add_scaled_identity(*other.children[0],alpha);

			}
			else{

				if(children[0] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child0 occured");

				children[0] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[0]->parent = this;
				children[0]->set_params(get_params());
				children[0]->resize(nRows/2, nCols/2);

				std::vector<int> rows, cols;
				std::vector<Treal> vals;

				rows.resize(nRows/2);
				cols.resize(nRows/2);
				vals.resize(nRows/2);

				for(int i = 0; i < nRows/2; ++i){
					rows[i] = i;
					cols[i] = i;
					vals[i] = alpha;
				}

				children[0]->assign_from_vectors(rows, cols, vals);
			}


			if(other.children[3] != NULL){

				if(children[3] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child3 occured");

				children[3] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();

				children[3]->parent = this;

				children[3]->add_scaled_identity(*other.children[3],alpha);

			}
			else{

				if(children[3] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child3 occured");

				children[3] = std::make_shared<HierarchicalBlockSparseMatrixSmartPointers<Treal> >();
				children[3]->parent = this;
				children[3]->set_params(get_params());
				children[3]->resize(nRows/2, nCols/2);

				std::vector<int> rows, cols;
				std::vector<Treal> vals;

				rows.resize(nRows/2);
				cols.resize(nRows/2);
				vals.resize(nRows/2);

				for(int i = 0; i < nRows/2; ++i){
					rows[i] = i;
					cols[i] = i;
					vals[i] = alpha;
				}

				children[3]->assign_from_vectors(rows, cols, vals);
			}


		}
		
} /* end namespace hbsm */

#endif