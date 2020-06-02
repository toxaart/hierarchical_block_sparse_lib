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
#include <functional>
#include <memory>
#include <unordered_map>
#include <map>
#if BUILD_WITH_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaManager.h"
#include "CudaSyncPtr.h"
#endif

#define USE_BATCH_MULTIPLY 0

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
			int nRows_orig; // before 'virtual size' has been computed
			int nCols_orig; // before 'virtual size' has been computed
			int blocksize; // size of blocks at the lowest level (blocksize x blocksize)
			Treal frob_norm_squared_internal;
			std::shared_ptr<HierarchicalBlockSparseMatrix<real> > children[4]; // array of pointers to the next level.
			HierarchicalBlockSparseMatrix<real>* parent; // way to go to top level;
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

			struct GemmTriplet {
				const HierarchicalBlockSparseMatrix<real>* a;
				const HierarchicalBlockSparseMatrix<real>* b;
				const HierarchicalBlockSparseMatrix<real>* c;
				bool ta;
				bool tb;

				GemmTriplet() : a(NULL), b(NULL), c(NULL), ta(false), tb(false) {}
			};

			typedef std::unordered_multimap<std::string, GemmTriplet> BatchMapMultiply;


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

			const Treal *get_submatrix_ptr() const {
				if(lowest_level())return &submatrix[0];
				else return NULL;
			}

			Treal *get_submatrix_ptr_for_modification() {
				if(lowest_level()) return &submatrix[0];
				else return NULL;
			}

			//replacement for add() in multiplication function to reduce number of allocations!
			static void add_to_first(HierarchicalBlockSparseMatrix<Treal> & first, HierarchicalBlockSparseMatrix<Treal> & second,
													size_t* no_of_resizes = NULL);

			static void collide_trees(HierarchicalBlockSparseMatrix<Treal> & first, HierarchicalBlockSparseMatrix<Treal> & second);

			bool self_frob_block_trunc(Treal trunc_value);

	public:
			struct Params {
			  int blocksize;
			};

			HierarchicalBlockSparseMatrix():nRows(0), nCols(0),nRows_orig(0), nCols_orig(0), blocksize(-1), frob_norm_squared_internal(0.0){
				parent = NULL;
			}

			~HierarchicalBlockSparseMatrix(){
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
				     const std::vector<Treal> & values);
			Treal get_frob_squared() const;

			Treal get_frob_norm_squared_internal() const {return frob_norm_squared_internal;}

			void update_internal_info();

			size_t get_nnz() const;

			int get_depth() const;

			int expected_depth() const;

			// get values with indices kept in vectors rows and cols
			void get_values(const std::vector<int> & rows,
						  const std::vector<int> & cols,
						  std::vector<Treal> & values) const;

			void get_all_values(std::vector<int> & rows,
						  std::vector<int> & cols,
						  std::vector<Treal> & values) const;

			size_t get_size() const;

			void write_to_buffer ( char * dataBuffer, size_t const bufferSize ) const;
			void assign_from_buffer (const char * dataBuffer, size_t const bufferSize );

			static void allocate_work_buffers(int max_dimension, int max_blocksize){}

			void copy(const HierarchicalBlockSparseMatrix<Treal> & other, size_t* no_of_resizes = NULL);

			void add_scaled_identity( HierarchicalBlockSparseMatrix<Treal> const & other, Treal alpha);

			static void add(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> const & B,
													HierarchicalBlockSparseMatrix<Treal> & C,
													size_t* no_of_resizes = NULL);



			static void multiply(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
			HierarchicalBlockSparseMatrix<Treal>& C,
			size_t* no_of_block_multiplies = NULL,
			size_t* no_of_resizes = NULL);

			void rescale(HierarchicalBlockSparseMatrix<Treal> const& other, Treal alpha);

			static void inv_chol(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & Z);
			static void inv_chol_trunc(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & Z,
			Treal trunc_value){ throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::inv_chol_trunc: function not yet implemented."); }

			static void symm_multiply(HierarchicalBlockSparseMatrix<Treal> const & A, bool sA,HierarchicalBlockSparseMatrix<Treal> const & B, bool sB,
						HierarchicalBlockSparseMatrix<Treal> & C);

			static void symm_square(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & C);

			static void symm_rk(HierarchicalBlockSparseMatrix<Treal> const & A, bool transposed, HierarchicalBlockSparseMatrix<Treal> & C);

			static void transpose(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & C);

			void get_upper_triangle(HierarchicalBlockSparseMatrix<Treal> & A) const;

			static void adjust_sizes(HierarchicalBlockSparseMatrix<Treal> & A, const int nRows_other, const int nCols_other);

			Treal get_trace() const;

	    static void set_to_identity(HierarchicalBlockSparseMatrix<Treal> & A, int nRows);

			// very misleading name, it counts number of elements in diagonal blocks even if elements are zero, what matters is that block exists!
			size_t get_nnz_diag_lowest_level() const;

			void print() const{
				std::vector<int> rows, cols;
				std::vector<Treal> vals;
				get_all_values(rows, cols, vals);
				for(int i = 0; i < rows.size(); ++i){
					std::cout << rows[i] << " " << cols[i] << " " << vals[i] << std::endl;
				}
			}

			static void spamm(HierarchicalBlockSparseMatrix<Treal> const & A, bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, bool tB,
			HierarchicalBlockSparseMatrix<Treal>& C,
			const Treal tau,
			bool updated,
			size_t* no_of_block_multiplies = NULL,
			size_t* no_of_resizes = NULL);

			void random_blocks(size_t nnz_blocks);

			static int get_blocksize(Params const & param, int max_dimension){return param.blocksize;}

			void get_col_sums(std::vector<Treal> & res) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_col_sums: function not yet implemented."); }
			void get_col_sums_part(std::vector<Treal> & res, const int row_start, const int row_end) const {throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_col_sums_part: function not yet implemented.");}
			void get_row_sums(std::vector<Treal> & res) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_row_sums: function not yet implemented."); }
			void get_row_sums_part(std::vector<Treal> & res, const int col_start, const int col_end) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_row_sums_part: function not yet implemented."); }
			void get_diag(std::vector<Treal> & res) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_diag: function not yet implemented."); }
			void get_diag_part(std::vector<Treal> & res, int row_start, int row_end) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_diag_part: function not yet implemented."); }
			void get_frob_squared_of_error_matrix(std::vector<Treal> & frob_squared_of_error_matrix,
					  std::vector<Treal> const & trunc_values) const;

			void get_spectral_squared_of_error_matrix(std::vector<Treal> & spectral_squared_of_error_matrix,
					      std::vector<Treal> const & trunc_values,
					      int diag, bool tr) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_spectral_squared_of_error_matrix: function not yet implemented."); }

			Treal get_frob_squared_symm() const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_frob_squared_symm: function not yet implemented."); }

			void get_frob_squared_of_error_matrix_symm(std::vector<Treal> & frob_squared_of_error_matrix,
					       std::vector<Treal> const & trunc_values) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_frob_squared_of_error_matrix_symm: function not yet implemented."); }

			bool frob_block_trunc(HierarchicalBlockSparseMatrix<Treal> & matrix_truncated, Treal trunc_value) const;
			bool frob_block_trunc_symm(HierarchicalBlockSparseMatrix<Treal> & matrix_truncated, Treal trunc_value) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::frob_block_trunc_symm: function not yet implemented."); }
			void set_neg_to_zero(const HierarchicalBlockSparseMatrix<Treal> & other) { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_row_sums_part: function not yet implemented."); }
			static void max(HierarchicalBlockSparseMatrix<Treal> const & A,
		    HierarchicalBlockSparseMatrix<Treal> const & B,
		    HierarchicalBlockSparseMatrix<Treal> & C) { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::max: function not yet implemented."); }

			Treal spectral_norm(int diag = 0, bool tr = false) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::spectral_norm: function not yet implemented."); }

			void get_nnz_in_submatrix(std::vector<int> & rows,
			      std::vector<int> & cols,
			      std::vector<Treal> & vals,
			      int M1, int M2, int N1, int N2) const { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::spectral_norm: function not yet implemented."); }

			static void symm_rk_TN(HierarchicalBlockSparseMatrix<Treal> const & A,
			   HierarchicalBlockSparseMatrix<Treal> & C) { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::symm_rk_TN: function not yet implemented."); }

			static void symm_rk_NT(HierarchicalBlockSparseMatrix<Treal> const & A,
			   HierarchicalBlockSparseMatrix<Treal> & C) { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::symm_rk_NT: function not yet implemented."); }

			void symm_to_nosymm() { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::symm_to_nosymm: function not yet implemented."); }

			// Set lower triangle to zero
			void nosymm_to_symm() { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::nosymm_to_symm: function not yet implemented."); }

			static void submatrix_inv_chol(std::vector<real> const & A,
				   std::vector<real> & Z,
				   int n,
				   int blocksize) { throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::submatrix_inv_chol: function not yet implemented."); }


			static void remove_dummy_levels(HierarchicalBlockSparseMatrix<Treal> & C, int n_dummy_levels);

			bool check_if_sizes_correspond_to_level();

			bool check_if_matrix_is_consistent() const;

			static bool worth_to_multiply(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B,const bool tB);

			static bool worth_to_spamm(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, const Treal tau);

			static std::vector<unsigned long int> count_skips(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus, const bool & apply_truncation, const bool & apply_spamm);

			static std::vector<unsigned long int> sum_skips(std::vector<unsigned long int> const & curr_level_skips, std::vector<std::vector<unsigned long int> > const & sub_skips);

			Treal get_max_abs_value() const;

			static std::vector<Treal> get_errors_of_approx_multiplication(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus, const bool & apply_truncation, const bool & apply_spamm);

			static std::vector<Treal> get_spamm_errors(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus);

			static std::vector<Treal> sum_spamm_errors(std::vector<real> const & curr_level_errors, std::vector<std::vector<real> > const & sub_errors);

      static void anticommutator(HierarchicalBlockSparseMatrix<Treal> const & A,
      HierarchicalBlockSparseMatrix<Treal> const & B,
      HierarchicalBlockSparseMatrix<Treal> & C){throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::anticommutator: function not applicable.");};

      static void symm_product(HierarchicalBlockSparseMatrix<Treal> const & A,
      HierarchicalBlockSparseMatrix<Treal> const & B,
            const bool first_transposed,
      HierarchicalBlockSparseMatrix<Treal> & C){throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::symm_product: function not applicable.");};

			static void get_batches_multiply(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
			HierarchicalBlockSparseMatrix<Treal>& C,
		  BatchMapMultiply & batches,
			int dummy_levels);

			HierarchicalBlockSparseMatrix<Treal>* get_matrix_ptr_by_position_code(const std::string & position_code);

			static void multiply_batches(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
			HierarchicalBlockSparseMatrix<Treal>& C,
		  BatchMapMultiply & batches);
	};

	template<class T>
	void append_to_list(std::list<T> & list_append_to, std::list<T> & appended_list){
		list_append_to.splice(list_append_to.end(), appended_list);
	}

	template <typename T>
		std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
		{
		  assert(a.size() == b.size());

		  std::vector<T> result;
		  result.reserve(a.size());

		  std::transform(a.begin(), a.end(), b.begin(),
		                 std::back_inserter(result), std::plus<T>());
		  return result;
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
		int HierarchicalBlockSparseMatrix<Treal>::get_level() const  {

			if(parent == NULL)
				return 0;
			else{

				HierarchicalBlockSparseMatrix<Treal> *tmp = parent;
				int counter = 0;
				while(tmp != NULL){
					tmp = tmp->parent;
					counter++;
				}
				return counter;
			}

		}

	template<class Treal>
		int HierarchicalBlockSparseMatrix<Treal>::get_depth() const  {

			if(empty()) return 0;

			if(!children_exist()) return 0;

			int depths[4];
			if(children[0] != NULL) depths[0] = children[0]->get_depth();
			else depths[0] = 0;

			if(children[1] != NULL) depths[1] = children[1]->get_depth();
			else depths[1] = 0;

			if(children[2] != NULL) depths[2] = children[2]->get_depth();
			else depths[2] = 0;

			if(children[3] != NULL) depths[3] = children[3]->get_depth();
			else depths[3] = 0;

			return  1 + *std::max_element(depths,depths+4);


		}

	template<class Treal>
		int HierarchicalBlockSparseMatrix<Treal>::expected_depth() const  {

			if(nRows_orig <= blocksize && nCols_orig <= blocksize) return 0;

			int maxdim = ((nRows_orig > nCols_orig) ? nRows_orig : nCols_orig);

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

			return P;

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::resize(int nRows_, int nCols_, size_t* no_of_resizes) {
		assert(blocksize > 0);

		if(!empty()) clear();

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
		int HierarchicalBlockSparseMatrix<Treal>::get_n_rows() const  {
			if(parent == NULL)
				return nRows_orig;
			else{
				const HierarchicalBlockSparseMatrix<Treal> *tmp = this;
				while(tmp->parent != NULL){
					tmp = tmp->parent;
                  }
				return tmp->nRows_orig;
			}
		}

	template<class Treal>
		int HierarchicalBlockSparseMatrix<Treal>::get_n_cols() const  {
            if(parent == NULL)
				return nCols_orig;
			else{
				const HierarchicalBlockSparseMatrix<Treal> *tmp = this;
				while(tmp->parent != NULL){
					tmp = tmp->parent;
                  }
				return tmp->nCols_orig;
			}
		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::clear() {

			if(empty()){
				return;
			}

			if(lowest_level()){
				submatrix.clear();
				nRows = 0;
				nCols = 0;
				if(children_exist()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::clear: children exist on leaf level.");
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
		Treal HierarchicalBlockSparseMatrix<Treal>::get_frob_squared() const  {

			if(empty())
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_frob_squared: empty matrix occured.");

		    if(lowest_level()){

				Treal frob_squared = 0.0;
				for(int i = 0; i < submatrix.size(); ++i){
					frob_squared += submatrix[i] * submatrix[i];
				}
				return frob_squared;
			}
			else{

				Treal frob_squared = 0.0;

				for(int i = 0; i < 4; ++i){
					if(children[i] != NULL) frob_squared += children[i]->get_frob_squared();
				}

				return frob_squared;
			}

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors_general(const std::vector<int> & rows,
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

				// submatrix is already filled with zeros!
				//for(int i = 0; i < nCols*nRows; ++i){
				//	submatrix[i] = 0.0;
				//}

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

			int n_elements = rows.size();

            rows0.reserve(n_elements);
            cols0.reserve(n_elements);
            vals0.reserve(n_elements);

            rows1.reserve(n_elements);
            cols1.reserve(n_elements);
            vals1.reserve(n_elements);

            rows2.reserve(n_elements);
            cols2.reserve(n_elements);
            vals2.reserve(n_elements);

            rows3.reserve(n_elements);
            cols3.reserve(n_elements);
            vals3.reserve(n_elements);

			// the offset, when providing parts of vectors to the next level their "coordinates" will be shifted
			int offset = nRows/2;

			for(size_t i = 0; i < n_elements; ++i){

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
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: non-null child0 matrix occured.");
				}
				children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
			    children[0]->set_params(get_params());
				children[0]->resize(nRows / 2, nCols / 2);
				children[0]->parent = this;
                children[0]->assign_from_vectors_general(rows0, cols0, vals0, useMax,true);
			}

			if(vals1.size() > 0){
				if(children[1] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: non-null child1 matrix occured.");
				}
				children[1] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
			    children[1]->set_params(get_params());
				children[1]->resize(nRows / 2, nCols / 2);
				children[1]->parent = this;
                children[1]->assign_from_vectors_general(rows1, cols1, vals1, useMax,true);
			}

			if(vals2.size() > 0){
				if(children[2] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: non-null child2 matrix occured.");
				}
				children[2] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
			    children[2]->set_params(get_params());
				children[2]->resize(nRows / 2, nCols / 2);
				children[2]->parent = this;
                children[2]->assign_from_vectors_general(rows2, cols2, vals2, useMax,true);
			}

			if(vals3.size() > 0){
				if(children[3] != NULL){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors: non-null child3 matrix occured.");
				}
				children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
			    children[3]->set_params(get_params());
				children[3]->resize(nRows / 2, nCols / 2);
				children[3]->parent = this;
                children[3]->assign_from_vectors_general(rows3, cols3, vals3, useMax,true);
			}
		}


	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values) {
			 return HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors_general(rows, cols, values, false, false);
		}

  	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors_max(const std::vector<int> & rows,
				     const std::vector<int> & cols,
				     const std::vector<Treal> & values) {
			 return HierarchicalBlockSparseMatrix<Treal>::assign_from_vectors_general(rows, cols, values, true, false);
		}

	 template<class Treal>
		Treal HierarchicalBlockSparseMatrix<Treal>::get_single_value(int row, int col) const {


			if ( empty() )
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_single_value: empty matrix occured.");


			if(!(0 <= row && row < nRows_orig && 0 <= col && col < nCols_orig ))
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_single_value: bad index at highest level.");




			if(!(0 <= row && row < nRows && 0 <= col && col < nCols ) ){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_single_value: bad index.");
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

			throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_single_value: reached impossible point.");
			return 0.0;

		}

	template<class Treal>
		std::string HierarchicalBlockSparseMatrix<Treal>::get_position_code() const  {

            HierarchicalBlockSparseMatrix<Treal> *tmp = parent;

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
		int HierarchicalBlockSparseMatrix<Treal>::get_first_col_position() const  {

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
		int HierarchicalBlockSparseMatrix<Treal>::get_first_row_position() const  {

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
		bool HierarchicalBlockSparseMatrix<Treal>::on_right_boundary() const  {
            if((get_first_col_position() + nCols > get_n_cols()) && (get_first_col_position() < get_n_cols())) return true;
            else return false;
		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::on_bottom_boundary() const  {
            if((get_first_row_position() + nRows > get_n_rows()) && (get_first_row_position()< get_n_rows())) return true;
            else return false;
		}

	template<class Treal>
		size_t HierarchicalBlockSparseMatrix<Treal>::get_nnz() const  {
			if(empty())
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_nnz: empty matrix occured.");

		    if(lowest_level()){

                //this seems to be more logical!
				int nnz = 0;
				for(int i = 0; i < submatrix.size(); ++i){
					if(fabs(submatrix[i]) > 0.0) nnz += 1;
				}
				return nnz;

                /* DOES NOT WORK CORRECTLY, USE ABOVE VERSION
                if(on_bottom_boundary() && !on_right_boundary()){
                    return (get_n_rows() % blocksize) * nRows;
                }
                if(!on_bottom_boundary() && on_right_boundary()){
                    return (get_n_cols() % blocksize) * nCols;
                }

                if(on_bottom_boundary() && on_right_boundary()){
                     return (get_n_rows() % blocksize) * (get_n_cols() % blocksize);
                }

                if(!on_bottom_boundary() && !on_right_boundary()){
                     return submatrix.size();
                }*/


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
		void HierarchicalBlockSparseMatrix<Treal>::get_values(const std::vector<int> & rows,
						  const std::vector<int> & cols,
						  std::vector<Treal> & values) const  {

			if(empty()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_values: empty matrix occured.");
			}

			if(rows.size() != cols.size()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::get_values: bad sizes.");
			}

			if(rows.size() == 0) return;

			values.resize(rows.size());

			for(int i = 0; i < rows.size(); ++i){
				values[i] = get_single_value(rows[i],cols[i]);
			}
		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::get_all_values(std::vector<int> & rows,
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
		size_t HierarchicalBlockSparseMatrix<Treal>::get_size() const  {
			if(empty()) return 5 * sizeof(int) + 4 * sizeof(size_t) + 1 * sizeof(Treal);

			if(lowest_level()){
				return 5 * sizeof(int) + 4 * sizeof(size_t) + 1 * sizeof(Treal) + submatrix.size() * sizeof(Treal);
			}

			size_t totalsize = 5 * sizeof(int) + 4 * sizeof(size_t) + 1 * sizeof(Treal); // keep sizes of children in a row!


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
		void HierarchicalBlockSparseMatrix<Treal>::write_to_buffer(char * dataBuffer, size_t const bufferSize) const  {
			if(bufferSize < get_size())
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix<Treal>::write_to_buffer(): buffer too small.");

			char* p = dataBuffer;

			if(empty()){

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

				memcpy(p, &frob_norm_squared_internal, sizeof(Treal));
				p += sizeof(Treal);
				n_bytes_written += sizeof(Treal);

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

				memcpy(p, &frob_norm_squared_internal, sizeof(Treal));
				p += sizeof(Treal);

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

			memcpy(p, &frob_norm_squared_internal, sizeof(Treal));
			p += sizeof(Treal);

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
		void HierarchicalBlockSparseMatrix<Treal>::assign_from_buffer(const char * dataBuffer, size_t const bufferSize) {

			const char *p = dataBuffer;

			int n_bytes_left_to_read = bufferSize;

			if (bufferSize < 5 * sizeof(int) + 4 * sizeof(size_t))
			  throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::assign_from_buffer(): buffer too small.");


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

			memcpy(&frob_norm_squared_internal, p, sizeof(Treal));
			p += sizeof(Treal);
			n_bytes_left_to_read -= sizeof(Treal);

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
				return;
			}

			if(child0_size > 0){
				if(children[0] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::assign_from_buffer(): non-null child exist!");

				std::vector<char> child0_buf(child0_size);
				memcpy(&child0_buf[0], p, child0_size);
				p += child0_size;
				n_bytes_left_to_read -= child0_size;

				children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[0]->assign_from_buffer(&child0_buf[0], child0_size);
				children[0]->parent = this;

			}


			if(child1_size > 0){
				if(children[1] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::assign_from_buffer(): non-null child exist!");

				std::vector<char> child1_buf(child1_size);
				memcpy(&child1_buf[0], p, child1_size);
				p += child1_size;
				n_bytes_left_to_read -= child1_size;

				children[1] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[1]->assign_from_buffer(&child1_buf[0], child1_size);
				children[1]->parent = this;

			}


			if(child2_size > 0){
				if(children[2] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::assign_from_buffer(): non-null child exist!");

				std::vector<char> child2_buf(child2_size);
				memcpy(&child2_buf[0], p, child2_size);
				p += child2_size;
				n_bytes_left_to_read -= child2_size;

				children[2] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[2]->assign_from_buffer(&child2_buf[0], child2_size);
				children[2]->parent = this;

			}


			if(child3_size > 0){
				if(children[3] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::assign_from_buffer(): non-null child exist!");

				std::vector<char> child3_buf(child3_size);
				memcpy(&child3_buf[0], p, child3_size);
				p += child3_size;
				n_bytes_left_to_read -= child3_size;

				children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[3]->assign_from_buffer(&child3_buf[0], child3_size);
				children[3]->parent = this;

			}

			// at this point, if n_bytes_left_to_read is 0, then we are done, if not, then it was a leaf matrix!
			if(n_bytes_left_to_read == 0){
				return;
			}
			else{

				assert(n_bytes_left_to_read == nRows * nCols * sizeof(Treal));
				submatrix.resize(nRows * nCols);

				memcpy(&submatrix[0], p, nRows * nCols * sizeof(Treal));
				p += nRows * nCols * sizeof(Treal);
			}

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::copy(const HierarchicalBlockSparseMatrix<Treal> &other, size_t* no_of_resizes) {

			if(this == &other) return; // no need to copy itself to itself

			if(other.empty()){
				this->clear();
				return;
			}

			if(!empty()) clear();

			set_params(other.get_params());
			resize(other.nRows,other.nCols, no_of_resizes);
			nRows_orig = other.nRows_orig;
			nCols_orig = other.nCols_orig;


			if(other.lowest_level()){
				memcpy(&submatrix[0], &(other.submatrix[0]), sizeof(Treal) * other.submatrix.size());
				return;
			}

			frob_norm_squared_internal = other.frob_norm_squared_internal;

			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL)
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::copy(): non-null child exist!");

				if(other.children[i] != NULL){
					children[i] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					children[i]->copy(*other.children[i], no_of_resizes);
					children[i]->parent = this;
				}


			}


		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::add_scaled_identity(const HierarchicalBlockSparseMatrix<Treal> & other, Treal alpha) {

			if(other.empty()){
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): empty matrix as input!");
			}

			if(other.nRows != other.nCols)
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-square matrix as input!");

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
				children[1] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[1]->parent = this;
				children[1]->copy(*(other.children[1]));
			}

			if(other.children[2] != NULL){
				children[2] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				children[2]->parent = this;
				children[2]->copy(*(other.children[2]));
			}


			if(other.children[0] != NULL){

				if(children[0] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child0 occured");

				children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();

				children[0]->parent = this;

				children[0]->add_scaled_identity(*other.children[0],alpha);

			}
			else{

				if(children[0] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child0 occured");

				children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
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

				children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();

				children[3]->parent = this;

				children[3]->add_scaled_identity(*other.children[3],alpha);

			}
			else{

				if(children[3] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_scaled_identity(): non-null child3 occured");

				children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
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

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::add(HierarchicalBlockSparseMatrix<Treal> const & A,
				       HierarchicalBlockSparseMatrix<Treal> const & B,
				       HierarchicalBlockSparseMatrix<Treal> & C,
					   size_t* no_of_resizes){


            if(!C.empty()) C.clear();

			if(A.empty() && B.empty()){
				return;
			}


			if(A.nRows_orig != B.nRows_orig || A.nCols_orig != B.nCols_orig ){

				printf("A sizes %d %d, orig sizes %d %d \n", A.nRows, A.nCols, A.nRows_orig, A.nCols_orig);
				printf("A parent sizes %d %d, orig sizes %d %d \n", A.parent->nRows, A.parent->nCols, A.parent->nRows_orig, A.parent->nCols_orig);
				printf("B sizes %d %d, orig sizes %d %d \n", B.nRows, B.nCols, B.nRows_orig, B.nCols_orig);
				printf("B parent sizes %d %d, orig sizes %d %d \n", B.parent->nRows, B.parent->nCols, B.parent->nRows_orig, B.parent->nCols_orig);
				printf("A depth = %d, expected depth = %d \n", A.get_depth(), A.expected_depth());
				printf("B depth = %d, expected depth = %d \n", B.get_depth(), B.expected_depth());
				printf("A empty? %d \n", A.empty());
				printf("B empty? %d \n", B.empty());
				printf("A children exist? %d \n", A.children_exist());
				printf("B children exist? %d \n", B.children_exist());
				//printf("A: \n");
				//A.print();
				//printf("B: \n");
				//B.print();
				/*

				printf("before remove_dummy_levels called \n");
				HierarchicalBlockSparseMatrix<Treal> A_copy;
				A_copy.copy(*A.parent);
				printf("A_copy sizes %d %d, orig sizes %d %d \n", A_copy.nRows, A_copy.nCols, A_copy.nRows_orig, A_copy.nCols_orig);
				printf("A_copy depth = %d, expected depth = %d \n", A_copy.get_depth(), A_copy.expected_depth());
				A_copy.remove_dummy_levels();
				printf("after remove_dummy_levels called \n");
				printf("A_copy sizes %d %d, orig sizes %d %d \n", A_copy.nRows, A_copy.nCols, A_copy.nRows_orig, A_copy.nCols_orig);
				printf("A_copy depth = %d, expected depth = %d \n", A_copy.get_depth(), A_copy.expected_depth());
				printf("A_copy empty? %d \n", A_copy.empty());
				printf("A_copy children exist? %d \n", A_copy.children_exist());
				*/
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add(): matrices to add have different sizes!");
			}

            C.set_params(A.get_params());

            C.resize(A.nRows_orig,A.nCols_orig, no_of_resizes);

			if(A.lowest_level()){


				int blocksize = A.blocksize;

				int noOfElements = blocksize*blocksize;
				Treal const ONEreal = 1.0;
				int  const ONEint  = 1;

				//copy A to C
				memcpy(&C.submatrix[0], &A.submatrix[0], sizeof(Treal) * noOfElements);

				// Add 1.0 * B to C.
				axpy(&noOfElements, &ONEreal, &B.submatrix[0], &ONEint, &C.submatrix[0], &ONEint);

				return;
			}

			for(int i = 0; i < 4; ++i){

				if(A.children[i] != NULL && B.children[i] != NULL){

					if(C.children[i] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add(): matrix C has non-null child!");


					C.children[i] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					C.children[i]->parent = &C;

					add(*A.children[i], *B.children[i], *C.children[i], no_of_resizes);

				}

				if(A.children[i] == NULL && B.children[i] != NULL){

					if(C.children[i] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add(): matrix C has non-null child!");
					C.children[i] = B.children[i]; // smart pointer assignment. Now C.children[i] has parent in B, but it does not matter since they are of the  same size!

				}

				if(A.children[i] != NULL && B.children[i] == NULL){

					if(C.children[i] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add(): matrix C has non-null child!");
					C.children[i] = A.children[i]; // smart pointer assignment. Now C.children[i] has parent in A, but it does not matter since they are of the  same size!

				}

				if(A.children[i] == NULL && B.children[i] == NULL){

					if(C.children[i] != NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add(): matrix C has non-null child!");

					//nothing to do here, leave it NULL
				}

			}

			return;
		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::add_to_first(HierarchicalBlockSparseMatrix<Treal> & first, HierarchicalBlockSparseMatrix<Treal> & second,
													size_t* no_of_resizes){

			// at this point we are 100% sure that both matrices are non-empty!

			if(first.nRows_orig != second.nRows_orig || first.nCols_orig != second.nCols_orig ){

				printf("add_to_first: first size %d size orig %d, level %d second size %d, size orig %d, level %d \n", first.nRows, first.nRows_orig, first.get_level(), second.nRows, second.nRows_orig, second.get_level());
				printf("add_to_first: first parent size %d size orig %d level %d, second parent size %d, size orig %d level %d \n", first.parent->nRows, first.parent->nRows_orig, first.parent->get_level(), second.parent->nRows, second.parent->nRows_orig, second.parent->get_level());
				throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::add_to_first(): matrices to add have different sizes!");
			}

			if(first.lowest_level()){
				int blocksize = first.blocksize;
				int noOfElements = blocksize*blocksize;
				Treal const ONEreal = 1.0;
				int  const ONEint  = 1;

				// Add 1.0 * second to first.
				axpy(&noOfElements, &ONEreal, &second.submatrix[0], &ONEint, &first.submatrix[0], &ONEint);
				return;
			}

			for(int i = 0; i < 4; ++i){
				if(first.children[i] != NULL && second.children[i] == NULL){
					continue;
				}
				if(first.children[i] == NULL && second.children[i] != NULL) {
					first.children[i] = second.children[i];
					first.children[i]->parent = &first;
					continue;
				}
				if(first.children[i] != NULL && second.children[i] != NULL){
					add_to_first(*first.children[i],  *second.children[i]);
				}

			}


		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::check_if_sizes_correspond_to_level(){

			if(lowest_level()){
				if(nRows == nCols && nRows == blocksize) return true;
				else return false;
			}

			int maxdim = ((get_n_rows() > get_n_cols()) ? get_n_rows() : get_n_cols());

			int n_covers = maxdim / blocksize;
			if(maxdim % blocksize != 0) n_covers += 1;

			int P = 1;
			int two_to_power_P = 2;
			while(n_covers >  two_to_power_P){
				two_to_power_P *= 2;
				P += 1;
			}

			int top_virtual_size = blocksize * two_to_power_P;

			int my_level = get_level();

			int size_at_level = top_virtual_size;

			for(int i = 0; i < my_level; ++i){
				size_at_level = size_at_level / 2;
			}

			if(nRows == nCols && nRows == size_at_level){

				bool child0_ok = true, child1_ok = true, child2_ok = true, child3_ok = true;

				if(children[0]!=NULL) child0_ok = children[0]->check_if_sizes_correspond_to_level();
				if(children[1]!=NULL) child1_ok = children[1]->check_if_sizes_correspond_to_level();
				if(children[2]!=NULL) child2_ok = children[2]->check_if_sizes_correspond_to_level();
				if(children[3]!=NULL) child3_ok = children[3]->check_if_sizes_correspond_to_level();

				return child0_ok && child1_ok && child2_ok && child3_ok;
			}
			else return false;

		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::check_if_matrix_is_consistent() const {

			if(lowest_level()){
				if(!children_exist() && submatrix.size() > 0) return true;
				else return false;
			}

			if(!children_exist()) return false;

			bool child0_ok = true, child1_ok = true, child2_ok = true, child3_ok = true;

			if(children[0]!=NULL) child0_ok = children[0]->check_if_matrix_is_consistent() && children[0]->nRows < nRows;
			if(children[1]!=NULL) child1_ok = children[1]->check_if_matrix_is_consistent() && children[1]->nRows < nRows;
			if(children[2]!=NULL) child2_ok = children[2]->check_if_matrix_is_consistent() && children[2]->nRows < nRows;
			if(children[3]!=NULL) child3_ok = children[3]->check_if_matrix_is_consistent() && children[3]->nRows < nRows;

			return child0_ok && child1_ok && child2_ok && child3_ok;

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::remove_dummy_levels(HierarchicalBlockSparseMatrix<Treal> & C, int n_dummy_levels){

			if(n_dummy_levels == 0) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::remove_dummy_levels(): n_dummy_levels = 0!");

			HierarchicalBlockSparseMatrix<Treal> *ptr = &C;

			/*
			printf("remove_dummy_levels called with n_dummy_levels = %d \n", n_dummy_levels);
			printf("C has sizes %d, orig size %d \n", C.nRows, C.nRows_orig);
			printf("C at lowetst level? %d \n", C.lowest_level());
			printf("C children_exist? %d \n", C.children_exist());
			printf("C submatrix.size()? %d \n", C.submatrix.size());
			if(C.children[0] != NULL) printf("C child0 is not null \n");
			if(C.children[1] != NULL) printf("C child1 is not null \n");
			if(C.children[2] != NULL) printf("C child2 is not null \n");
			if(C.children[3] != NULL) printf("C child3 is not null \n");*/

			if(!C.children_exist() && !C.lowest_level()){
				int nRows = C.nRows;
				for(int i = 0; i < n_dummy_levels; ++i){
					nRows = nRows / 2;
				}
				C.nRows = nRows;
				C.nCols = nRows;
				return;
			}


			for(int i = 0; i < n_dummy_levels; ++i){
				if(ptr->children[0] != NULL) ptr = ptr->children[0].get();
				else throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::remove_dummy_levels(): error when squeezing matrix!");
			}

			// ok, we have found a child which is the "correct matrix", ptr is pointing to it

			//printf("C has sizes %d, orig size %d \n", C.nRows, C.nRows_orig);
			//printf("Coorrect matrix:  size %d, size orig %d, depth %d \n", ptr->nRows,  ptr->nRows_orig,  ptr->get_depth());
			//ptr->print();
			//printf("\n");

			// "replace" C with this "correct" matrix

			C.nRows = ptr->nRows;
			C.nCols = ptr->nCols;

			if(ptr->lowest_level()){
				C.submatrix.clear();
				C.submatrix.resize(C.blocksize*C.blocksize);
				std::memcpy(&C.submatrix[0], &ptr->submatrix[0], C.submatrix.size() * sizeof(Treal));
				for(int i = 0; i < 4; ++i) C.children[0] = nullptr;
			}
			else{
				for(int i = 0; i < 4; ++i) C.children[i] = ptr->children[i];
			}

			ptr->parent = &C;
			//printf("After replacement: C size %d, size orig %d, depth %d \n", C.nRows, C.nRows_orig, C.get_depth());

	}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::worth_to_multiply(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB){

			if(A.lowest_level() && B.lowest_level() ) return true;

			if(A.get_depth() < B.get_depth()){

				// worth if worth(A*B0) || worth(A*B1) || worth(A*B2) || worth(A*B3)

				bool B0 = false, B1 = false, B2 = false, B3 = false;

				if(B.children[0] != NULL) B0 = true;
				if(B.children[1] != NULL) B1 = true;
				if(B.children[2] != NULL) B2 = true;
				if(B.children[3] != NULL) B3 = true;

				if(B0 && worth_to_multiply(A, tA, *B.children[0],tB)) return true;
				if(tB){
					if(B2 && worth_to_multiply(A, tA, *B.children[2],tB)) return true;
					if(B1 && worth_to_multiply(A, tA, *B.children[1],tB)) return true;
				}
				else{
					if(B1 && worth_to_multiply(A, tA, *B.children[1],tB)) return true;
					if(B2 && worth_to_multiply(A, tA, *B.children[2],tB)) return true;
				}
				if(B3 && worth_to_multiply(A, tA, *B.children[3],tB)) return true;

				return false;

			}

			if(A.get_depth() > B.get_depth()){

				// worth if worth(A0*B) || worth(A1*B) || worth(A2*B) || worth(A3*B)

				bool A0 = false, A1 = false, A2 = false, A3 = false;

				if(A.children[0] != NULL) A0 = true;
				if(A.children[1] != NULL) A1 = true;
				if(A.children[2] != NULL) A2 = true;
				if(A.children[3] != NULL) A3 = true;

				if(A0 && worth_to_multiply(*A.children[0], tA, B,tB)) return true;
				if(tA){
					if(A2 && worth_to_multiply(*A.children[2], tA, B,tB)) return true;
					if(A1 && worth_to_multiply(*A.children[1], tA, B,tB)) return true;
				}
				else{
					if(A1 && worth_to_multiply(*A.children[1], tA, B,tB)) return true;
					if(A2 && worth_to_multiply(*A.children[2], tA, B,tB)) return true;
				}
				if(A3 && worth_to_multiply(*A.children[3], tA, B,tB)) return true;

				return false;

			}

			bool A0 = false, A1 = false, A2 = false, A3 = false, B0 = false, B1 = false, B2 = false, B3 = false;

			if(A.children[0] != NULL) A0 = true;
			if(A.children[1] != NULL) A1 = true;
			if(A.children[2] != NULL) A2 = true;
			if(A.children[3] != NULL) A3 = true;

			if(B.children[0] != NULL) B0 = true;
			if(B.children[1] != NULL) B1 = true;
			if(B.children[2] != NULL) B2 = true;
			if(B.children[3] != NULL) B3 = true;

			if(!tA && !tB){

				if(A0 && B0 && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)) return true;
				if(A2 && B1 && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)) return true;
				if(A1 && B0 && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)) return true;
				if(A3 && B1 && worth_to_multiply(*A.children[3], tA, *B.children[1], tB)) return true;
				if(A0 && B2 && worth_to_multiply(*A.children[0], tA, *B.children[2], tB)) return true;
				if(A2 && B3 && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)) return true;
				if(A1 && B2 && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)) return true;
				if(A3 && B3 && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)) return true;

				return false;

			}

			if(!tA && tB){

				if(A0 && B0 && worth_to_multiply(*A.children[0], tA, *B.children[0], tB) ) return true;
				if(A2 && B2 && worth_to_multiply(*A.children[2], tA, *B.children[2], tB) ) return true;
				if(A1 && B0 && worth_to_multiply(*A.children[1], tA, *B.children[0], tB) ) return true;
				if(A3 && B2 && worth_to_multiply(*A.children[3], tA, *B.children[2], tB) ) return true;
				if(A0 && B1 && worth_to_multiply(*A.children[0], tA, *B.children[1], tB) ) return true;
				if(A2 && B3 && worth_to_multiply(*A.children[2], tA, *B.children[3], tB) ) return true;
				if(A1 && B1 && worth_to_multiply(*A.children[1], tA, *B.children[1], tB) ) return true;
				if(A3 && B3 && worth_to_multiply(*A.children[3], tA, *B.children[3], tB) ) return true;

				return false;

			}

			if(tA && !tB){

				if(A0 && B0 && worth_to_multiply(*A.children[0], tA, *B.children[0], tB) ) return true;
				if(A1 && B1 && worth_to_multiply(*A.children[1], tA, *B.children[1], tB) ) return true;
				if(A2 && B0 && worth_to_multiply(*A.children[2], tA, *B.children[0], tB) ) return true;
				if(A3 && B1 && worth_to_multiply(*A.children[3], tA, *B.children[1], tB) ) return true;
				if(A0 && B2 && worth_to_multiply(*A.children[0], tA, *B.children[2], tB) ) return true;
				if(A1 && B3 && worth_to_multiply(*A.children[1], tA, *B.children[3], tB) ) return true;
				if(A2 && B2 && worth_to_multiply(*A.children[2], tA, *B.children[2], tB) ) return true;
				if(A3 && B3 && worth_to_multiply(*A.children[3], tA, *B.children[3], tB) ) return true;

				return false;

			}

			if(tA && tB){

				if(A0 && B0 && worth_to_multiply(*A.children[0], tA, *B.children[0], tB) ) return true;
				if(A1 && B2 && worth_to_multiply(*A.children[1], tA, *B.children[2], tB) ) return true;
				if(A2 && B0 && worth_to_multiply(*A.children[2], tA, *B.children[0], tB) ) return true;
				if(A3 && B2 && worth_to_multiply(*A.children[3], tA, *B.children[2], tB) ) return true;
				if(A0 && B1 && worth_to_multiply(*A.children[0], tA, *B.children[1], tB) ) return true;
				if(A1 && B3 && worth_to_multiply(*A.children[1], tA, *B.children[3], tB) ) return true;
				if(A2 && B1 && worth_to_multiply(*A.children[2], tA, *B.children[1], tB) ) return true;
				if(A3 && B3 && worth_to_multiply(*A.children[3], tA, *B.children[3], tB) ) return true;

				return false;

			}

			return true;

		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::worth_to_spamm(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, const Treal tau){

			if(A.get_frob_norm_squared_internal() * B.get_frob_norm_squared_internal() <= tau * tau) return false;

			if(A.lowest_level() && B.lowest_level()) return true;

			if(A.get_depth() < B.get_depth()){

				// worth if worth(A*B0) || worth(A*B1) || worth(A*B2) || worth(A*B3)

				bool B0 = false, B1 = false, B2 = false, B3 = false;

				if(B.children[0] != NULL) B0 = true;
				if(B.children[1] != NULL) B1 = true;
				if(B.children[2] != NULL) B2 = true;
				if(B.children[3] != NULL) B3 = true;

				if(B0 && worth_to_spamm(A, tA, *B.children[0],tB, tau)) return true;
				if(tB){
					if(B2 && worth_to_spamm(A, tA, *B.children[2], tB, tau)) return true;
					if(B1 && worth_to_spamm(A, tA, *B.children[1], tB, tau)) return true;
				}
				else{
					if(B1 && worth_to_spamm(A, tA, *B.children[1], tB, tau)) return true;
					if(B2 && worth_to_spamm(A, tA, *B.children[2], tB, tau)) return true;
				}
				if(B3 && worth_to_spamm(A, tA, *B.children[3], tB, tau)) return true;

				return false;

			}

			if(A.get_depth() > B.get_depth()){

				// worth if worth(A0*B) || worth(A1*B) || worth(A2*B) || worth(A3*B)

				bool A0 = false, A1 = false, A2 = false, A3 = false;

				if(A.children[0] != NULL) A0 = true;
				if(A.children[1] != NULL) A1 = true;
				if(A.children[2] != NULL) A2 = true;
				if(A.children[3] != NULL) A3 = true;

				if(A0 && worth_to_spamm(*A.children[0], tA, B, tB, tau)) return true;
				if(tA){
					if(A2 && worth_to_spamm(*A.children[2], tA, B, tB, tau)) return true;
					if(A1 && worth_to_spamm(*A.children[1], tA, B, tB, tau)) return true;
				}
				else{
					if(A1 && worth_to_spamm(*A.children[1], tA, B, tB, tau)) return true;
					if(A2 && worth_to_spamm(*A.children[2], tA, B, tB, tau)) return true;
				}
				if(A3 && worth_to_spamm(*A.children[3], tA, B, tB, tau)) return true;

				return false;

			}

			bool A0 = false, A1 = false, A2 = false, A3 = false, B0 = false, B1 = false, B2 = false, B3 = false;

			if(A.children[0] != NULL) A0 = true;
			if(A.children[1] != NULL) A1 = true;
			if(A.children[2] != NULL) A2 = true;
			if(A.children[3] != NULL) A3 = true;

			if(B.children[0] != NULL) B0 = true;
			if(B.children[1] != NULL) B1 = true;
			if(B.children[2] != NULL) B2 = true;
			if(B.children[3] != NULL) B3 = true;

			if(!tA && !tB){

				if(A0 && B0 && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau)) return true;
				if(A2 && B1 && worth_to_spamm(*A.children[2], tA, *B.children[1], tB, tau)) return true;
				if(A1 && B0 && worth_to_spamm(*A.children[1], tA, *B.children[0], tB, tau)) return true;
				if(A3 && B1 && worth_to_spamm(*A.children[3], tA, *B.children[1], tB, tau)) return true;
				if(A0 && B2 && worth_to_spamm(*A.children[0], tA, *B.children[2], tB, tau)) return true;
				if(A2 && B3 && worth_to_spamm(*A.children[2], tA, *B.children[3], tB, tau)) return true;
				if(A1 && B2 && worth_to_spamm(*A.children[1], tA, *B.children[2], tB, tau)) return true;
				if(A3 && B3 && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau)) return true;

				return false;

			}

			if(!tA && tB){

				if(A0 && B0 && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau) ) return true;
				if(A2 && B2 && worth_to_spamm(*A.children[2], tA, *B.children[2], tB, tau) ) return true;
				if(A1 && B0 && worth_to_spamm(*A.children[1], tA, *B.children[0], tB, tau) ) return true;
				if(A3 && B2 && worth_to_spamm(*A.children[3], tA, *B.children[2], tB, tau) ) return true;
				if(A0 && B1 && worth_to_spamm(*A.children[0], tA, *B.children[1], tB, tau) ) return true;
				if(A2 && B3 && worth_to_spamm(*A.children[2], tA, *B.children[3], tB, tau) ) return true;
				if(A1 && B1 && worth_to_spamm(*A.children[1], tA, *B.children[1], tB, tau) ) return true;
				if(A3 && B3 && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau) ) return true;

				return false;

			}

			if(tA && !tB){

				if(A0 && B0 && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau) ) return true;
				if(A1 && B1 && worth_to_spamm(*A.children[1], tA, *B.children[1], tB, tau) ) return true;
				if(A2 && B0 && worth_to_spamm(*A.children[2], tA, *B.children[0], tB, tau) ) return true;
				if(A3 && B1 && worth_to_spamm(*A.children[3], tA, *B.children[1], tB, tau) ) return true;
				if(A0 && B2 && worth_to_spamm(*A.children[0], tA, *B.children[2], tB, tau) ) return true;
				if(A1 && B3 && worth_to_spamm(*A.children[1], tA, *B.children[3], tB, tau) ) return true;
				if(A2 && B2 && worth_to_spamm(*A.children[2], tA, *B.children[2], tB, tau) ) return true;
				if(A3 && B3 && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau) ) return true;

				return false;

			}

			if(tA && tB){

				if(A0 && B0 && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau) ) return true;
				if(A1 && B2 && worth_to_spamm(*A.children[1], tA, *B.children[2], tB, tau) ) return true;
				if(A2 && B0 && worth_to_spamm(*A.children[2], tA, *B.children[0], tB, tau) ) return true;
				if(A3 && B2 && worth_to_spamm(*A.children[3], tA, *B.children[2], tB, tau) ) return true;
				if(A0 && B1 && worth_to_spamm(*A.children[0], tA, *B.children[1], tB, tau) ) return true;
				if(A1 && B3 && worth_to_spamm(*A.children[1], tA, *B.children[3], tB, tau) ) return true;
				if(A2 && B1 && worth_to_spamm(*A.children[2], tA, *B.children[1], tB, tau) ) return true;
				if(A3 && B3 && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau) ) return true;

				return false;

			}

			return true;

		}


	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::multiply(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
                        HierarchicalBlockSparseMatrix<Treal>& C, size_t* no_of_block_multiplies, size_t* no_of_resizes){

#if USE_BATCH_MULTIPLY

			if(!tA && !tB){

				std::unordered_multimap<std::string, GemmTriplet> batches;

				int AM = A.nRows_orig;
				int AN = A.nCols_orig;

				int BM = B.nRows_orig;
				int BN = B.nCols_orig;

				size_t max_n_muls = ( (AM / A.blocksize) + 1) * ( (AN / A.blocksize) + 1) * ( (BN / A.blocksize) + 1);

				batches.reserve(max_n_muls);

				get_batches_multiply(A, false, B, false, C, batches, 0);

				multiply_batches(A, false, B, false, C, batches);

				if(no_of_block_multiplies != NULL){
					*no_of_block_multiplies = batches.size();
				}

				return;

			}

#endif

			/*
			if(A.nRows < B.nRows){ // A is to be adjusted
			    HierarchicalBlockSparseMatrix<Treal>  A_alias;
				A_alias.copy(A);
				adjust_sizes(A_alias,B.nRows, B.nCols);
				multiply(A_alias, tA, B, tB, C, no_of_block_multiplies, no_of_resizes);
				return;
			}
			if(A.nRows > B.nRows){ // B is to be adjusted
				HierarchicalBlockSparseMatrix<Treal>  B_alias;
				B_alias.copy(B);
				adjust_sizes(B_alias, A.nRows, A.nCols);
				multiply(A, tA, B_alias, tB, C, no_of_block_multiplies, no_of_resizes);
				return;
			}	*/



			if(A.get_depth() < B.get_depth()){

				if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): non-empty matrix to write result!");

				C.set_params(A.get_params());

				if(!tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(B.nRows_orig, B.nCols_orig, no_of_resizes);
				}

				if(!tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(B.nRows_orig, B.nRows_orig, no_of_resizes);
				}

				if(tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(B.nCols_orig, B.nCols_orig, no_of_resizes);
				}

				if(tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig, no_of_resizes);
                    C.resize(B.nCols_orig, B.nRows_orig, no_of_resizes);
				}

				if(!worth_to_multiply(A,tA,B,tB)){
					return;
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB0;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB1;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB2;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB3;

				if(B.children[0] != NULL && worth_to_multiply(A, tA, *B.children[0], tB)){
					AxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(A, tA, *B.children[0], tB, *AxB0, no_of_block_multiplies, no_of_resizes);
					C.children[0] = AxB0;
					C.children[0]->parent = &C;
				}


				if(tB){
					if(B.children[2] != NULL  && worth_to_multiply(A, tA, *B.children[2], tB)){
						AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(A, tA, *B.children[2], tB, *AxB1, no_of_block_multiplies, no_of_resizes);
						C.children[1] = AxB1;
						C.children[1]->parent = &C;
					}
				}
				else{
					if(B.children[1] != NULL  && worth_to_multiply(A, tA, *B.children[1], tB)){
						AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(A, tA, *B.children[1], tB, *AxB1, no_of_block_multiplies, no_of_resizes);
						C.children[1] = AxB1;
						C.children[1]->parent = &C;
					}
				}


				if(tB){
					if(B.children[1] != NULL  && worth_to_multiply(A, tA, *B.children[1], tB))	{
						AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(A, tA, *B.children[1], tB, *AxB2, no_of_block_multiplies, no_of_resizes);
						C.children[2] = AxB2;
						C.children[2]->parent = &C;
					}
				}
				else{
					if(B.children[2] != NULL  && worth_to_multiply(A, tA, *B.children[2], tB)){
						AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(A, tA, *B.children[2], tB, *AxB2, no_of_block_multiplies, no_of_resizes);
						C.children[2] = AxB2;
						C.children[2]->parent = &C;
					}
				}

				if(B.children[3] != NULL  && worth_to_multiply(A, tA, *B.children[3], tB)){
					AxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(A, tA, *B.children[3], tB, *AxB3, no_of_block_multiplies, no_of_resizes);
					C.children[3] = AxB3;
					C.children[3]->parent = &C;
				}


				return;


			}


			if(A.get_depth() > B.get_depth()){

				if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): non-empty matrix to write result!");

				C.set_params(A.get_params());

				if(!tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(A.nRows_orig, A.nCols_orig, no_of_resizes);
				}

				if(!tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(A.nRows_orig, A.nRows_orig, no_of_resizes);
				}

				if(tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(A.nCols_orig, A.nCols_orig, no_of_resizes);
				}

				if(tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(A.nCols_orig, A.nRows_orig, no_of_resizes);
				}

				if(!worth_to_multiply(A,tA,B,tB)){
					return;
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB;

				if(A.children[0] != NULL && worth_to_multiply(*A.children[0], tA, B, tB)){
					A0xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, B, tB, *A0xB, no_of_block_multiplies, no_of_resizes);
					C.children[0] = A0xB;
					C.children[0]->parent = &C;
				}

				if(tA){
					if(A.children[2] != NULL && worth_to_multiply(*A.children[2], tA, B, tB)){
						A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
						C.children[1] = A1xB;
						C.children[1]->parent = &C;
					}
				}
				else{
					if(A.children[1] != NULL && worth_to_multiply(*A.children[1], tA, B, tB)){
						A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[1], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
						C.children[1] = A1xB;
						C.children[1]->parent = &C;
					}
				}

				if(tA){
					if(A.children[1] != NULL && worth_to_multiply(*A.children[1], tA, B, tB)){
						A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[1], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
						C.children[2] = A2xB;
						C.children[2]->parent = &C;
					}
				}
				else{
					if(A.children[2] != NULL && worth_to_multiply(*A.children[2], tA, B, tB)){
						A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
						C.children[2] = A2xB;
						C.children[2]->parent = &C;
					}
				}

				if(A.children[3] != NULL && worth_to_multiply(*A.children[3], tA, B, tB)){
					A3xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, B, tB, *A3xB, no_of_block_multiplies, no_of_resizes);
					C.children[3] = A3xB;
					C.children[3]->parent = &C;
				}

				return;

			}

			if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): non-empty matrix to write result!");

			C.set_params(A.get_params());


			if(!worth_to_multiply(A,tA,B,tB)){

				// case when both matrices are 0 level, but there is no sense in multiplying them, just resize C and return.
				assert(A.get_level() == 0 && B.get_level() == 0);

				if(!tA && !tB) C.resize(A.nRows_orig,B.nCols_orig, no_of_resizes);
				if(!tA && tB) C.resize(A.nRows_orig,B.nRows_orig, no_of_resizes);
				if(tA && !tB) C.resize(A.nCols_orig,B.nCols_orig, no_of_resizes);
				if(tA && tB) C.resize(A.nCols_orig,B.nRows_orig, no_of_resizes);

				return;
			}

			if(A.get_level() == 0 && no_of_resizes != NULL) *no_of_resizes = 0;
			if(A.get_level() == 0 && no_of_block_multiplies != NULL) *no_of_block_multiplies = 0;

			bool squeeze_needed = false;
			int n_dummy_levels = 0;

			if(!tA && !tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig)throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL)C.resize(A.nRows_orig,B.nCols_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nCols, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nCols_orig, no_of_resizes);



				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nRows_orig;
					C.nCols_orig = B.nCols_orig;
				}
			}

			if(!tA && tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nRows_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nRows_orig;
					C.nCols_orig = B.nRows_orig;
				}

			}

			if(tA && !tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nCols_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nCols, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nCols_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nCols_orig;
					C.nCols_orig = B.nCols_orig;
				}
			}

			if(tA && tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nRows_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nRows, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nRows_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nCols_orig;
					C.nCols_orig = B.nRows_orig;
				}
			}


			if(A.lowest_level()){

				if(!B.lowest_level()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): A is at lowest level, B is not!");

				const Treal ZERO = 0.0;
				const Treal ONE = 1.0;

				//at this point all submatrices are square and have equal sizes!
				if(!tA && !tB){

					int M = C.nRows;
					int N = C.nCols;
					int K = A.nCols;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::N().bt, Transpose::N().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}


				if(!tA && tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nCols;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::N().bt, Transpose::T().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}

				if(tA && !tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nRows;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::T().bt, Transpose::N().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}


				if(tA && tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nRows;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::T().bt, Transpose::T().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}

				if(no_of_block_multiplies != NULL) (*no_of_block_multiplies)++;

				return;
			}


			if(!tA && !tB){


				// C0 = A0xB0 + A2xB1
				// C1 = A1xB0 + A3xB1
				// C2 = A0xB2 + A2xB3
				// C3 = A1xB2 + A3xB3


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
					A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB1;
				if(A.children[2] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)){
					A2xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[1], tB, *A2xB1, no_of_block_multiplies, no_of_resizes);
				}

				if(A0xB0 != NULL && A2xB1 == NULL){C.children[0] = A0xB0; C.children[0]->parent = &C;}
				if(A0xB0 == NULL && A2xB1 != NULL){C.children[0] = A2xB1; C.children[0]->parent = &C;}
                if( A0xB0 != NULL && A2xB1 != NULL){
					//printf("mul: before 1st add \n");
                    add_to_first(*A0xB0, *A2xB1);
                   // printf("mul: after 1st add \n");
					C.children[0] = A0xB0;
                    C.children[0]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0;
				if(A.children[1] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)){
					A1xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB1;
				if(A.children[3] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[3], tA,*B.children[1], tB)){
					A3xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[1], tB, *A3xB1, no_of_block_multiplies, no_of_resizes);
				}


				if(A1xB0 != NULL && A3xB1 == NULL){ C.children[1] = A1xB0; C.children[1]->parent = &C;}
				if(A1xB0 == NULL && A3xB1 != NULL){ C.children[1] = A3xB1; C.children[1]->parent = &C;}
                if(A1xB0 != NULL && A3xB1 != NULL){
                   // printf("mul: before 2nd add \n");
					add_to_first(*A1xB0, *A3xB1);
                   // printf("mul: after 2nd add \n");
					C.children[1] = A1xB0;
					C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
				if(A.children[0] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[0], tA,*B.children[2], tB)){
					A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[2], tB, *A0xB2, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
				if(A.children[2] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)){
					A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3, no_of_block_multiplies, no_of_resizes);
				}

				if(A0xB2 != NULL && A2xB3 == NULL){C.children[2] = A0xB2; C.children[2]->parent = &C;}
				if(A0xB2 == NULL && A2xB3 != NULL){C.children[2] = A2xB3; C.children[2]->parent = &C;}
                if(A0xB2 != NULL && A2xB3 != NULL){
                   // printf("mul: before 3rd add \n");
					add_to_first(*A0xB2, *A2xB3);
                   // printf("mul: after 3rd add \n");
					C.children[2] = A0xB2;
					C.children[2]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB2;
				if(A.children[1] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)){
					A1xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[2], tB, *A1xB2, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
					A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3, no_of_block_multiplies, no_of_resizes);
				}

				if(A1xB2 != NULL && A3xB3 == NULL){C.children[3] = A1xB2; C.children[3]->parent = &C;}
				if(A1xB2 == NULL && A3xB3 != NULL){C.children[3] = A3xB3; C.children[3]->parent = &C;}
				if(A1xB2 != NULL && A3xB3 != NULL){
                    //printf("mul: before 4th add \n");
					add_to_first(*A1xB2, *A3xB3);
                   // printf("mul: after 4th add \n");
					C.children[3] = A1xB2;
					C.children[3]->parent = &C;
                }



				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(!tA && tB){

				// C0 = A0xB0^T + A2xB2^T
				// C1 = A1xB0^T + A3xB2^T
				// C2 = A0xB1^T + A2xB3^T
				// C3 = A1xB^T + A3xB3^T

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A0xB0T;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
					A0xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB2T;
				if(A.children[2] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[2], tB)){
					A2xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[2], tB, *A2xB2T, no_of_block_multiplies, no_of_resizes);
				}

				if(A0xB0T != NULL && A2xB2T == NULL){C.children[0] = A0xB0T; C.children[0]->parent = &C;}
				if(A0xB0T == NULL && A2xB2T != NULL){C.children[0] = A2xB2T; C.children[0]->parent = &C;}
                if(A0xB0T != NULL && A2xB2T != NULL){
                    add_to_first(*A0xB0T, *A2xB2T);
					C.children[0] = A0xB0T;
                    C.children[0]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0T;
				if(A.children[1] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)){
					A1xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB2T;
				if(A.children[3] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[2], tB)){
					A3xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[2], tB, *A3xB2T, no_of_block_multiplies, no_of_resizes);
				}

				if(A1xB0T != NULL && A3xB2T == NULL){C.children[1] = A1xB0T; C.children[1]->parent = &C;}
				if(A1xB0T == NULL && A3xB2T != NULL){C.children[1] = A3xB2T; C.children[1]->parent = &C;}
                if(A1xB0T != NULL && A3xB2T != NULL){
					add_to_first(*A1xB0T, *A3xB2T);
                    C.children[1] = A1xB0T;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB1T;
				if(A.children[0] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[1], tB)){
					A0xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[1], tB, *A0xB1T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3T;
				if(A.children[2] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)){
					A2xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3T, no_of_block_multiplies, no_of_resizes);
				}

				if(A0xB1T != NULL && A2xB3T == NULL){C.children[2] = A0xB1T; C.children[2]->parent = &C;}
				if(A0xB1T == NULL && A2xB3T != NULL){C.children[2] = A2xB3T; C.children[2]->parent = &C;}
                if(A0xB1T != NULL && A2xB3T != NULL){
					add_to_first(*A0xB1T, *A2xB3T);
                    C.children[2] = A0xB1T;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB1T;
				if(A.children[1] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[1], tB)){
					A1xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[1], tB, *A1xB1T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3T;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
					A3xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3T, no_of_block_multiplies, no_of_resizes);
				}

                if(A1xB1T != NULL && A3xB3T == NULL){C.children[3] = A1xB1T; C.children[3]->parent = &C;}
				if(A1xB1T == NULL && A3xB3T != NULL){C.children[3] = A3xB3T; C.children[3]->parent = &C;}
                if(A1xB1T != NULL && A3xB3T != NULL){
                    add_to_first(*A1xB1T, *A3xB3T);
					C.children[3] = A1xB1T;
                    C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(tA && !tB){
				// C0 = A0^TB0 + A1^TB1
				// C1 = A2^TB0 + A3^TB1
				// C2 = A0^TB2 + A1^TB3
				// C3 = A2^TB2 + A3^TB3

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
					A0TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[0], tB, *A0TxB0, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB1;
				if(A.children[1] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[1], tB)){
					A1TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[1], tB, *A1TxB1, no_of_block_multiplies, no_of_resizes);
				}

				if(A0TxB0 != NULL && A1TxB1 == NULL){C.children[0] = A0TxB0;  C.children[0]->parent = &C;}
			    if(A0TxB0 == NULL && A1TxB1 != NULL){C.children[0] = A1TxB1;  C.children[0]->parent = &C;}
                if(A0TxB0 != NULL && A1TxB1 != NULL){
					add_to_first(*A0TxB0,*A1TxB1);
                    C.children[0] = A0TxB0;
                    C.children[0]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0;
				if(A.children[2] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[0], tB)){
					A2TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[0], tB, *A2TxB0, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB1;
				if(A.children[3] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[1], tB)){
					A3TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[1], tB, *A3TxB1, no_of_block_multiplies, no_of_resizes);
				}

				if(A2TxB0 != NULL && A3TxB1 == NULL){C.children[1] = A2TxB0; C.children[1]->parent = &C;}
				if(A2TxB0 == NULL && A3TxB1 != NULL){C.children[1] = A3TxB1; C.children[1]->parent = &C;}
                if(A2TxB0 != NULL && A3TxB1 != NULL){
                    add_to_first(*A2TxB0,*A3TxB1);
					C.children[1] = A2TxB0;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB2;
				if(A.children[0] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[2], tB)){
					A0TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[2], tB, *A0TxB2, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3;
				if(A.children[1] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[3], tB)){
					A1TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[3], tB, *A1TxB3, no_of_block_multiplies, no_of_resizes);
				}

                if(A0TxB2 != NULL && A1TxB3 == NULL){C.children[2] = A0TxB2; C.children[2]->parent = &C;}
				if(A0TxB2 == NULL && A1TxB3 != NULL){C.children[2] = A1TxB3; C.children[2]->parent = &C;}
                if(A0TxB2 != NULL && A1TxB3 != NULL){
                    add_to_first(*A0TxB2,*A1TxB3);
					C.children[2] = A0TxB2;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB2;
				if(A.children[2] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[2], tB)){
					A2TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[2], tB, *A2TxB2, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
					A3TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[3], tB, *A3TxB3, no_of_block_multiplies, no_of_resizes);
				}

				if(A2TxB2 != NULL && A3TxB3 == NULL){C.children[3] = A2TxB2; C.children[3]->parent = &C;}
				if(A2TxB2 == NULL && A3TxB3 != NULL){C.children[3] = A3TxB3; C.children[3]->parent = &C;}
                if(A2TxB2 != NULL && A3TxB3 != NULL){
                    add_to_first(*A2TxB2,*A3TxB3);
					C.children[3] = A2TxB2;
                    C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(tA && tB){
				// C0 = A0^TB0^T + A1^TB2^T
				// C1 = A2^TB0^T + A3^TB2^T
				// C2 = A0^TB1^T + A1^TB3^T
				// C3 = A2^TB1^T + A3^TB3^T


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0T;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
					A0TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[0], tB, *A0TxB0T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB2T;
				if(A.children[1] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)){
					A1TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[2], tB, *A1TxB2T, no_of_block_multiplies, no_of_resizes);
				}

				if(A0TxB0T != NULL && A1TxB2T == NULL){C.children[0] = A0TxB0T; C.children[0]->parent = &C;}
				if(A0TxB0T == NULL && A1TxB2T != NULL){C.children[0] = A1TxB2T; C.children[0]->parent = &C;}
                if(A0TxB0T != NULL && A1TxB2T != NULL){
                    add_to_first(*A0TxB0T,*A1TxB2T);
					C.children[0] = A0TxB0T;
                    C.children[0]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0T;
				if(A.children[2] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[0], tB)){
					A2TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[0], tB, *A2TxB0T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB2T;
				if(A.children[3] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[2], tB)){
					A3TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[2], tB, *A3TxB2T, no_of_block_multiplies, no_of_resizes);
				}

				if(A2TxB0T != NULL && A3TxB2T == NULL){C.children[1] = A2TxB0T; C.children[1]->parent = &C;}
				if(A2TxB0T == NULL && A3TxB2T != NULL){C.children[1] = A3TxB2T; C.children[1]->parent = &C;}
                if(A2TxB0T != NULL && A3TxB2T != NULL){
                    add_to_first(*A2TxB0T,*A3TxB2T);
					C.children[1] = A2TxB0T;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB1T;
				if(A.children[0] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[1], tB)){
					A0TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[0], tA, *B.children[1], tB, *A0TxB1T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3T;
				if(A.children[1] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[3], tB)){
					A1TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[1], tA, *B.children[3], tB, *A1TxB3T, no_of_block_multiplies, no_of_resizes);
				}

				if(A0TxB1T != NULL && A1TxB3T == NULL){C.children[2] = A0TxB1T; C.children[2]->parent = &C;}
				if(A0TxB1T == NULL && A1TxB3T != NULL){C.children[2] = A1TxB3T; C.children[2]->parent = &C;}
                if(A0TxB1T != NULL && A1TxB3T != NULL){
                    add_to_first(*A0TxB1T,*A1TxB3T);
					C.children[2] = A0TxB1T;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB1T;
				if(A.children[2] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)){
					A2TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[2], tA, *B.children[1], tB, *A2TxB1T, no_of_block_multiplies, no_of_resizes);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3T;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
					A3TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*A.children[3], tA, *B.children[3], tB, *A3TxB3T, no_of_block_multiplies, no_of_resizes);
				}

				if(A2TxB1T != NULL && A3TxB3T == NULL){C.children[3] = A2TxB1T; C.children[3]->parent = &C;}
				if(A2TxB1T == NULL && A3TxB3T != NULL){C.children[3] = A3TxB3T; C.children[3]->parent = &C;}
                if(A2TxB1T != NULL && A3TxB3T != NULL){
                    add_to_first(*A2TxB1T,*A3TxB3T);
					C.children[3] = A2TxB1T;
                    C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;

			}



			return;
		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::adjust_sizes(HierarchicalBlockSparseMatrix<Treal> & A, const int nRows_other, const int nCols_other){


			std::vector<int> rows, cols;
			std::vector<Treal> vals;

			A.get_all_values(rows, cols, vals);

			//save original sizes
			int n_rows_orig = A.get_n_rows();
			int n_cols_orig = A.get_n_cols();

			A.clear(); // params are preserved

			A.resize(nRows_other, nCols_other);

			// original sizes are to be restored manually!
			if(A.parent == NULL){ // only top level!
				A.nRows_orig = n_rows_orig;
				A.nCols_orig = n_cols_orig;
			}

			A.assign_from_vectors(rows,cols,vals);
		}

	 template<class Treal>
        void HierarchicalBlockSparseMatrix<Treal>::random_blocks(size_t nnz_blocks){

            // assume matrix is resized and we are at the top level;

            int M = nRows_orig;
            int N = nCols_orig;

            int n_block_dim1 = M / blocksize;
            if(M % blocksize > 0) n_block_dim1 += 1;

            int n_block_dim2 = N / blocksize;
            if(N % blocksize > 0) n_block_dim2 += 1;

            int total_number_of_blocks = n_block_dim1 * n_block_dim2;

            if(nnz_blocks > total_number_of_blocks) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::random_blocks():too many blocks!");

            std::vector<int> v;
            v.resize(total_number_of_blocks);
            for(int i = 0; i < total_number_of_blocks; ++i) v[i] = i;

            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(v.begin(), v.end(), g);

            std::vector<int> row_indices, col_indices;
            std::vector<Treal> values;

            row_indices.resize(nnz_blocks * blocksize * blocksize);
            col_indices.resize(nnz_blocks * blocksize * blocksize);
            values.resize(nnz_blocks * blocksize * blocksize);

            size_t element_counter = 0;

            double lower_bound = -1.0;
            double upper_bound = 1.0;
            std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
            std::default_random_engine re;

            for(int k = 0; k < nnz_blocks; ++k){

				//std::cout << "V[k] = " << v[k] <<std::endl;
                int block_row_pos = blocksize * (v[k] / n_block_dim1);
                int block_col_pos = blocksize * (v[k] % n_block_dim1);

				//std::cout << "Block K = " << k << " is at " << block_row_pos << " " << block_col_pos << std::endl;

                for(int i = 0; i < blocksize*blocksize; ++i){

                    int element_row_pos = block_row_pos + i / blocksize;
                    int element_col_pos = block_col_pos + i % blocksize;

                    if(element_row_pos > get_n_rows()-1 || element_col_pos > get_n_cols()-1) continue;

                    values[element_counter] = unif(re);
                    row_indices[element_counter] = element_row_pos;
                    col_indices[element_counter] = element_col_pos;

                    element_counter++;
                }

            }

            row_indices.resize(element_counter);
            col_indices.resize(element_counter);
            values.resize(element_counter);



            assign_from_vectors(row_indices, col_indices, values);



        }

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::rescale(HierarchicalBlockSparseMatrix<Treal> const & other, Treal alpha){

			if(!empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::rescale(): non-empty matrix called this method!");

            if(other.empty()){
                clear();
                return;
            }

			set_params(other.get_params());
            resize(other.nRows_orig,other.nCols_orig);

            if(other.lowest_level()){
                for(int i = 0; i < nRows * nCols; ++i) submatrix[i] = other.submatrix[i] * alpha;
                return;
            }

            for(int i = 0; i < 4; ++i){
                if(children[i] != NULL)
                    throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::rescale(): non-null child exist!");

                if(other.children[i] != NULL){
                    children[i] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
                    children[i]->rescale(*other.children[i], alpha);
					children[i]->parent = this;
                }

            }

		}

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::inv_chol(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & Z){


			if(!Z.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::inv_chol(): non-empty matrix to write result!");
			if(A.nRows_orig != A.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::inv_chol(): call for non-square matrix!");

			Z.set_params(A.get_params());

			Z.resize(A.nRows_orig, A.nCols_orig);

			if(Z.lowest_level()){

				if(Z.get_first_row_position() > Z.get_n_rows()-1 || Z.get_first_col_position() > Z.get_n_cols()-1) return;

				int blocksize = Z.nRows;

				int n = Z.nRows;

				if(Z.on_bottom_boundary()) n = Z.get_n_rows() % Z.blocksize; // fill only necessary elements

				for(int i = 0; i < n; ++i) Z.submatrix[i*blocksize + i] = 1.0;

				Z.submatrix[0] = std::sqrt(1/A.submatrix[0]);
				for (int i = 1; i < n; i++) {
					Treal R;
					for (int j = 0; j < i; j++) {
						R = 0;

						for (int k = 0; k < n; k++)
							R += A.submatrix[k*blocksize+j] * Z.submatrix[i*blocksize+k];

						R *= Z.submatrix[j*blocksize+j];

						for (int k = 0; k < n; k++)
							Z.submatrix[i*blocksize+k] -= Z.submatrix[j*blocksize+k] * R;
					}
					R = 0;
					for (int k = 0; k < n; k++)
						R += A.submatrix[k*blocksize+i] * Z.submatrix[i*blocksize+k];

					R = std::sqrt(1/R);
					for (int k = 0; k < n; k++)
						Z.submatrix[i*blocksize+k] *= R;
				}

			   return;


			}

			// Z_00 = invChol(A_00)
			if(A.children[0] != NULL){
				Z.children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				Z.children[0]->parent = &Z;
				inv_chol(*A.children[0], *Z.children[0]);
			}


			if(Z.children[0] == NULL) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::inv_chol(): smth went wrong since Z_00 is NULL!");

			// R = Z_00^T * A_01
			std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > R;

			if(A.children[2] != NULL){
				R = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				multiply(*Z.children[0], true, *A.children[2], false, *R);
			}

			// T = Z_00 * R
			std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > T;
			if(R != NULL){
				T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				multiply(*Z.children[0], false, *R, false, *T);
			}


			// X = -T
			std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > X;
			if(T != NULL){
				X = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				X->rescale(*T,-1.0);
			}

			// Y = A_10 * X
			std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > Y;

			if(A.children[1] != NULL && X != NULL){
				Y = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				multiply(*A.children[1], false, *X, false, *Y);
			}


			// Q = Y + A_11
			std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > Q;
			if(A.children[3] != NULL){
				if(Y != NULL){
					Q = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					add(*Y, *A.children[3], *Q);
				}
				else{
					Q = A.children[3];
				}
			}
			else{
				Q = Y;
			}

			// Z_11 = invChol(Q)
			if(Z.children[3] == NULL){
				if(Q != NULL){
					Z.children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					Z.children[3]->parent = &Z;
					inv_chol(*Q, *Z.children[3]);
				}
			}
			else throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::inv_chol(): smth went wrong since Z_11 was not NULL!");

			// Z_01 = X * Z_11
			if(Z.children[2] == NULL){
				if(X != NULL){
					Z.children[2] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					multiply(*X, false, *Z.children[3], false, *Z.children[2]);
				}
			}
			else std::runtime_error("Error in HierarchicalBlockSparseMatrix::inv_chol(): smth went wrong since Z_01 was not NULL!");

			// Z_10 = 0
			assert(Z.children[1] == NULL);

			return;

		}

template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::symm_multiply(HierarchicalBlockSparseMatrix<Treal> const & A, bool sA,
						 HierarchicalBlockSparseMatrix<Treal> const & B, bool sB,
						 HierarchicalBlockSparseMatrix<Treal> & C){

				if(A.nRows < B.nRows){ // A is to be adjusted
					HierarchicalBlockSparseMatrix<Treal>  A_alias;
					A_alias.copy(A);
					adjust_sizes(A_alias,B.nRows, B.nCols);
					symm_multiply(A_alias, sA, B, sB, C);
					return;
				}
				if(A.nRows > B.nRows){ // B is to be adjusted
					HierarchicalBlockSparseMatrix<Treal>  B_alias;
					B_alias.copy(B);
					adjust_sizes(B_alias, A.nRows, A.nCols);
					symm_multiply(A, sA, B_alias, sB, C);
					return;
				}


				if (!sA && !sB) throw std::runtime_error("Error in hbsm::symm_multiply, Neither A nor B are symmetric, one and only one of them should be symmetric.");

			    if (sA && sB) throw std::runtime_error("Error in hbsm::symm_multiply, Both A and B are symmetric, one and only one of them should be symmetric.");

				if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::symm_multiply(): non-empty matrix to write result!");

				if(A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::symm_multiply(): matrices have bad sizes!");

				C.set_params(A.get_params());

				C.resize(A.nRows_orig,B.nCols_orig);



				if(A.lowest_level()){

					const Treal ZERO = 0.0;
					const Treal ONE = 1.0;

					//at this point all submatrices are square and have equal sizes!
					int M     = C.nRows;
					int N     = C.nCols;

					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();

					if(sA) symm("L", "U", &M, &N, &ONE, aptr, &M, bptr, &M, &ZERO, cptr, &M);
					if(sB) symm("R", "U", &M, &N, &ONE, bptr, &N, aptr, &M, &ZERO, cptr, &M);

					return;
				}


				if(sA){

					// same as regular multiplication, except that if A0 or A3 is involved, then recursive call is made

					// C0 = A0xB0 + A2xB1
					// C1 = A1xB0 + A3xB1
					// C2 = A0xB2 + A2xB3
					// C3 = A1xB2 + A3xB3

					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
					if(A.children[0] != NULL && B.children[0] != NULL){
						A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[0], sA, *B.children[0], sB, *A0xB0);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB1;
					if(A.children[2] != NULL && B.children[1] != NULL){
						A2xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], false, *B.children[1], false, *A2xB1);
					}


					if(A0xB0 != NULL && A2xB1 == NULL){C.children[0] = A0xB0;C.children[0]->parent = &C;}
					if(A0xB0 == NULL && A2xB1 != NULL){C.children[0] = A2xB1;C.children[0]->parent = &C;}
					if(A0xB0 != NULL && A2xB1 != NULL){
						add_to_first(*A0xB0, *A2xB1);
						C.children[0] = A0xB0;
						C.children[0]->parent = &C;
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0;
					if(A.children[2] != NULL && B.children[0] != NULL){
						A2TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], true, *B.children[0], false, *A2TxB0);
					}



					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB1;
					if(A.children[3] != NULL && B.children[1] != NULL){
						A3xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[3], sA, *B.children[1], sB,* A3xB1);
					}



					if(A2TxB0 != NULL && A3xB1 == NULL){C.children[1] = A2TxB0;C.children[1]->parent = &C;}
					if(A2TxB0 == NULL && A3xB1 != NULL){C.children[1] = A3xB1; C.children[1]->parent = &C;}
					if(A2TxB0 != NULL && A3xB1 != NULL){
						add_to_first(*A2TxB0, *A3xB1);
						C.children[1] = A2TxB0;
						C.children[1]->parent = &C;
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
					if(A.children[0] != NULL && B.children[2] != NULL){
						A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[0], sA, *B.children[2], sB, *A0xB2);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
					if(A.children[2] != NULL && B.children[3] != NULL){
						A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], false, *B.children[3], false, *A2xB3);
					}


					if(A0xB2 != NULL && A2xB3 == NULL){C.children[2] = A0xB2;C.children[2]->parent = &C;}
					if(A0xB2 == NULL && A2xB3 != NULL){C.children[2] = A2xB3;C.children[2]->parent = &C;}
					if(A0xB2 != NULL && A2xB3 != NULL){
						add_to_first(*A0xB2, *A2xB3);
						C.children[2] = A0xB2;
						C.children[2]->parent = &C;
					}

					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB2;
					if(A.children[2] != NULL && B.children[2] != NULL){
						A2TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], true, *B.children[2], false, *A2TxB2);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
					if(A.children[3] != NULL && B.children[3] != NULL){
						A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[3], sA, *B.children[3], sB, *A3xB3);
					}


				    if(A2TxB2 != NULL && A3xB3 == NULL){C.children[3] = A2TxB2;C.children[3]->parent = &C;}
					if(A2TxB2 == NULL && A3xB3 != NULL){C.children[3] = A3xB3;C.children[3]->parent = &C;}
					if(A2TxB2 != NULL && A3xB3 != NULL){
						add_to_first(*A2TxB2, *A3xB3);
						C.children[3] = A2TxB2;
						C.children[3]->parent = &C;
					}

					return;


				}

				if(sB){

					// same as regular multiplication, except that if B0 or B3 is involved, then recursive call is made
					// C0 = A0xB0 + A2xB1
					// C1 = A1xB0 + A3xB1
					// C2 = A0xB2 + A2xB3
					// C3 = A1xB2 + A3xB3


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
					if(A.children[0] != NULL && B.children[0] != NULL){
						A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[0], sA, *B.children[0], sB, *A0xB0);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB2T;
					if(A.children[2] != NULL && B.children[2] != NULL){
						A2xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[2], false, *B.children[2], true, *A2xB2T);
					}


					if(A0xB0 != NULL && A2xB2T == NULL){C.children[0] = A0xB0;C.children[0]->parent = &C;}
					if(A0xB0 == NULL && A2xB2T != NULL){C.children[0] = A2xB2T;C.children[0]->parent = &C;}
					if(A0xB0 != NULL && A2xB2T != NULL){
						add_to_first(*A0xB0, *A2xB2T);
						C.children[0] = A0xB0;
						C.children[0]->parent = &C;
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0;
					if(A.children[1] != NULL && B.children[0] != NULL){
						A1xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[1], sA, *B.children[0], sB, *A1xB0);
					}



					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB2T;
					if(A.children[3] != NULL && B.children[2] != NULL){
						A3xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[3], false, *B.children[2], true, *A3xB2T);
					}


					if(A1xB0 != NULL && A3xB2T == NULL){C.children[1] = A1xB0;C.children[1]->parent = &C;}
					if(A1xB0 == NULL && A3xB2T != NULL){C.children[1] = A3xB2T;C.children[1]->parent = &C;}
					if(A1xB0 != NULL && A3xB2T != NULL){
						add_to_first(*A1xB0, *A3xB2T);
						C.children[1] = A1xB0;
						C.children[1]->parent = &C;
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
					if(A.children[0] != NULL && B.children[2] != NULL){
						A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[0], false, *B.children[2], false, *A0xB2);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
					if(A.children[2] != NULL && B.children[3] != NULL){
						A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[2], sA, *B.children[3], sB, *A2xB3);
					}


					if(A0xB2 != NULL && A2xB3 == NULL){C.children[2] = A0xB2;C.children[2]->parent = &C;}
					if(A0xB2 == NULL && A2xB3 != NULL){C.children[2] = A2xB3;C.children[2]->parent = &C;}
					if(A0xB2 != NULL && A2xB3 != NULL){
						add_to_first(*A0xB2, *A2xB3);
						C.children[2] = A0xB2;
						C.children[2]->parent = &C;
					}

					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB2;
					if(A.children[1] != NULL && B.children[2] != NULL){
						A1xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						multiply(*A.children[1], false, *B.children[2], false, *A1xB2);
					}


					std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
					if(A.children[3] != NULL && B.children[3] != NULL){
						A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						symm_multiply(*A.children[3], sA, *B.children[3], sB, *A3xB3);
					}



					if(A1xB2 != NULL && A3xB3 == NULL){C.children[3] = A1xB2;C.children[3]->parent = &C;}
					if(A1xB2 == NULL && A3xB3 != NULL){C.children[3] = A3xB3;C.children[3]->parent = &C;}
					if(A1xB2 != NULL && A3xB3 != NULL){
						add_to_first(*A1xB2,*A3xB3);
						C.children[3] = A1xB2;
						C.children[3]->parent = &C;
					}

					return;

				}

				return;

		 }


	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::get_upper_triangle(HierarchicalBlockSparseMatrix<Treal> & A) const {

			if(get_n_rows() != get_n_cols()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_upper_triangle(): call for non-square matrix!");

			if(!A.empty())A.clear();

			A.set_params(get_params());
			A.resize(nRows_orig,nCols_orig);

			if(lowest_level()){

				int blocksize = A.nRows;
				int M = A.nRows;
				if(A.on_bottom_boundary()) M = A.get_n_rows() % A.blocksize; // fill only necessary elements

				for (int col = 0; col < M; col++){
					for (int row = 0; row <= col; row++) {
						 A.submatrix[col * nRows + row] = submatrix[col * nRows + row];
					}
			   }

			   return;
			}

			// children 0 and 3 are recursive, child 2 is just a copy
			if(children[0] != NULL){
				A.children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				A.children[0]->parent = &A;
				children[0]->get_upper_triangle(*A.children[0]);
			}

			// child2
			if(children[2] != NULL){
				A.children[2] = children[2];
				A.children[2]->parent = &A;
			}

			if(children[3] != NULL){
				A.children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				A.children[3]->parent = &A;
				children[3]->get_upper_triangle(*A.children[3]);
			}

			return;
		}


	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::symm_square(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & C){

			if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::symm_square(): non-empty matrix to write result!");

			if(A.nCols_orig != A.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::symm_square(): matrix has bad sizes!");

			C.set_params(A.get_params());

			C.resize(A.nRows_orig,A.nCols_orig);


			if(A.lowest_level()){

				int blocksize = A.nRows;
				int M = A.nRows;
				if(A.on_bottom_boundary()) M = A.get_n_rows() % A.blocksize; // fill only necessary elements

				for (int row = 0; row < M; row++){
				  for (int col = row; col < M; col++){
					 // Here we rely on that resize has set the elements of C to zero.

					 for (int ind = 0; ind < row; ind++){
						C.submatrix[row + col * blocksize] += A.submatrix[ind + row * blocksize] * A.submatrix[ind + col * blocksize];
					 }

					 for (int ind = row; ind <= col; ind++){
						C.submatrix[row + col * blocksize] += A.submatrix[row + ind * blocksize] * A.submatrix[ind + col * blocksize];
					 }

					 for (int ind = col + 1; ind < M; ind++) {
						C.submatrix[row + col * blocksize] += A.submatrix[row + ind * blocksize] * A.submatrix[col + ind * blocksize];
					 }
				  }
				}

				return;
			}


			// same as regular multiplication, except that if A0*A0 and A3*A3 are done recursively,
			// multiplication invilving A0 XOR A3 is computed using symm_multiply
			// rest - standard routine

			// C0 = A0xA0 + A2xA2^T
			// C1 = A2^TxA0 + A3xA2^T  - NO NEED IN THIS
			// C2 = A0xA2 + A2xA3
			// C3 = A2^TxA2 + A3xA3

			if(A.children[2] != NULL){

				// C0 = A0xA0 + A2xA2^T
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A0xA0;
				if(A.children[0] != NULL){
					A0xA0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					symm_square(*A.children[0],  *A0xA0);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xA2T;
				A2xA2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				multiply(*A.children[2], false, *A.children[2], true, *A2xA2T);

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xA2T_U;
				A2xA2T_U = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				A2xA2T->get_upper_triangle(*A2xA2T_U);

				if(A0xA0 != NULL && A2xA2T_U == NULL){C.children[0] = A0xA0; C.children[0]->parent = &C;}
				if(A0xA0 == NULL && A2xA2T_U != NULL){C.children[0] = A2xA2T_U; C.children[0]->parent = &C;}
				if(A0xA0 != NULL && A2xA2T_U != NULL){
					add_to_first(*A0xA0, *A2xA2T_U);
					C.children[0] = A0xA0;
					C.children[0]->parent = &C;
				}

				// C2 = A0xA2 + A2xA3
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A0xA2;
				if(A.children[0] != NULL){
					A0xA2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					symm_multiply(*A.children[0], true, *A.children[2], false, *A0xA2);
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A2xA3;
				if(A.children[3] != NULL){
					A2xA3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					symm_multiply(*A.children[2], false, *A.children[3], true, *A2xA3);
				}

				if(A0xA2 != NULL && A2xA3 == NULL){C.children[2] = A0xA2; C.children[2]->parent = &C;}
				if(A0xA2 == NULL && A2xA3 != NULL){C.children[2] = A2xA3; C.children[2]->parent = &C;}
				if(A0xA2 != NULL && A2xA3 != NULL){
					add_to_first(*A0xA2, *A2xA3);
					C.children[2] = A0xA2;
					C.children[2]->parent = &C;
				}

				// C3 = A2TxA2 + A3xA3
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxA2;
				A2TxA2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				multiply(*A.children[2], true, *A.children[2], false, *A2TxA2);

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxA2_U;
				A2TxA2_U = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				A2TxA2->get_upper_triangle(*A2TxA2_U);

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xA3;
				if(A.children[3] != NULL){
					A3xA3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					symm_square(*A.children[3], *A3xA3);
				}

				if(A2TxA2_U != NULL && A3xA3 == NULL){C.children[3] = A2TxA2_U; C.children[3]->parent = &C;}
				if(A2TxA2_U == NULL && A3xA3 != NULL){C.children[3] = A3xA3; C.children[3]->parent = &C;}
				if(A2TxA2_U != NULL && A3xA3 != NULL){
					add_to_first(*A2TxA2_U, *A3xA3);
					C.children[3] = A2TxA2_U;
					C.children[3]->parent = &C;
				}

			}
			else{ // so much less to compute

				// all summands which included A2 disappear now

				// C1 = A0 * A0
				if(A.children[0] != NULL){
					C.children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					C.children[0]->parent = &C;
					symm_square(*A.children[0],  *C.children[0]);
				}

				// C1 = NULL

				// C2 = NULL

				// C3 = A3*A3
				if(A.children[3] != NULL){
					C.children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					C.children[3]->parent = &C;
					symm_square(*A.children[3],  *C.children[3]);
				}

			}


			return;

		}

	template<class Treal>
        void HierarchicalBlockSparseMatrix<Treal>::symm_rk(HierarchicalBlockSparseMatrix<Treal> const & A, bool transposed, HierarchicalBlockSparseMatrix<Treal> & C){

            if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::symm_rk(): non-empty matrix to write result!");

            C.set_params(A.get_params());
            if(!transposed){
                std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > product;
				product = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
                multiply(A, false, A, true, *product);
                product->get_upper_triangle(C);

            }
            else{
                std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > product;
				product = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
                multiply(A, true, A, false, *product);
                product->get_upper_triangle(C);
            }

        }

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::transpose(HierarchicalBlockSparseMatrix<Treal> const & A, HierarchicalBlockSparseMatrix<Treal> & C){

			if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::transpose(): non-empty matrix to write result!");

			C.set_params(A.get_params());

			C.resize(A.nCols_orig,A.nRows_orig); // reverse order

			if(A.lowest_level()){
				int blocksize = A.blocksize;
				int M = A.nRows;
				int N = A.nCols;
				if(A.on_bottom_boundary()) M = A.get_n_rows() % A.blocksize; // fill only necessary elements
				if(A.on_right_boundary()) N = A.get_n_cols() % A.blocksize; // fill only necessary elements

				for(int col = 0; col < N; ++col){
					for(int row = 0; row < M; ++row){
						C.submatrix[row*blocksize + col] = A.submatrix[col*blocksize + row];
					}
				}

				return;
			}

			if(A.children[0] != NULL) {
				C.children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				transpose(*A.children[0], *C.children[0]);
			}

			if(A.children[1] != NULL){
				C.children[2] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				transpose(*A.children[1], *C.children[2]);
			}

			if(A.children[2] != NULL){
				C.children[1] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				transpose(*A.children[2], *C.children[1]);
			}

			if(A.children[3] != NULL){
				C.children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
				transpose(*A.children[3], *C.children[3]);
			}

			return;

		}

	template<class Treal>
        Treal HierarchicalBlockSparseMatrix<Treal>::get_trace() const{

            if(lowest_level()){

                Treal trace = 0.0;

                for(int i = 0; i < blocksize; ++i){
                    trace += submatrix[i * blocksize + i];
                }

                return trace;
            }

            Treal trace = 0.0;

            if(children[0] != NULL) trace += children[0]->get_trace();
            if(children[3] != NULL) trace += children[3]->get_trace();

            return trace;

        }

	template<class Treal>
        void HierarchicalBlockSparseMatrix<Treal>::set_to_identity(HierarchicalBlockSparseMatrix<Treal> & A, int nRows) {

            // assume params are set

            A.clear();

            A.resize(nRows, nRows);

            if(A.lowest_level()){

                int blocksize = A.blocksize;
				int n = A.nRows;
				if(A.on_bottom_boundary()) n = A.get_n_rows() % blocksize; // fill only necessary elements

                for(int i = 0; i < n; ++i) A.submatrix[i*blocksize + i] = 1.0;

                return;
            }

            int blocksize = A.blocksize;

            A.children[0] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
            A.children[0]->parent = &A;
            A.children[0]->set_params(A.get_params());
            set_to_identity(*A.children[0], A.nRows/2);

            if(A.children[0]->get_first_row_position() + A.children[0]->nRows >= A.get_n_rows() ) return;

            A.children[3] = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
            A.children[3]->parent = &A;
            A.children[3]->set_params(A.get_params());
            set_to_identity(*A.children[3], A.nRows/2);

            return;
        }


	template<class Treal>
        size_t HierarchicalBlockSparseMatrix<Treal>::get_nnz_diag_lowest_level()const{

            if(lowest_level()){

				size_t n = nRows;

				//if(on_bottom_boundary()) n = get_n_rows() % blocksize; // fill only necessary elements

                return n*n;
            }

            size_t nnz = 0;

            if(children[0] != NULL) nnz += children[0]->get_nnz_diag_lowest_level();
            if(children[3] != NULL) nnz += children[3]->get_nnz_diag_lowest_level();

            return nnz;

        }


	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::get_frob_squared_of_error_matrix(std::vector<Treal> & frob_squared_of_error_matrix,
					  std::vector<Treal> const & trunc_values) const{

				// frob_squared_of_error_matrix[i] is a frob norm squared of the matrix with all blocks which have frob norm > trunc_values[i] removed
				// IT CONTRADICTS WITH WHAT IS WRITTEN IN BLOCK SPARSE MATRIX LIB, PROBABLY A MISTAKE THERE!

				if(lowest_level()){

					Treal block_frob_norm_squared = get_frob_squared();
					Treal block_frob_norm = std::sqrt(block_frob_norm_squared);

					for(int i = 0; i < trunc_values.size(); ++i){
						if(block_frob_norm < trunc_values[i]) frob_squared_of_error_matrix[i] += block_frob_norm_squared;
					}


					return;
				}


			    std::vector<Treal> frob_squared_of_error_matrix_child0, frob_squared_of_error_matrix_child1, frob_squared_of_error_matrix_child2, frob_squared_of_error_matrix_child3;
				frob_squared_of_error_matrix_child0.resize(trunc_values.size());
				frob_squared_of_error_matrix_child1.resize(trunc_values.size());
				frob_squared_of_error_matrix_child2.resize(trunc_values.size());
				frob_squared_of_error_matrix_child3.resize(trunc_values.size());

				if(children[0] != NULL) children[0]->get_frob_squared_of_error_matrix(frob_squared_of_error_matrix_child0, trunc_values);
				if(children[1] != NULL) children[1]->get_frob_squared_of_error_matrix(frob_squared_of_error_matrix_child1, trunc_values);
				if(children[2] != NULL) children[2]->get_frob_squared_of_error_matrix(frob_squared_of_error_matrix_child2, trunc_values);
				if(children[3] != NULL) children[3]->get_frob_squared_of_error_matrix(frob_squared_of_error_matrix_child3, trunc_values);

				frob_squared_of_error_matrix.resize(trunc_values.size());

				for(int i = 0; i < trunc_values.size(); ++i){
					frob_squared_of_error_matrix[i] = frob_squared_of_error_matrix_child0[i] + frob_squared_of_error_matrix_child1[i] +
												frob_squared_of_error_matrix_child2[i] + frob_squared_of_error_matrix_child3[i];
				}


		  }

	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::update_internal_info() {

			if(empty()) return;

			frob_norm_squared_internal = 0.0;

			if(lowest_level()){

				frob_norm_squared_internal = get_frob_squared();

				return;
			}

			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL){
					children[i]->update_internal_info();
					frob_norm_squared_internal += children[i]->frob_norm_squared_internal;
				}
			}


	    }



	template<class Treal>
		void HierarchicalBlockSparseMatrix<Treal>::spamm(HierarchicalBlockSparseMatrix<Treal> const & A, bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, bool tB,
                        HierarchicalBlockSparseMatrix<Treal>& C, const Treal tau, bool updated, size_t* no_of_block_multiplies, size_t* no_of_resizes){



			if(!updated){ // this is to be in line with rules of Chunks and Tasks,
						  // since once a chunk is registered, its modification is not allowed

				printf("SpAMM called with updated = false, copying will be done! \n");

				HierarchicalBlockSparseMatrix<Treal>  A_alias;
				A_alias.copy(A);

				HierarchicalBlockSparseMatrix<Treal>  B_alias;
				B_alias.copy(B);

				A_alias.update_internal_info();
				B_alias.update_internal_info();

				spamm(A_alias, tA, B_alias, tB, C, tau, true, no_of_block_multiplies, no_of_resizes);
				return;

			}




			if(A.get_depth() < B.get_depth()){

				if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): non-empty matrix to write result!");

				C.set_params(A.get_params());

				if(!tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(B.nRows_orig, B.nCols_orig, no_of_resizes);
				}

				if(!tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(B.nRows_orig, B.nRows_orig, no_of_resizes);
				}

				if(tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(B.nCols_orig, B.nCols_orig, no_of_resizes);
				}

				if(tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(B.nCols_orig, B.nRows_orig, no_of_resizes);
				}

				if(!worth_to_spamm(A,tA,B,tB,tau)) return;

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB0;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB1;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB2;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB3;

				if(B.children[0] != NULL && worth_to_spamm(A, tA, *B.children[0], tB, tau)){
					AxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					spamm(A, tA, *B.children[0], tB, *AxB0, tau, true, no_of_block_multiplies, no_of_resizes);
				}


				if(tB){
					if(B.children[2] != NULL  && worth_to_spamm(A, tA, *B.children[2], tB, tau)){
						AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(A, tA, *B.children[2], tB, *AxB1, tau, true,  no_of_block_multiplies, no_of_resizes);
					}
				}
				else{
					if(B.children[1] != NULL  && worth_to_spamm(A, tA, *B.children[1], tB, tau)){
						AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(A, tA, *B.children[1], tB, *AxB1, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}


				if(tB){
					if(B.children[1] != NULL  && worth_to_spamm(A, tA, *B.children[1], tB, tau))	{
						AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(A, tA, *B.children[1], tB, *AxB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}
				else{
					if(B.children[2] != NULL  && worth_to_spamm(A, tA, *B.children[2], tB, tau)){
						AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(A, tA, *B.children[2], tB, *AxB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(B.children[3] != NULL  && worth_to_spamm(A, tA, *B.children[3], tB, tau)){
					AxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					spamm(A, tA, *B.children[3], tB, *AxB3, tau, true, no_of_block_multiplies, no_of_resizes);
				}

				if(AxB0 != NULL){
					C.children[0] = AxB0;
					C.children[0]->parent = &C;
				}

				if(AxB1 != NULL){
					C.children[1] = AxB1;
					C.children[1]->parent = &C;
				}

				if(AxB2 != NULL){
					C.children[2] = AxB2;
					C.children[2]->parent = &C;
				}

				if(AxB3 != NULL){
					C.children[3] = AxB3;
					C.children[3]->parent = &C;
				}

				return;

			}


			if(A.get_depth() > B.get_depth()){

				if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): non-empty matrix to write result!");

				C.set_params(A.get_params());

				if(!tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(A.nRows_orig, A.nCols_orig, no_of_resizes);
				}

				if(!tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(A.nRows_orig, A.nRows_orig, no_of_resizes);
				}

				if(tA && !tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig, no_of_resizes);
                    else C.resize(A.nCols_orig, A.nCols_orig, no_of_resizes);
				}

				if(tA && tB){
					if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
					if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig, no_of_resizes);
                    else C.resize(A.nCols_orig, A.nRows_orig, no_of_resizes);
				}

				if(!worth_to_spamm(A,tA,B,tB,tau)) return;

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB;
				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB;

				if(A.children[0] != NULL && worth_to_spamm(*A.children[0], tA, B, tB, tau)){
					A0xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					spamm(*A.children[0], tA, B, tB, *A0xB, tau, true, no_of_block_multiplies, no_of_resizes);
				}

				if(tA){
					if(A.children[2] != NULL && worth_to_spamm(*A.children[2], tA, B, tB, tau)){
						A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, B, tB, *A1xB, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}
				else{
					if(A.children[1] != NULL && worth_to_spamm(*A.children[1], tA, B, tB, tau)){
						A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, B, tB, *A1xB, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(tA){
					if(A.children[1] != NULL && worth_to_spamm(*A.children[1], tA, B, tB, tau)){
						A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, B, tB, *A1xB, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}
				else{
					if(A.children[2] != NULL && worth_to_spamm(*A.children[2], tA, B, tB, tau)){
						A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, B, tB, *A1xB, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A.children[3] != NULL && worth_to_spamm(*A.children[3], tA, B, tB, tau)){
					A3xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
					spamm(*A.children[3], tA, B, tB, *A3xB, tau, true, no_of_block_multiplies, no_of_resizes);
				}

				if(A0xB != NULL){
					C.children[0] = A0xB;
					C.children[0]->parent = &C;
				}

				if(A1xB != NULL){
					C.children[1] = A1xB;
					C.children[1]->parent = &C;
				}

				if(A2xB != NULL){
					C.children[2] = A2xB;
					C.children[2]->parent = &C;
				}

				if(A3xB != NULL){
					C.children[3] = A3xB;
					C.children[3]->parent = &C;
				}

				return;

			}



			if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): non-empty matrix to write result!");

			C.set_params(A.get_params());

			if(!worth_to_spamm(A,tA,B,tB,tau)){

				// case when both matrices are 0 level, but there is no sense in multiplying them, just resize C and return.
				assert(A.get_level() == 0 && B.get_level() == 0);

				if(!tA && !tB) C.resize(A.nRows_orig,B.nCols_orig, no_of_resizes);
				if(!tA && tB) C.resize(A.nRows_orig,B.nRows_orig, no_of_resizes);
				if(tA && !tB) C.resize(A.nCols_orig,B.nCols_orig, no_of_resizes);
				if(tA && tB) C.resize(A.nCols_orig,B.nRows_orig, no_of_resizes);

				return;

			}

			if(A.get_level() == 0 && no_of_resizes != NULL) *no_of_resizes = 0;
			if(A.get_level() == 0 && no_of_block_multiplies != NULL) *no_of_block_multiplies = 0;

			bool squeeze_needed = false;
			int n_dummy_levels = 0;

			if(!tA && !tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig)throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nCols_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nCols, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nCols_orig, no_of_resizes);


				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nRows_orig;
					C.nCols_orig = B.nCols_orig;
				}



			}

			if(!tA && tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nRows_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nRows_orig;
					C.nCols_orig = B.nRows_orig;
				}

			}

			if(tA && !tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nCols_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nCols, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nCols_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nCols_orig;
					C.nCols_orig = B.nCols_orig;
				}
			}

			if(tA && tB){
				if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): matrices have bad sizes!");
				if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nRows_orig, no_of_resizes);
				if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nRows, no_of_resizes);
				if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nRows_orig, no_of_resizes);

				if(C.nRows < A.nRows && C.nRows < B.nRows){
					squeeze_needed = true;
					int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

					int ratio = nRows / C.nRows;

					int P = 1;
					int two_to_power_P = 2;
					while(ratio >  two_to_power_P){
						two_to_power_P *= 2;
						P += 1;
					}
					n_dummy_levels = P;
					C.resize(nRows,nRows, no_of_resizes);
					C.nRows_orig = A.nCols_orig;
					C.nCols_orig = B.nRows_orig;
				}
			}


			if(A.lowest_level()){

				const Treal ZERO = 0.0;
				const Treal ONE = 1.0;

				//at this point all submatrices are square and have equal sizes!
				if(!tA && !tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nCols;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::N().bt, Transpose::N().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);

				}


				if(!tA && tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nCols;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::N().bt, Transpose::T().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}

				if(tA && !tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nRows;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::T().bt, Transpose::N().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}


				if(tA && tB){
					int M = C.nRows;
					int N = C.nCols;
					int K = A.nRows;
					const Treal *aptr = A.get_submatrix_ptr();
					const Treal *bptr = B.get_submatrix_ptr();
					Treal *cptr = C.get_submatrix_ptr_for_modification();
					int lda = A.nRows;
					int ldb = B.nRows;
					gemm(Transpose::T().bt, Transpose::T().bt, &M, &N, &K, &ONE, aptr, &lda, bptr, &ldb, &ZERO, cptr, &M);
				}

				if(no_of_block_multiplies != NULL) (*no_of_block_multiplies)++;
				return;
			}


			Treal tau_squared = tau * tau;

			if(!tA && !tB){

				// C0 = A0xB0 + A2xB1
				// C1 = A1xB0 + A3xB1
				// C2 = A0xB2 + A2xB3
				// C3 = A1xB2 + A3xB3


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() * B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[0], tB, *A0xB0, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB1;
				if(A.children[2] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[1], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A2xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[1], tB, *A2xB1, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				if(A0xB0 != NULL && A2xB1 == NULL){ C.children[0] = A0xB0; C.children[0]->parent = &C;}
				if(A0xB0 == NULL && A2xB1 != NULL){ C.children[0] = A2xB1; C.children[0]->parent = &C;}
                if( A0xB0 != NULL && A2xB1 != NULL){
					add_to_first(*A0xB0, *A2xB1);
					C.children[0] = A0xB0;
                    C.children[0]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0;
				if(A.children[1] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[0], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A1xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[0], tB, *A1xB0, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB1;
				if(A.children[3] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[1], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A3xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[1], tB, *A3xB1, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}


				if(A1xB0 != NULL && A3xB1 == NULL){ C.children[1] = A1xB0; C.children[1]->parent = &C;}
				if(A1xB0 == NULL && A3xB1 != NULL){ C.children[1] = A3xB1; C.children[1]->parent = &C;}
                if(A1xB0 != NULL && A3xB1 != NULL){
					add_to_first(*A1xB0, *A3xB1);
					C.children[1] = A1xB0;
					C.children[1]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
				if(A.children[0] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[2], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[2], tB, *A0xB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
				if(A.children[2] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[3], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[3], tB, *A2xB3, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				if(A0xB2 != NULL && A2xB3 == NULL){C.children[2] = A0xB2; C.children[2]->parent = &C;}
				if(A0xB2 == NULL && A2xB3 != NULL){C.children[2] = A2xB3; C.children[2]->parent = &C;}
                if(A0xB2 != NULL && A2xB3 != NULL){
					add_to_first(*A0xB2, *A2xB3);
					C.children[2] = A0xB2;
					C.children[2]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB2;
				if(A.children[1] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[2], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A1xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[2], tB, *A1xB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[3], tB, *A3xB3, tau, true, no_of_block_multiplies, no_of_resizes);
					}

				}

				if(A1xB2 != NULL && A3xB3 == NULL){C.children[3] = A1xB2; C.children[3]->parent = &C;}
				if(A1xB2 == NULL && A3xB3 != NULL){C.children[3] = A3xB3; C.children[3]->parent = &C;}
				if(A1xB2 != NULL && A3xB3 != NULL){
					add_to_first(*A1xB2, *A3xB3);
					C.children[3] = A1xB2;
					C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(!tA && tB){

				// C0 = A0xB0^T + A2xB2^T
				// C1 = A1xB0^T + A3xB2^T
				// C2 = A0xB1^T + A2xB3^T
				// C3 = A1xB^T + A3xB3^T

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A0xB0T;
				if(A.children[0] != NULL && B.children[0] != NULL  && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A0xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[0], tB, *A0xB0T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB2T;
				if(A.children[2] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[2], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A2xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[2], tB, *A2xB2T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A0xB0T != NULL && A2xB2T == NULL){C.children[0] = A0xB0T; C.children[0]->parent = &C;}
				if(A0xB0T == NULL && A2xB2T != NULL){C.children[0] = A2xB2T; C.children[0]->parent = &C;}
                if(A0xB0T != NULL && A2xB2T != NULL){
                    add_to_first(*A0xB0T, *A2xB2T);
					C.children[0] = A0xB0T;
                    C.children[0]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0T;
				if(A.children[1] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[0], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A1xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[0], tB, *A1xB0T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB2T;
				if(A.children[3] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[2], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A3xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[2], tB, *A3xB2T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A1xB0T != NULL && A3xB2T == NULL){C.children[1] = A1xB0T; C.children[1]->parent = &C;}
				if(A1xB0T == NULL && A3xB2T != NULL){C.children[1] = A3xB2T; C.children[1]->parent = &C;}
                if(A1xB0T != NULL && A3xB2T != NULL){
					add_to_first(*A1xB0T, *A3xB2T);
                    C.children[1] = A1xB0T;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB1T;
				if(A.children[0] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[1], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A0xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[1], tB, *A0xB1T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3T;
				if(A.children[2] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[3], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A2xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[3], tB, *A2xB3T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A0xB1T != NULL && A2xB3T == NULL){C.children[2] = A0xB1T; C.children[2]->parent = &C;}
				if(A0xB1T == NULL && A2xB3T != NULL){C.children[2] = A2xB3T; C.children[2]->parent = &C;}
                if(A0xB1T != NULL && A2xB3T != NULL){
					add_to_first(*A0xB1T, *A2xB3T);
                    C.children[2] = A0xB1T;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB1T;
				if(A.children[1] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[1], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A1xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[1], tB, *A1xB1T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3T;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A3xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[3], tB, *A3xB3T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

                if(A1xB1T != NULL && A3xB3T == NULL){C.children[3] = A1xB1T; C.children[3]->parent = &C;}
				if(A1xB1T == NULL && A3xB3T != NULL){C.children[3] = A3xB3T; C.children[3]->parent = &C;}
                if(A1xB1T != NULL && A3xB3T != NULL){
                    add_to_first(*A1xB1T, *A3xB3T);
					C.children[3] = A1xB1T;
                    C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(tA && !tB){
				// C0 = A0^TB0 + A1^TB1
				// C1 = A2^TB0 + A3^TB1
				// C2 = A0^TB2 + A1^TB3
				// C3 = A2^TB2 + A3^TB3

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A0TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[0], tB, *A0TxB0, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB1;
				if(A.children[1] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[1], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A1TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[1], tB, *A1TxB1, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A0TxB0 != NULL && A1TxB1 == NULL){C.children[0] = A0TxB0;  C.children[0]->parent = &C;}
			    if(A0TxB0 == NULL && A1TxB1 != NULL){C.children[0] = A1TxB1;  C.children[0]->parent = &C;}
                if(A0TxB0 != NULL && A1TxB1 != NULL){
					add_to_first(*A0TxB0,*A1TxB1);
                    C.children[0] = A0TxB0;
                    C.children[0]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0;
				if(A.children[2] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[0], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A2TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[0], tB, *A2TxB0, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB1;
				if(A.children[3] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[1], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A3TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[1], tB, *A3TxB1, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A2TxB0 != NULL && A3TxB1 == NULL){C.children[1] = A2TxB0; C.children[1]->parent = &C;}
				if(A2TxB0 == NULL && A3TxB1 != NULL){C.children[1] = A3TxB1; C.children[1]->parent = &C;}
                if(A2TxB0 != NULL && A3TxB1 != NULL){
                    add_to_first(*A2TxB0,*A3TxB1);
					C.children[1] = A2TxB0;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB2;
				if(A.children[0] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[2], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A0TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[2], tB, *A0TxB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3;
				if(A.children[1] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[3], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A1TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[3], tB, *A1TxB3, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

                if(A0TxB2 != NULL && A1TxB3 == NULL){C.children[2] = A0TxB2; C.children[2]->parent = &C;}
				if(A0TxB2 == NULL && A1TxB3 != NULL){C.children[2] = A1TxB3; C.children[2]->parent = &C;}
                if(A0TxB2 != NULL && A1TxB3 != NULL){
                    add_to_first(*A0TxB2,*A1TxB3);
					C.children[2] = A0TxB2;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB2;
				if(A.children[2] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[2], tB, tau)){


					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A2TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[2], tB, *A2TxB2, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A3TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[3], tB, *A3TxB3, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A2TxB2 != NULL && A3TxB3 == NULL){C.children[3] = A2TxB2; C.children[3]->parent = &C;}
				if(A2TxB2 == NULL && A3TxB3 != NULL){C.children[3] = A3TxB3; C.children[3]->parent = &C;}
                if(A2TxB2 != NULL && A3TxB3 != NULL){
                    add_to_first(*A2TxB2,*A3TxB3);
					C.children[3] = A2TxB2;
                    C.children[3]->parent = &C;
                }


				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;
			}


			if(tA && tB){
				// C0 = A0^TB0^T + A1^TB2^T
				// C1 = A2^TB0^T + A3^TB2^T
				// C2 = A0^TB1^T + A1^TB3^T
				// C3 = A2^TB1^T + A3^TB3^T


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0T;
				if(A.children[0] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[0], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A0TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[0], tB, *A0TxB0T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB2T;
				if(A.children[1] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[2], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A1TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[2], tB, *A1TxB2T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A0TxB0T != NULL && A1TxB2T == NULL){C.children[0] = A0TxB0T; C.children[0]->parent = &C;}
				if(A0TxB0T == NULL && A1TxB2T != NULL){C.children[0] = A1TxB2T; C.children[0]->parent = &C;}
                if(A0TxB0T != NULL && A1TxB2T != NULL){
                    add_to_first(*A0TxB0T,*A1TxB2T);
					C.children[0] = A0TxB0T;
                    C.children[0]->parent = &C;
                }


				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0T;
				if(A.children[2] != NULL && B.children[0] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[0], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[0]->get_frob_norm_squared_internal() > tau_squared){
						A2TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[0], tB, *A2TxB0T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB2T;
				if(A.children[3] != NULL && B.children[2] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[2], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[2]->get_frob_norm_squared_internal() > tau_squared){
						A3TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[2], tB, *A3TxB2T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A2TxB0T != NULL && A3TxB2T == NULL){C.children[1] = A2TxB0T; C.children[1]->parent = &C;}
				if(A2TxB0T == NULL && A3TxB2T != NULL){C.children[1] = A3TxB2T; C.children[1]->parent = &C;}
                if(A2TxB0T != NULL && A3TxB2T != NULL){
                    add_to_first(*A2TxB0T,*A3TxB2T);
					C.children[1] = A2TxB0T;
                    C.children[1]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB1T;
				if(A.children[0] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[0], tA, *B.children[1], tB, tau)){

					if(A.children[0]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A0TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[0], tA, *B.children[1], tB, *A0TxB1T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3T;
				if(A.children[1] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[1], tA, *B.children[3], tB, tau)){

					if(A.children[1]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A1TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[1], tA, *B.children[3], tB, *A1TxB3T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A0TxB1T != NULL && A1TxB3T == NULL){C.children[2] = A0TxB1T; C.children[2]->parent = &C;}
				if(A0TxB1T == NULL && A1TxB3T != NULL){C.children[2] = A1TxB3T; C.children[2]->parent = &C;}
                if(A0TxB1T != NULL && A1TxB3T != NULL){
                    add_to_first(*A0TxB1T,*A1TxB3T);
					C.children[2] = A0TxB1T;
                    C.children[2]->parent = &C;
                }

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB1T;
				if(A.children[2] != NULL && B.children[1] != NULL && worth_to_spamm(*A.children[2], tA, *B.children[1], tB, tau)){

					if(A.children[2]->get_frob_norm_squared_internal() *  B.children[1]->get_frob_norm_squared_internal() > tau_squared){
						A2TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[2], tA, *B.children[1], tB, *A2TxB1T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3T;
				if(A.children[3] != NULL && B.children[3] != NULL && worth_to_spamm(*A.children[3], tA, *B.children[3], tB, tau)){

					if(A.children[3]->get_frob_norm_squared_internal() *  B.children[3]->get_frob_norm_squared_internal() > tau_squared){
						A3TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
						spamm(*A.children[3], tA, *B.children[3], tB, *A3TxB3T, tau, true, no_of_block_multiplies, no_of_resizes);
					}
				}

				if(A2TxB1T != NULL && A3TxB3T == NULL){C.children[3] = A2TxB1T; C.children[3]->parent = &C;}
				if(A2TxB1T == NULL && A3TxB3T != NULL){C.children[3] = A3TxB3T; C.children[3]->parent = &C;}
                if(A2TxB1T != NULL && A3TxB3T != NULL){
                    add_to_first(*A2TxB1T,*A3TxB3T);
					C.children[3] = A2TxB1T;
                    C.children[3]->parent = &C;
                }

				if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
					throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::spamm(): resulting matrix is not consistent, smth went wrong!");
				}
				if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

				return;

			}


			return;
		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::self_frob_block_trunc(Treal trunc_value) {

			bool at_least_one_removed = false;

			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL){ // if child exist
					if(children[i]->get_frob_squared() < trunc_value*trunc_value){
						children[i] = nullptr;
						at_least_one_removed = at_least_one_removed || true;
					}
					else{
						bool child_cleared = children[i]->self_frob_block_trunc(trunc_value);
						at_least_one_removed = at_least_one_removed || child_cleared;
					}
				}
			}


			for(int i = 0; i < 4; ++i){
				if(children[i] != NULL){
					if(!children[i]->lowest_level() && !children[i]->children_exist()){
						children[i] = nullptr;
					}
				}
			}

			return at_least_one_removed;

		}

	template<class Treal>
		bool HierarchicalBlockSparseMatrix<Treal>::frob_block_trunc(HierarchicalBlockSparseMatrix<Treal> & matrix_truncated, Treal trunc_value) const {

			if(!matrix_truncated.empty()) matrix_truncated.clear();

			matrix_truncated.copy(*this);

			return matrix_truncated.self_frob_block_trunc(trunc_value);
		}

	template<class Treal>
		std::vector<unsigned long int> HierarchicalBlockSparseMatrix<Treal>::count_skips(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA,
			 HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus,
		   const bool & apply_truncation, const bool & apply_spamm){

	    // look at the current level
			std::vector<unsigned long int> curr_level_skips;
			const int length = taus.size();
			curr_level_skips.resize(length);

			Treal A_norm = std::sqrt(A.get_frob_norm_squared_internal());
			Treal B_norm = std::sqrt(B.get_frob_norm_squared_internal());

			Treal product_of_norms = A_norm * B_norm;

			for(int i = 0; i < length; ++i){

					// depending on values of apply_truncation and apply_spamm decide if this multiplication is to be skipped

					//truncmul
					if(apply_truncation && !apply_spamm){
						if(A_norm < taus[i] || B_norm < taus[i])	curr_level_skips[i] = 1;
						else curr_level_skips[i] = 0;
					}

					//SpAMM
					if(!apply_truncation && apply_spamm){
						if(product_of_norms < taus[i]) curr_level_skips[i] = 1;
						else curr_level_skips[i] = 0;
					}

					//hybrid
					if(apply_truncation && apply_spamm){
						if(A_norm < taus[i] || B_norm < taus[i] || product_of_norms < taus[i]) curr_level_skips[i] = 1;
						else curr_level_skips[i] = 0;
					}

					if(!apply_truncation && !apply_spamm){
						curr_level_skips[i] = 0;
					}

			}


			if(A.lowest_level() && B.lowest_level()){
					return curr_level_skips;
			}

			if(A.get_depth() < B.get_depth()){

					bool B0 = false, B1 = false, B2 = false, B3 = false;
					if(B.children[0] != NULL) B0 = true;
					if(B.children[1] != NULL) B1 = true;
					if(B.children[2] != NULL) B2 = true;
					if(B.children[3] != NULL) B3 = true;

					std::vector<unsigned long int> sub_skips0(length,0),sub_skips1(length,0),sub_skips2(length,0),sub_skips3(length,0);

					if(B0) sub_skips0 = count_skips(A, tA, *B.children[0],tB, taus, apply_truncation, apply_spamm);
					if(tB){
						if(B2) sub_skips1 = count_skips(A, tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);
						if(B1) sub_skips2 = count_skips(A, tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);
					}
					else{
						if(B1) sub_skips1 = count_skips(A, tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);
					  if(B2) sub_skips2 = count_skips(A, tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);
					}
					if(B3) sub_skips3 = count_skips(A, tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);

					std::vector<std::vector<unsigned long int> > all_subskips;
					all_subskips.push_back(std::move(sub_skips0));
					all_subskips.push_back(std::move(sub_skips1));
					all_subskips.push_back(std::move(sub_skips2));
					all_subskips.push_back(std::move(sub_skips3));

					return sum_skips(curr_level_skips, all_subskips);
			}

			if(A.get_depth() > B.get_depth()){

					bool A0 = false, A1 = false, A2 = false, A3 = false;

					std::vector<unsigned long int> sub_skips0(length,0),sub_skips1(length,0),sub_skips2(length,0),sub_skips3(length,0);

					if(A.children[0] != NULL) A0 = true;
					if(A.children[1] != NULL) A1 = true;
					if(A.children[2] != NULL) A2 = true;
					if(A.children[3] != NULL) A3 = true;

					if(A0) sub_skips0 = count_skips(*A.children[0], tA, B, tB, taus, apply_truncation, apply_spamm);
					if(tA){
						if(A2) sub_skips1 = count_skips(*A.children[2], tA, B, tB, taus, apply_truncation, apply_spamm);
						if(A1) sub_skips2 = count_skips(*A.children[1], tA, B, tB, taus, apply_truncation, apply_spamm);
					}
					else{
						if(A1) sub_skips1 = count_skips(*A.children[1], tA, B, tB, taus, apply_truncation, apply_spamm);
						if(A2) sub_skips2 = count_skips(*A.children[2], tA, B, tB, taus, apply_truncation, apply_spamm);
					}
					if(A3) sub_skips3 = count_skips(*A.children[3], tA, B, tB, taus, apply_truncation, apply_spamm);

					std::vector<std::vector<unsigned long int> > all_subskips;
					all_subskips.push_back(std::move(sub_skips0));
					all_subskips.push_back(std::move(sub_skips1));
					all_subskips.push_back(std::move(sub_skips2));
					all_subskips.push_back(std::move(sub_skips3));

					return sum_skips(curr_level_skips, all_subskips);
			}

			bool A0 = false, A1 = false, A2 = false, A3 = false, B0 = false, B1 = false, B2 = false, B3 = false;

			if(A.children[0] != NULL) A0 = true;
			if(A.children[1] != NULL) A1 = true;
			if(A.children[2] != NULL) A2 = true;
			if(A.children[3] != NULL) A3 = true;

			if(B.children[0] != NULL) B0 = true;
			if(B.children[1] != NULL) B1 = true;
			if(B.children[2] != NULL) B2 = true;
			if(B.children[3] != NULL) B3 = true;

			std::vector<unsigned long int> sub_skips0(length,0),sub_skips1(length,0),sub_skips2(length,0),sub_skips3(length,0),sub_skips4(length,0),sub_skips5(length,0),sub_skips6(length,0),sub_skips7(length,0);

			if(!tA && !tB){
					if(A0 && B0) sub_skips0 = count_skips(*A.children[0], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B1) sub_skips1 = count_skips(*A.children[2], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B0) sub_skips2 = count_skips(*A.children[1], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B1) sub_skips3 = count_skips(*A.children[3], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A0 && B2) sub_skips4 = count_skips(*A.children[0], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B3) sub_skips5 = count_skips(*A.children[2], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B2) sub_skips6 = count_skips(*A.children[1], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B3) sub_skips7 = count_skips(*A.children[3], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);
 			}

			if(!tA && tB){
					if(A0 && B0) sub_skips0 = count_skips(*A.children[0], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B2) sub_skips1 = count_skips(*A.children[2], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B0) sub_skips2 = count_skips(*A.children[1], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B2) sub_skips3 = count_skips(*A.children[3], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A0 && B1) sub_skips4 = count_skips(*A.children[0], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B3) sub_skips5 = count_skips(*A.children[2], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B1) sub_skips6 = count_skips(*A.children[1], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B3) sub_skips7 = count_skips(*A.children[3], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);
			}

			if(tA && !tB){
					if(A0 && B0) sub_skips0 = count_skips(*A.children[0], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B1) sub_skips1 = count_skips(*A.children[1], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B0) sub_skips2 = count_skips(*A.children[2], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B1) sub_skips3 = count_skips(*A.children[3], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A0 && B2) sub_skips4 = count_skips(*A.children[0], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B3) sub_skips5 = count_skips(*A.children[1], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B2) sub_skips6 = count_skips(*A.children[2], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B3) sub_skips7 = count_skips(*A.children[3], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);
			}

			if(tA && tB){
					if(A0 && B0) sub_skips0 = count_skips(*A.children[0], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B2) sub_skips1 = count_skips(*A.children[1], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B0) sub_skips2 = count_skips(*A.children[2], tA, *B.children[0], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B2) sub_skips3 = count_skips(*A.children[3], tA, *B.children[2], tB, taus, apply_truncation, apply_spamm);;

					if(A0 && B1) sub_skips4 = count_skips(*A.children[0], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A1 && B3) sub_skips5 = count_skips(*A.children[1], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);

					if(A2 && B1) sub_skips6 = count_skips(*A.children[2], tA, *B.children[1], tB, taus, apply_truncation, apply_spamm);

					if(A3 && B3) sub_skips7 = count_skips(*A.children[3], tA, *B.children[3], tB, taus, apply_truncation, apply_spamm);
			}



			std::vector<std::vector<unsigned long int> > all_subskips;
			all_subskips.push_back(std::move(sub_skips0));
			all_subskips.push_back(std::move(sub_skips1));
			all_subskips.push_back(std::move(sub_skips2));
			all_subskips.push_back(std::move(sub_skips3));
			all_subskips.push_back(std::move(sub_skips4));
			all_subskips.push_back(std::move(sub_skips5));
			all_subskips.push_back(std::move(sub_skips6));
			all_subskips.push_back(std::move(sub_skips7));

			return sum_skips(curr_level_skips, all_subskips);



	}

 template<class Treal>
 		std::vector<unsigned long int> HierarchicalBlockSparseMatrix<Treal>::sum_skips(std::vector<unsigned long int> const & curr_level_skips, std::vector<std::vector<unsigned long int> > const & sub_skips){

			  std::vector<unsigned long int> total_skips;
				total_skips.resize(curr_level_skips.size());

				for(int j = 0; j < curr_level_skips.size(); ++j){

					if(curr_level_skips[j] == 1) total_skips[j] = 1;
					else{
						for(int i = 0; i < sub_skips.size();++i){
							total_skips[j] += sub_skips[i][j];
						}
					}
				}

				return total_skips;
		}

	template<class Treal>
		Treal HierarchicalBlockSparseMatrix<Treal>::get_max_abs_value() const {
				std::vector<Treal> all_values;
				std::vector<int> rows, cols;

				get_all_values(rows,cols,all_values);

				Treal max_abs_value = 0.0;

				for(auto i = 0; i < all_values.size(); ++i){
					if(fabs(all_values[i]) > max_abs_value) max_abs_value = all_values[i];
				}

				return max_abs_value;
		}

	template<class Treal>
	  std::vector<Treal> HierarchicalBlockSparseMatrix<Treal>::get_errors_of_approx_multiplication(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus, const bool & apply_truncation, const bool & apply_spamm){

			std::vector<unsigned long int> skips = count_skips(A, tA, B, tB, taus, apply_truncation, apply_spamm);

			assert(skips.size() == taus.size());

			std::vector<Treal> errors;
			errors.resize(skips.size());

			Treal c1 = A.get_max_abs_value();
			Treal c2 = B.get_max_abs_value();

			//truncmul
			if(apply_truncation && !apply_spamm){
				for(int i = 0; i < errors.size(); ++i){
					errors[i] = std::max(c1,c2) * std::sqrt(taus[i] * taus[i] * skips[i]);
				}
			}

			//spamm
			if(!apply_truncation && apply_spamm){
				for(int i = 0; i < errors.size(); ++i){
					errors[i] = std::sqrt(taus[i] * taus[i] * skips[i]);
				}
			}

			//hybrid
			if(apply_truncation	&& apply_spamm){
				for(int i = 0; i < errors.size(); ++i){
					errors[i] = std::max(1.0,std::max(c1,c2)) * std::sqrt(taus[i] * taus[i] * skips[i]);
				}
			}

			//regular multiplication - do nothing
			if(!apply_truncation && !apply_spamm){

			}

			return errors;

		}

		template<class Treal>
		  std::vector<Treal> HierarchicalBlockSparseMatrix<Treal>::get_spamm_errors(HierarchicalBlockSparseMatrix<Treal> const & A, const bool tA, HierarchicalBlockSparseMatrix<Treal> const & B, const bool tB, std::vector<Treal> const & taus){



				// look at the current level
			 const int length = taus.size();
			 std::vector<real> curr_level_errors;

			 if(!worth_to_multiply(A, tA, B, tB)){
				 return curr_level_errors;
			 }

			 curr_level_errors.resize(length,0.0);

			 Treal A_norm = std::sqrt(A.get_frob_norm_squared_internal());
			 Treal B_norm = std::sqrt(B.get_frob_norm_squared_internal());

			 Treal product_of_norms = A_norm * B_norm;

			 for(int i = 0; i < length; ++i){

				 	 // ensure that not only matrix product is small enough, but also that it is not empty, which might be the case
					 if(product_of_norms < taus[i]) curr_level_errors[i] = product_of_norms;

			 }

			 if(A.lowest_level() && B.lowest_level()){
					 return curr_level_errors;
			 }

			 if(A.get_depth() < B.get_depth()){

 					bool B0 = false, B1 = false, B2 = false, B3 = false;
 					if(B.children[0] != NULL) B0 = true;
 					if(B.children[1] != NULL) B1 = true;
 					if(B.children[2] != NULL) B2 = true;
 					if(B.children[3] != NULL) B3 = true;

 					std::vector<real> sub_errors0,sub_errors1,sub_errors2,sub_errors3;

 					if(B0){
					  sub_errors0 = get_spamm_errors(A, tA, *B.children[0],tB, taus);
				  }
 					if(tB){
 						if(B2) sub_errors1 = get_spamm_errors(A, tA, *B.children[2], tB, taus);
 						if(B1) sub_errors2 = get_spamm_errors(A, tA, *B.children[1], tB, taus);
 					}
 					else{
 						if(B1) sub_errors1 = get_spamm_errors(A, tA, *B.children[1], tB, taus);
 						if(B2) sub_errors2 = get_spamm_errors(A, tA, *B.children[2], tB, taus);
 					}
 					if(B3) sub_errors3 = get_spamm_errors(A, tA, *B.children[3], tB, taus);

					if(sub_errors0.empty() && sub_errors1.empty() && sub_errors2.empty() && sub_errors3.empty()){
						return curr_level_errors;
					}

 					std::vector<std::vector<real> > all_suberrors;
 					all_suberrors.push_back(std::move(sub_errors0));
 					all_suberrors.push_back(std::move(sub_errors1));
 					all_suberrors.push_back(std::move(sub_errors2));
 					all_suberrors.push_back(std::move(sub_errors3));

 					return sum_spamm_errors(curr_level_errors, all_suberrors);
 			}

 			if(A.get_depth() > B.get_depth()){

 					bool A0 = false, A1 = false, A2 = false, A3 = false;

 					std::vector<real> sub_errors0,sub_errors1,sub_errors2,sub_errors3;

 					if(A.children[0] != NULL) A0 = true;
 					if(A.children[1] != NULL) A1 = true;
 					if(A.children[2] != NULL) A2 = true;
 					if(A.children[3] != NULL) A3 = true;

 					if(A0) sub_errors0 = get_spamm_errors(*A.children[0], tA, B, tB, taus);
 					if(tA){
 						if(A2) sub_errors1 = get_spamm_errors(*A.children[2], tA, B, tB, taus);
 						if(A1) sub_errors2 = get_spamm_errors(*A.children[1], tA, B, tB, taus);
 					}
 					else{
 						if(A1) sub_errors1 = get_spamm_errors(*A.children[1], tA, B, tB, taus);
 						if(A2) sub_errors2 = get_spamm_errors(*A.children[2], tA, B, tB, taus);
 					}
 					if(A3) sub_errors3 = get_spamm_errors(*A.children[3], tA, B, tB, taus);

					if(sub_errors0.empty() && sub_errors1.empty() && sub_errors2.empty() && sub_errors3.empty()){
						return curr_level_errors;
					}

					std::vector<std::vector<real> > all_suberrors;
					all_suberrors.push_back(std::move(sub_errors0));
					all_suberrors.push_back(std::move(sub_errors1));
					all_suberrors.push_back(std::move(sub_errors2));
					all_suberrors.push_back(std::move(sub_errors3));

 					return sum_spamm_errors(curr_level_errors, all_suberrors);
 			}

			bool A0 = false, A1 = false, A2 = false, A3 = false, B0 = false, B1 = false, B2 = false, B3 = false;

			if(A.children[0] != NULL) A0 = true;
			if(A.children[1] != NULL) A1 = true;
			if(A.children[2] != NULL) A2 = true;
			if(A.children[3] != NULL) A3 = true;

			if(B.children[0] != NULL) B0 = true;
			if(B.children[1] != NULL) B1 = true;
			if(B.children[2] != NULL) B2 = true;
			if(B.children[3] != NULL) B3 = true;

			std::vector<real> sub_errors0,sub_errors1,sub_errors2,sub_errors3,sub_errors4,sub_errors5,sub_errors6,sub_errors7;

			if(!tA && !tB){
					if(A0 && B0) sub_errors0 = get_spamm_errors(*A.children[0], tA, *B.children[0], tB, taus);

					if(A2 && B1) sub_errors1 = get_spamm_errors(*A.children[2], tA, *B.children[1], tB, taus);

					if(A1 && B0) sub_errors2 = get_spamm_errors(*A.children[1], tA, *B.children[0], tB, taus);

					if(A3 && B1) sub_errors3 = get_spamm_errors(*A.children[3], tA, *B.children[1], tB, taus);

					if(A0 && B2) sub_errors4 = get_spamm_errors(*A.children[0], tA, *B.children[2], tB, taus);

					if(A2 && B3) sub_errors5 = get_spamm_errors(*A.children[2], tA, *B.children[3], tB, taus);

					if(A1 && B2) sub_errors6 = get_spamm_errors(*A.children[1], tA, *B.children[2], tB, taus);

					if(A3 && B3) sub_errors7 = get_spamm_errors(*A.children[3], tA, *B.children[3], tB, taus);
 			}

			if(!tA && tB){
					if(A0 && B0) sub_errors0 = get_spamm_errors(*A.children[0], tA, *B.children[0], tB, taus);

					if(A2 && B2) sub_errors1 = get_spamm_errors(*A.children[2], tA, *B.children[2], tB, taus);

					if(A1 && B0) sub_errors2 = get_spamm_errors(*A.children[1], tA, *B.children[0], tB, taus);

					if(A3 && B2) sub_errors3 = get_spamm_errors(*A.children[3], tA, *B.children[2], tB, taus);

					if(A0 && B1) sub_errors4 = get_spamm_errors(*A.children[0], tA, *B.children[1], tB, taus);

					if(A2 && B3) sub_errors5 = get_spamm_errors(*A.children[2], tA, *B.children[3], tB, taus);

					if(A1 && B1) sub_errors6 = get_spamm_errors(*A.children[1], tA, *B.children[1], tB, taus);

					if(A3 && B3) sub_errors7 = get_spamm_errors(*A.children[3], tA, *B.children[3], tB, taus);
			}

			if(tA && !tB){
					if(A0 && B0) sub_errors0 = get_spamm_errors(*A.children[0], tA, *B.children[0], tB, taus);

					if(A1 && B1) sub_errors1 = get_spamm_errors(*A.children[1], tA, *B.children[1], tB, taus);

					if(A2 && B0) sub_errors2 = get_spamm_errors(*A.children[2], tA, *B.children[0], tB, taus);

					if(A3 && B1) sub_errors3 = get_spamm_errors(*A.children[3], tA, *B.children[1], tB, taus);

					if(A0 && B2) sub_errors4 = get_spamm_errors(*A.children[0], tA, *B.children[2], tB, taus);

					if(A1 && B3) sub_errors5 = get_spamm_errors(*A.children[1], tA, *B.children[3], tB, taus);

					if(A2 && B2) sub_errors6 = get_spamm_errors(*A.children[2], tA, *B.children[2], tB, taus);

					if(A3 && B3) sub_errors7 = get_spamm_errors(*A.children[3], tA, *B.children[3], tB, taus);
			}

			if(tA && tB){
					if(A0 && B0) sub_errors0 = get_spamm_errors(*A.children[0], tA, *B.children[0], tB, taus);

					if(A1 && B2) sub_errors1 = get_spamm_errors(*A.children[1], tA, *B.children[2], tB, taus);

					if(A2 && B0) sub_errors2 = get_spamm_errors(*A.children[2], tA, *B.children[0], tB, taus);

					if(A3 && B2) sub_errors3 = get_spamm_errors(*A.children[3], tA, *B.children[2], tB, taus);

					if(A0 && B1) sub_errors4 = get_spamm_errors(*A.children[0], tA, *B.children[1], tB, taus);

					if(A1 && B3) sub_errors5 = get_spamm_errors(*A.children[1], tA, *B.children[3], tB, taus);

					if(A2 && B1) sub_errors6 = get_spamm_errors(*A.children[2], tA, *B.children[1], tB, taus);

					if(A3 && B3) sub_errors7 = get_spamm_errors(*A.children[3], tA, *B.children[3], tB, taus);
			}

			if(sub_errors0.empty() && sub_errors1.empty() && sub_errors2.empty() && sub_errors3.empty() &&
				 sub_errors4.empty() && sub_errors5.empty() && sub_errors6.empty() && sub_errors7.empty() ){
				return curr_level_errors;
			}

			std::vector<std::vector<real> > all_suberrors;

			if(sub_errors0.empty() && sub_errors1.empty()) all_suberrors.push_back(sub_errors0);
			if(sub_errors0.empty() && !sub_errors1.empty()) all_suberrors.push_back(sub_errors1);
			if(!sub_errors0.empty() && sub_errors1.empty()) all_suberrors.push_back(sub_errors0);
			if(!sub_errors0.empty() && !sub_errors1.empty()) all_suberrors.push_back(sub_errors0+sub_errors1);

			if(sub_errors2.empty() && sub_errors3.empty()) all_suberrors.push_back(sub_errors2);
			if(sub_errors2.empty() && !sub_errors3.empty()) all_suberrors.push_back(sub_errors3);
			if(!sub_errors2.empty() && sub_errors3.empty()) all_suberrors.push_back(sub_errors2);
			if(!sub_errors2.empty() && !sub_errors3.empty()) all_suberrors.push_back(sub_errors2+sub_errors3);

			if(sub_errors4.empty() && sub_errors5.empty()) all_suberrors.push_back(sub_errors4);
			if(sub_errors4.empty() && !sub_errors5.empty()) all_suberrors.push_back(sub_errors5);
			if(!sub_errors4.empty() && sub_errors5.empty()) all_suberrors.push_back(sub_errors4);
			if(!sub_errors4.empty() && !sub_errors5.empty()) all_suberrors.push_back(sub_errors4+sub_errors5);

			if(sub_errors6.empty() && sub_errors7.empty()) all_suberrors.push_back(sub_errors6);
			if(sub_errors6.empty() && !sub_errors7.empty()) all_suberrors.push_back(sub_errors7);
			if(!sub_errors6.empty() && sub_errors7.empty()) all_suberrors.push_back(sub_errors6);
			if(!sub_errors6.empty() && !sub_errors7.empty()) all_suberrors.push_back(sub_errors6+sub_errors7);

			return sum_spamm_errors(curr_level_errors, all_suberrors);

		}

		template<class Treal>
	  		std::vector<Treal> HierarchicalBlockSparseMatrix<Treal>::sum_spamm_errors(std::vector<real> const & curr_level_errors, std::vector<std::vector<real> > const & sub_errors){

	 			  std::vector<real> total_errors(curr_level_errors.size(),0.0);


	 				for(int j = 0; j < curr_level_errors.size(); ++j){

					// WHAT IS COMMENTED OUT IS DUE TO THE NEW VERSION OF THE ALGORITHM PROPOSED BY EMANUEL
	 				//	if(curr_level_errors[j] > 0.0) total_errors[j] = curr_level_errors[j];
	 				//	else{
	 						for(int i = 0; i < sub_errors.size();++i){
								if(sub_errors[i].size() == 0) continue;
								else total_errors[j] += sub_errors[i][j] * sub_errors[i][j];
	 						}
							total_errors[j] = std::sqrt(total_errors[j]);
	 					//}
	 				}

	 				return total_errors;
	 		}


			template<class Treal>
				void HierarchicalBlockSparseMatrix<Treal>::get_batches_multiply(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
		                        HierarchicalBlockSparseMatrix<Treal>& C, BatchMapMultiply & batches, int dummy_levels_in){

					//	std::cout << "inside get_batches_multiply, dummy_levels_in = " << dummy_levels_in << std::endl;

						if(A.get_depth() < B.get_depth()){


							if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): non-empty matrix to write result!");

							C.set_params(A.get_params());

							if(!tA && !tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig);
			                    else C.resize(B.nRows_orig, B.nCols_orig);
							}

							if(!tA && tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig);
			                    else C.resize(B.nRows_orig, B.nRows_orig);
							}

							if(tA && !tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig);
			                    else C.resize(B.nCols_orig, B.nCols_orig);
							}

							if(tA && tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig);
			                    C.resize(B.nCols_orig, B.nRows_orig);
							}

							if(!worth_to_multiply(A,tA,B,tB)){
								return;
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB0;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB1;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB2;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > AxB3;

							if(B.children[0] != NULL && worth_to_multiply(A, tA, *B.children[0], tB)){
								AxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(A, tA, *B.children[0], tB, *AxB0, no_of_block_multiplies, no_of_resizes);
								C.children[0] = AxB0;
								C.children[0]->parent = &C;
								get_batches_multiply(A, tA, *B.children[0], tB, *AxB0, batches, dummy_levels_in);
							}


							if(tB){
								if(B.children[2] != NULL  && worth_to_multiply(A, tA, *B.children[2], tB)){
									AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(A, tA, *B.children[2], tB, *AxB1, no_of_block_multiplies, no_of_resizes);
									C.children[1] = AxB1;
									C.children[1]->parent = &C;
									get_batches_multiply(A, tA, *B.children[2], tB, *AxB1, batches, dummy_levels_in);
								}
							}
							else{
								if(B.children[1] != NULL  && worth_to_multiply(A, tA, *B.children[1], tB)){
									AxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(A, tA, *B.children[1], tB, *AxB1, no_of_block_multiplies, no_of_resizes);
									C.children[1] = AxB1;
									C.children[1]->parent = &C;;
									get_batches_multiply(A, tA, *B.children[1], tB, *AxB1, batches,dummy_levels_in);
								}
							}


							if(tB){
								if(B.children[1] != NULL  && worth_to_multiply(A, tA, *B.children[1], tB))	{
									AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(A, tA, *B.children[1], tB, *AxB2, no_of_block_multiplies, no_of_resizes);
									C.children[2] = AxB2;
									C.children[2]->parent = &C;
									get_batches_multiply(A, tA, *B.children[1], tB, *AxB2, batches, dummy_levels_in);
								}
							}
							else{
								if(B.children[2] != NULL  && worth_to_multiply(A, tA, *B.children[2], tB)){
									AxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(A, tA, *B.children[2], tB, *AxB2, no_of_block_multiplies, no_of_resizes);
									C.children[2] = AxB2;
									C.children[2]->parent = &C;
									get_batches_multiply(A, tA, *B.children[2], tB, *AxB2, batches, dummy_levels_in);
								}
							}

							if(B.children[3] != NULL  && worth_to_multiply(A, tA, *B.children[3], tB)){
								AxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(A, tA, *B.children[3], tB, *AxB3, no_of_block_multiplies, no_of_resizes);
								C.children[3] = AxB3;
								C.children[3]->parent = &C;
								get_batches_multiply(A, tA, *B.children[3], tB, *AxB3, batches, dummy_levels_in);
							}

							return;

						}


						if(A.get_depth() > B.get_depth()){


							if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): non-empty matrix to write result!");

							C.set_params(A.get_params());

							if(!tA && !tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nCols_orig);
			                    else C.resize(A.nRows_orig, A.nCols_orig);
							}

							if(!tA && tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nRows_orig, B.nRows_orig);
			                    else C.resize(A.nRows_orig, A.nRows_orig);
							}

							if(tA && !tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nCols_orig);
			                    else C.resize(A.nCols_orig, A.nCols_orig);
							}

							if(tA && tB){
								if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
								if(A.get_level() == 0 && B.get_level() == 0) C.resize(A.nCols_orig, B.nRows_orig);
			                    else C.resize(A.nCols_orig, A.nRows_orig);
							}

							if(!worth_to_multiply(A,tA,B,tB)){
								return;
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB;
							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB;

							if(A.children[0] != NULL && worth_to_multiply(*A.children[0], tA, B, tB)){
								A0xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[0], tA, B, tB, *A0xB, no_of_block_multiplies, no_of_resizes);
								C.children[0] = A0xB;
								C.children[0]->parent = &C;
								get_batches_multiply(*A.children[0], tA, B, tB, *A0xB, batches, dummy_levels_in);
							}

							if(tA){
								if(A.children[2] != NULL && worth_to_multiply(*A.children[2], tA, B, tB)){
									A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(*A.children[2], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
									C.children[1] = A1xB;
									C.children[1]->parent = &C;
									get_batches_multiply(*A.children[2], tA, B, tB, *A1xB, batches, dummy_levels_in);
								}
							}
							else{
								if(A.children[1] != NULL && worth_to_multiply(*A.children[1], tA, B, tB)){
									A1xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(*A.children[1], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
									C.children[1] = A1xB;
									C.children[1]->parent = &C;
									get_batches_multiply(*A.children[1], tA, B, tB, *A1xB, batches, dummy_levels_in);
								}
							}

							if(tA){
								if(A.children[1] != NULL && worth_to_multiply(*A.children[1], tA, B, tB)){
									A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(*A.children[1], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
									C.children[2] = A2xB;
									C.children[2]->parent = &C;
									get_batches_multiply(*A.children[1], tA, B, tB, *A2xB, batches, dummy_levels_in);
								}
							}
							else{
								if(A.children[2] != NULL && worth_to_multiply(*A.children[2], tA, B, tB)){
									A2xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
									//multiply(*A.children[2], tA, B, tB, *A1xB, no_of_block_multiplies, no_of_resizes);
									C.children[2] = A2xB;
									C.children[2]->parent = &C;
									get_batches_multiply(*A.children[2], tA, B, tB, *A2xB, batches, dummy_levels_in);
								}
							}

							if(A.children[3] != NULL && worth_to_multiply(*A.children[3], tA, B, tB)){
								A3xB = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[3], tA, B, tB, *A3xB, no_of_block_multiplies, no_of_resizes);
								C.children[3] = A3xB;
								C.children[3]->parent = &C;
								get_batches_multiply(*A.children[3], tA, B, tB, *A3xB, batches, dummy_levels_in);
							}

							return;

						}


						if(!C.empty()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): non-empty matrix to write result! wow");

						C.set_params(A.get_params());


						if(!worth_to_multiply(A,tA,B,tB)){

							// case when both matrices are 0 level, but there is no sense in multiplying them, just resize C and return.
							assert(A.get_level() == 0 && B.get_level() == 0);

							if(!tA && !tB) C.resize(A.nRows_orig,B.nCols_orig);
							if(!tA && tB) C.resize(A.nRows_orig,B.nRows_orig);
							if(tA && !tB) C.resize(A.nCols_orig,B.nCols_orig);
							if(tA && tB) C.resize(A.nCols_orig,B.nRows_orig);

							return;
						}

						bool squeeze_needed = false;
						int n_dummy_levels = 0;

						if(!tA && !tB){
							if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nRows_orig)throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
							if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL)C.resize(A.nRows_orig,B.nCols_orig);
							if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nCols);
							if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nCols_orig);

						//	std::cout << C.nRows << " " <<  A.nRows << " " <<  C.nRows << " " << B.nRows << std::endl;
						//	std::cout << "After resize: position code is " << C.get_position_code() << std::endl;

							if(C.nRows < A.nRows && C.nRows < B.nRows){
								squeeze_needed = true;
								int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

								int ratio = nRows / C.nRows;

								int P = 1;
								int two_to_power_P = 2;
								while(ratio >  two_to_power_P){
									two_to_power_P *= 2;
									P += 1;
								}
								n_dummy_levels = P;
								C.resize(nRows,nRows);
								C.nRows_orig = A.nRows_orig;
								C.nCols_orig = B.nCols_orig;

								//std::cout << "SQEEZE NEEDED with n_dummy_levels " << n_dummy_levels << std::endl;
							}
						}

						if(!tA && tB){
							if(A.get_level() == 0 && B.get_level() == 0 && A.nCols_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
							if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows_orig);
							if(A.parent != NULL && B.parent == NULL) C.resize(A.nRows_orig,B.nRows);
							if(A.parent == NULL && B.parent != NULL) C.resize(A.nRows,B.nRows_orig);

							if(C.nRows < A.nRows && C.nRows < B.nRows){
								squeeze_needed = true;
								int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

								int ratio = nRows / C.nRows;

								int P = 1;
								int two_to_power_P = 2;
								while(ratio >  two_to_power_P){
									two_to_power_P *= 2;
									P += 1;
								}
								n_dummy_levels = P;
								C.resize(nRows,nRows);
								C.nRows_orig = A.nRows_orig;
								C.nCols_orig = B.nRows_orig;
							}

						}

						if(tA && !tB){
							if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nRows_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
							if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nCols_orig);
							if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nCols);
							if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nCols_orig);

							if(C.nRows < A.nRows && C.nRows < B.nRows){
								squeeze_needed = true;
								int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

								int ratio = nRows / C.nRows;

								int P = 1;
								int two_to_power_P = 2;
								while(ratio >  two_to_power_P){
									two_to_power_P *= 2;
									P += 1;
								}
								n_dummy_levels = P;
								C.resize(nRows,nRows);
								C.nRows_orig = A.nCols_orig;
								C.nCols_orig = B.nCols_orig;
							}
						}

						if(tA && tB){
							if(A.get_level() == 0 && B.get_level() == 0 && A.nRows_orig != B.nCols_orig) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): matrices have bad sizes!");
							if(A.parent != NULL && B.parent != NULL || A.parent == NULL && B.parent == NULL) C.resize(A.nCols_orig,B.nRows_orig);
							if(A.parent != NULL && B.parent == NULL) C.resize(A.nCols_orig, B.nRows);
							if(A.parent == NULL && B.parent != NULL) C.resize(A.nCols, B.nRows_orig);

							if(C.nRows < A.nRows && C.nRows < B.nRows){
								squeeze_needed = true;
								int nRows = ((A.nRows > B.nRows) ? A.nRows : B.nRows);

								int ratio = nRows / C.nRows;

								int P = 1;
								int two_to_power_P = 2;
								while(ratio >  two_to_power_P){
									two_to_power_P *= 2;
									P += 1;
								}
								n_dummy_levels = P;
								C.resize(nRows,nRows);
								C.nRows_orig = A.nCols_orig;
								C.nCols_orig = B.nRows_orig;
							}
						}


						if(A.lowest_level()){

							if(!B.lowest_level()) throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): A is at lowest level, B is not!");

							GemmTriplet triplet; //(&A, &B, &C, tA, tB);
							triplet.a = &A;
							triplet.b = &B;
							triplet.c = &C;
							triplet.ta = tA;
							triplet.tb = tB;

							std::string position_code = C.get_position_code();

							//std::cout << "lowest level postition code " << position_code << std::endl;

						//	std::cout << "Indide get_batches_multiply lowest level " <<std::endl;
						//	std::cout << "dummy_levels_in " << dummy_levels_in << std::endl;

							if(dummy_levels_in > 0){
								position_code.erase(0, dummy_levels_in);
							}

							batches.insert(std::pair<std::string, GemmTriplet>(position_code, triplet));

							//std::cout << "Batches lowest level, key is " << C.get_position_code() << std::endl;

							return;
						}

						const int n_dummy_levels_to_pass = (dummy_levels_in > n_dummy_levels) ? dummy_levels_in : n_dummy_levels;

						if(!tA && !tB){


							// C0 = A0xB0 + A2xB1
							// C1 = A1xB0 + A3xB1
							// C2 = A0xB2 + A2xB3
							// C3 = A1xB2 + A3xB3
							//std::cout << "Level " << C.get_level();

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
							if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
								A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0, no_of_block_multiplies, no_of_resizes);
								//A0xB0->parent = &C;
								C.children[0] = A0xB0;
								C.children[0]->parent = &C;
								//std::cout << " child0 set" << std::endl;
								get_batches_multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0, batches, n_dummy_levels_to_pass);
							//	C.children[0] = NULL;
							//	C.children[0]->parent = NULL;
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB1;
							if(A.children[2] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)){
								A2xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[2], tA, *B.children[1], tB, *A2xB1, no_of_block_multiplies, no_of_resizes);
								C.children[0] = A2xB1;
								C.children[0]->parent = &C;
								//std::cout << " child0 set" << std::endl;
								get_batches_multiply(*A.children[2], tA, *B.children[1], tB, *A2xB1, batches, n_dummy_levels_to_pass);
							//	C.children[0] = NULL;
							//	C.children[0]->parent = NULL;
							}

							if(A0xB0 != NULL && A2xB1 == NULL){C.children[0] = A0xB0; C.children[0]->parent = &C;}
							if(A0xB0 == NULL && A2xB1 != NULL){C.children[0] = A2xB1; C.children[0]->parent = &C;}
              if( A0xB0 != NULL && A2xB1 != NULL){
	                    collide_trees(*A0xB0, *A2xB1);
			               	C.children[0] = A0xB0;
	                    C.children[0]->parent = &C;
              }


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0;
							if(A.children[1] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)){
								A1xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0, no_of_block_multiplies, no_of_resizes);
								C.children[1] = A1xB0;
								C.children[1]->parent = &C;
								//std::cout << " child1 set" << std::endl;
								get_batches_multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0, batches, n_dummy_levels_to_pass);
							//	C.children[1] = NULL;
								//C.children[1]->parent = NULL;
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB1;
							if(A.children[3] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[3], tA,*B.children[1], tB)){
								A3xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[3], tA, *B.children[1], tB, *A3xB1, no_of_block_multiplies, no_of_resizes);
								C.children[1] = A3xB1;
								C.children[1]->parent = &C;
								//std::cout << " child1 set" << std::endl;
								get_batches_multiply(*A.children[3], tA, *B.children[1], tB, *A3xB1, batches, n_dummy_levels_to_pass);
								//C.children[1] = NULL;
							//	C.children[1]->parent = NULL;
							}


							if(A1xB0 != NULL && A3xB1 == NULL){ C.children[1] = A1xB0; C.children[1]->parent = &C;}
							if(A1xB0 == NULL && A3xB1 != NULL){ C.children[1] = A3xB1; C.children[1]->parent = &C;}
              if(A1xB0 != NULL && A3xB1 != NULL){
								collide_trees(*A1xB0, *A3xB1);
								C.children[1] = A1xB0;
								C.children[1]->parent = &C;
              }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
							if(A.children[0] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[0], tA,*B.children[2], tB)){
								A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[0], tA, *B.children[2], tB, *A0xB2, no_of_block_multiplies, no_of_resizes);
								C.children[2] = A0xB2;
								C.children[2]->parent = &C;
							//	std::cout << " child2 set" << std::endl;
								get_batches_multiply(*A.children[0], tA, *B.children[2], tB, *A0xB2, batches, n_dummy_levels_to_pass);
							//	C.children[2] = NULL;
								//C.children[2]->parent = NULL;
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
							if(A.children[2] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)){
								A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3, no_of_block_multiplies, no_of_resizes);
								C.children[2] = A2xB3;
								C.children[2]->parent = &C;
							//	std::cout << " child2 set" << std::endl;
								get_batches_multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3, batches, n_dummy_levels_to_pass);
								//C.children[2] = NULL;
								//C.children[2]->parent = NULL;
							}

							if(A0xB2 != NULL && A2xB3 == NULL){C.children[2] = A0xB2; C.children[2]->parent = &C;}
							if(A0xB2 == NULL && A2xB3 != NULL){C.children[2] = A2xB3; C.children[2]->parent = &C;}
              if(A0xB2 != NULL && A2xB3 != NULL){
			                   // printf("mul: before 3rd add \n");
								collide_trees(*A0xB2, *A2xB3);
			                   // printf("mul: after 3rd add \n");
								C.children[2] = A0xB2;
								C.children[2]->parent = &C;
              }


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB2;
							if(A.children[1] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)){
								A1xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[1], tA, *B.children[2], tB, *A1xB2, no_of_block_multiplies, no_of_resizes);
								C.children[3] = A1xB2;
								C.children[3]->parent = &C;
								//std::cout << " child3 set" << std::endl;
								get_batches_multiply(*A.children[1], tA, *B.children[2], tB, *A1xB2, batches, n_dummy_levels_to_pass);
								//C.children[3] = NULL;
								//C.children[3]->parent = NULL;

							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
							if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
								A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3, no_of_block_multiplies, no_of_resizes);
								C.children[3] = A3xB3;
								C.children[3]->parent = &C;
							//	std::cout << " child3 set" << std::endl;
								get_batches_multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3, batches, n_dummy_levels_to_pass);
								//C.children[3] = NULL;
								//C.children[3]->parent = NULL;
							}

							if(A1xB2 != NULL && A3xB3 == NULL){C.children[3] = A1xB2; C.children[3]->parent = &C;}
							if(A1xB2 == NULL && A3xB3 != NULL){C.children[3] = A3xB3; C.children[3]->parent = &C;}
							if(A1xB2 != NULL && A3xB3 != NULL){
			                    //printf("mul: before 4th add \n");
								collide_trees(*A1xB2, *A3xB3);
			                   // printf("mul: after 4th add \n");
								C.children[3] = A1xB2;
								C.children[3]->parent = &C;
              }




							if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
								throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::get_batches_multiply(): resulting matrix is not consistent, smth went wrong!");
							}
							if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

							return;
						}


/*
						if(!tA && !tB){


							// C0 = A0xB0 + A2xB1
							// C1 = A1xB0 + A3xB1
							// C2 = A0xB2 + A2xB3
							// C3 = A1xB2 + A3xB3


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB0;
							if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
								A0xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0, no_of_block_multiplies, no_of_resizes);
								C.children[0] = A0xB0;
								C.children[0]->parent = &C;
								std::cout << "qweqwe " << n_dummy_levels << std::endl;
								get_batches_multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0, batches, n_dummy_levels_to_pass);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB1;
							if(A.children[2] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)){
								A2xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[2], tA, *B.children[1], tB, *A2xB1, no_of_block_multiplies, no_of_resizes);
								C.children[0] = A2xB1; C.children[0]->parent = &C;
								get_batches_multiply(*A.children[2], tA, *B.children[1], tB, *A2xB1, batches, n_dummy_levels_to_pass);
							}

			        if( A0xB0 != NULL && A2xB1 != NULL){
                    //add_to_first(*A0xB0, *A2xB1);
										// batches have already been created, need to change pointers
                   	C.children[0] = A0xB0;
                    C.children[0]->parent = &C;
              }


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0;
							if(A.children[1] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)){
								A1xB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0, no_of_block_multiplies, no_of_resizes);
								C.children[1] = A1xB0;
								C.children[1]->parent = &C;
								get_batches_multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0, batches, n_dummy_levels_to_pass);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB1;
							if(A.children[3] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[3], tA,*B.children[1], tB)){
								A3xB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[3], tA, *B.children[1], tB, *A3xB1, no_of_block_multiplies, no_of_resizes);
								C.children[1] = A3xB1;
								C.children[1]->parent = &C;
								get_batches_multiply(*A.children[3], tA, *B.children[1], tB, *A3xB1, batches, n_dummy_levels_to_pass);
							}


              if(A1xB0 != NULL && A3xB1 != NULL){
								//add_to_first(*A1xB0, *A3xB1);
								C.children[1] = A1xB0;
								C.children[1]->parent = &C;
              }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB2;
							if(A.children[0] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[0], tA,*B.children[2], tB)){
								A0xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[0], tA, *B.children[2], tB, *A0xB2, no_of_block_multiplies, no_of_resizes);
								C.children[2] = A0xB2; C.children[2]->parent = &C;
								get_batches_multiply(*A.children[0], tA, *B.children[2], tB, *A0xB2, batches, n_dummy_levels_to_pass);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3;
							if(A.children[2] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)){
								A2xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3, no_of_block_multiplies, no_of_resizes);
								C.children[2] = A2xB3; C.children[2]->parent = &C;
								get_batches_multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3, batches, n_dummy_levels_to_pass);
							}

							if(A0xB2 != NULL && A2xB3 != NULL){
								//add_to_first(*A0xB2, *A2xB3);
								C.children[2] = A0xB2;
								C.children[2]->parent = &C;
              }


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB2;
							if(A.children[1] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)){
								A1xB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[1], tA, *B.children[2], tB, *A1xB2, no_of_block_multiplies, no_of_resizes);
								C.children[3] = A1xB2; C.children[3]->parent = &C;
								get_batches_multiply(*A.children[1], tA, *B.children[2], tB, *A1xB2, batches, n_dummy_levels_to_pass);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3;
							if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
								A3xB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								//multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3, no_of_block_multiplies, no_of_resizes);
								C.children[3] = A3xB3; C.children[3]->parent = &C;
								get_batches_multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3, batches, n_dummy_levels_to_pass);
							}

							if(A1xB2 != NULL && A3xB3 != NULL){
								//add_to_first(*A1xB2, *A3xB3);
								C.children[3] = A1xB2;
								C.children[3]->parent = &C;
              }



							if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
								throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
							}
							if(squeeze_needed){
								remove_dummy_levels(C, n_dummy_levels);
							}

							return;
						}

*/

/*
						if(!tA && tB){

							// C0 = A0xB0^T + A2xB2^T
							// C1 = A1xB0^T + A3xB2^T
							// C2 = A0xB1^T + A2xB3^T
							// C3 = A1xB^T + A3xB3^T

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> >  A0xB0T;
							if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
								A0xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[0], tB, *A0xB0T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB2T;
							if(A.children[2] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[2], tB)){
								A2xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[2], tB, *A2xB2T, no_of_block_multiplies, no_of_resizes);
							}

							if(A0xB0T != NULL && A2xB2T == NULL){C.children[0] = A0xB0T; C.children[0]->parent = &C;}
							if(A0xB0T == NULL && A2xB2T != NULL){C.children[0] = A2xB2T; C.children[0]->parent = &C;}
			                if(A0xB0T != NULL && A2xB2T != NULL){
			                    add_to_first(*A0xB0T, *A2xB2T);
								C.children[0] = A0xB0T;
			                    C.children[0]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB0T;
							if(A.children[1] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[0], tB)){
								A1xB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[0], tB, *A1xB0T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB2T;
							if(A.children[3] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[2], tB)){
								A3xB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[2], tB, *A3xB2T, no_of_block_multiplies, no_of_resizes);
							}

							if(A1xB0T != NULL && A3xB2T == NULL){C.children[1] = A1xB0T; C.children[1]->parent = &C;}
							if(A1xB0T == NULL && A3xB2T != NULL){C.children[1] = A3xB2T; C.children[1]->parent = &C;}
			                if(A1xB0T != NULL && A3xB2T != NULL){
								add_to_first(*A1xB0T, *A3xB2T);
			                    C.children[1] = A1xB0T;
			                    C.children[1]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0xB1T;
							if(A.children[0] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[1], tB)){
								A0xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[1], tB, *A0xB1T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2xB3T;
							if(A.children[2] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[3], tB)){
								A2xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[3], tB, *A2xB3T, no_of_block_multiplies, no_of_resizes);
							}

							if(A0xB1T != NULL && A2xB3T == NULL){C.children[2] = A0xB1T; C.children[2]->parent = &C;}
							if(A0xB1T == NULL && A2xB3T != NULL){C.children[2] = A2xB3T; C.children[2]->parent = &C;}
			                if(A0xB1T != NULL && A2xB3T != NULL){
								add_to_first(*A0xB1T, *A2xB3T);
			                    C.children[2] = A0xB1T;
			                    C.children[2]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1xB1T;
							if(A.children[1] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[1], tB)){
								A1xB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[1], tB, *A1xB1T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3xB3T;
							if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
								A3xB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[3], tB, *A3xB3T, no_of_block_multiplies, no_of_resizes);
							}

			                if(A1xB1T != NULL && A3xB3T == NULL){C.children[3] = A1xB1T; C.children[3]->parent = &C;}
							if(A1xB1T == NULL && A3xB3T != NULL){C.children[3] = A3xB3T; C.children[3]->parent = &C;}
			                if(A1xB1T != NULL && A3xB3T != NULL){
			                    add_to_first(*A1xB1T, *A3xB3T);
								C.children[3] = A1xB1T;
			                    C.children[3]->parent = &C;
			                }

							if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
								throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
							}
							if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

							return;
						}


						if(tA && !tB){
							// C0 = A0^TB0 + A1^TB1
							// C1 = A2^TB0 + A3^TB1
							// C2 = A0^TB2 + A1^TB3
							// C3 = A2^TB2 + A3^TB3

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0;
							if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
								A0TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[0], tB, *A0TxB0, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB1;
							if(A.children[1] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[1], tB)){
								A1TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[1], tB, *A1TxB1, no_of_block_multiplies, no_of_resizes);
							}

							if(A0TxB0 != NULL && A1TxB1 == NULL){C.children[0] = A0TxB0;  C.children[0]->parent = &C;}
						    if(A0TxB0 == NULL && A1TxB1 != NULL){C.children[0] = A1TxB1;  C.children[0]->parent = &C;}
			                if(A0TxB0 != NULL && A1TxB1 != NULL){
								add_to_first(*A0TxB0,*A1TxB1);
			                    C.children[0] = A0TxB0;
			                    C.children[0]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0;
							if(A.children[2] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[0], tB)){
								A2TxB0 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[0], tB, *A2TxB0, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB1;
							if(A.children[3] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[1], tB)){
								A3TxB1 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[1], tB, *A3TxB1, no_of_block_multiplies, no_of_resizes);
							}

							if(A2TxB0 != NULL && A3TxB1 == NULL){C.children[1] = A2TxB0; C.children[1]->parent = &C;}
							if(A2TxB0 == NULL && A3TxB1 != NULL){C.children[1] = A3TxB1; C.children[1]->parent = &C;}
			                if(A2TxB0 != NULL && A3TxB1 != NULL){
			                    add_to_first(*A2TxB0,*A3TxB1);
								C.children[1] = A2TxB0;
			                    C.children[1]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB2;
							if(A.children[0] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[2], tB)){
								A0TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[2], tB, *A0TxB2, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3;
							if(A.children[1] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[3], tB)){
								A1TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[3], tB, *A1TxB3, no_of_block_multiplies, no_of_resizes);
							}

			                if(A0TxB2 != NULL && A1TxB3 == NULL){C.children[2] = A0TxB2; C.children[2]->parent = &C;}
							if(A0TxB2 == NULL && A1TxB3 != NULL){C.children[2] = A1TxB3; C.children[2]->parent = &C;}
			                if(A0TxB2 != NULL && A1TxB3 != NULL){
			                    add_to_first(*A0TxB2,*A1TxB3);
								C.children[2] = A0TxB2;
			                    C.children[2]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB2;
							if(A.children[2] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[2], tB)){
								A2TxB2 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[2], tB, *A2TxB2, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3;
							if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
								A3TxB3 = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[3], tB, *A3TxB3, no_of_block_multiplies, no_of_resizes);
							}

							if(A2TxB2 != NULL && A3TxB3 == NULL){C.children[3] = A2TxB2; C.children[3]->parent = &C;}
							if(A2TxB2 == NULL && A3TxB3 != NULL){C.children[3] = A3TxB3; C.children[3]->parent = &C;}
			                if(A2TxB2 != NULL && A3TxB3 != NULL){
			                    add_to_first(*A2TxB2,*A3TxB3);
								C.children[3] = A2TxB2;
			                    C.children[3]->parent = &C;
			                }

							if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
								throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
							}
							if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

							return;
						}


						if(tA && tB){
							// C0 = A0^TB0^T + A1^TB2^T
							// C1 = A2^TB0^T + A3^TB2^T
							// C2 = A0^TB1^T + A1^TB3^T
							// C3 = A2^TB1^T + A3^TB3^T


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB0T;
							if(A.children[0] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[0], tB)){
								A0TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[0], tB, *A0TxB0T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB2T;
							if(A.children[1] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[2], tB)){
								A1TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[2], tB, *A1TxB2T, no_of_block_multiplies, no_of_resizes);
							}

							if(A0TxB0T != NULL && A1TxB2T == NULL){C.children[0] = A0TxB0T; C.children[0]->parent = &C;}
							if(A0TxB0T == NULL && A1TxB2T != NULL){C.children[0] = A1TxB2T; C.children[0]->parent = &C;}
			                if(A0TxB0T != NULL && A1TxB2T != NULL){
			                    add_to_first(*A0TxB0T,*A1TxB2T);
								C.children[0] = A0TxB0T;
			                    C.children[0]->parent = &C;
			                }


							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB0T;
							if(A.children[2] != NULL && B.children[0] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[0], tB)){
								A2TxB0T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[0], tB, *A2TxB0T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB2T;
							if(A.children[3] != NULL && B.children[2] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[2], tB)){
								A3TxB2T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[2], tB, *A3TxB2T, no_of_block_multiplies, no_of_resizes);
							}

							if(A2TxB0T != NULL && A3TxB2T == NULL){C.children[1] = A2TxB0T; C.children[1]->parent = &C;}
							if(A2TxB0T == NULL && A3TxB2T != NULL){C.children[1] = A3TxB2T; C.children[1]->parent = &C;}
			                if(A2TxB0T != NULL && A3TxB2T != NULL){
			                    add_to_first(*A2TxB0T,*A3TxB2T);
								C.children[1] = A2TxB0T;
			                    C.children[1]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A0TxB1T;
							if(A.children[0] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[0], tA, *B.children[1], tB)){
								A0TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[0], tA, *B.children[1], tB, *A0TxB1T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A1TxB3T;
							if(A.children[1] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[1], tA, *B.children[3], tB)){
								A1TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[1], tA, *B.children[3], tB, *A1TxB3T, no_of_block_multiplies, no_of_resizes);
							}

							if(A0TxB1T != NULL && A1TxB3T == NULL){C.children[2] = A0TxB1T; C.children[2]->parent = &C;}
							if(A0TxB1T == NULL && A1TxB3T != NULL){C.children[2] = A1TxB3T; C.children[2]->parent = &C;}
			                if(A0TxB1T != NULL && A1TxB3T != NULL){
			                    add_to_first(*A0TxB1T,*A1TxB3T);
								C.children[2] = A0TxB1T;
			                    C.children[2]->parent = &C;
			                }

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A2TxB1T;
							if(A.children[2] != NULL && B.children[1] != NULL && worth_to_multiply(*A.children[2], tA, *B.children[1], tB)){
								A2TxB1T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[2], tA, *B.children[1], tB, *A2TxB1T, no_of_block_multiplies, no_of_resizes);
							}

							std::shared_ptr<HierarchicalBlockSparseMatrix<Treal> > A3TxB3T;
							if(A.children[3] != NULL && B.children[3] != NULL && worth_to_multiply(*A.children[3], tA, *B.children[3], tB)){
								A3TxB3T = std::make_shared<HierarchicalBlockSparseMatrix<Treal> >();
								multiply(*A.children[3], tA, *B.children[3], tB, *A3TxB3T, no_of_block_multiplies, no_of_resizes);
							}

							if(A2TxB1T != NULL && A3TxB3T == NULL){C.children[3] = A2TxB1T; C.children[3]->parent = &C;}
							if(A2TxB1T == NULL && A3TxB3T != NULL){C.children[3] = A3TxB3T; C.children[3]->parent = &C;}
			                if(A2TxB1T != NULL && A3TxB3T != NULL){
			                    add_to_first(*A2TxB1T,*A3TxB3T);
								C.children[3] = A2TxB1T;
			                    C.children[3]->parent = &C;
			                }

							if(A.get_level() != 0 && B.get_level() != 0 && !C.check_if_matrix_is_consistent()){
								throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::multiply(): resulting matrix is not consistent, smth went wrong!");
							}
							if(squeeze_needed) remove_dummy_levels(C, n_dummy_levels);

							return;

						}



						return;
*/
				}

			template<class Treal>
				HierarchicalBlockSparseMatrix<Treal>* HierarchicalBlockSparseMatrix<Treal>::get_matrix_ptr_by_position_code(const std::string & position_code){
					int length = position_code.length();
					HierarchicalBlockSparseMatrix<Treal>* ptr = NULL, *cur_ptr = NULL;
					//std::cout << "Inside get_matrix_ptr_by_position_code, code: " << position_code << std::endl;

					cur_ptr = this;

					for(int k = 0; k < length; ++k){
						int pos = position_code[k] - '0';

						/*std::cout << "level k = " << k << std::endl;

						if(cur_ptr->children[0] != NULL) std::cout << "Child0 exists" << std::endl;
						if(cur_ptr->children[1] != NULL) std::cout << "Child1 exists" << std::endl;
						if(cur_ptr->children[2] != NULL) std::cout << "Child2 exists" << std::endl;
						if(cur_ptr->children[3] != NULL) std::cout << "Child3 exists" << std::endl;

						std::cout << "go to child " << pos << std::endl;*/

						switch(pos){
							case 0:
								assert(cur_ptr->children[0] != NULL);
								cur_ptr = cur_ptr->children[0].get();
								break;
							case 1:
								assert(cur_ptr->children[1] != NULL);
								cur_ptr = cur_ptr->children[1].get();
								break;
							case 2:
								assert(cur_ptr->children[2] != NULL);
								cur_ptr = cur_ptr->children[2].get();
								break;
							case 3:
								assert(cur_ptr->children[3] != NULL);
								cur_ptr = cur_ptr->children[3].get();
								break;
						}

					}

					return cur_ptr;
				}

		template<class Treal>
		 void HierarchicalBlockSparseMatrix<Treal>::multiply_batches(HierarchicalBlockSparseMatrix<Treal> const& A, bool tA, HierarchicalBlockSparseMatrix<Treal> const& B, bool tB,
			HierarchicalBlockSparseMatrix<Treal>& C, BatchMapMultiply & batches){

				const Treal ONE = 1.0;

				for(auto i = batches.begin(); i != batches.end(); ++i){
					std::string position_code = i->first;

					//std::cout << "Inside multiply_batches, code " << position_code << std::endl;

					//std::cout << "C depth = " << C.get_depth() << std::endl;

					const HierarchicalBlockSparseMatrix<Treal> *ptr_A = i->second.a;
					const HierarchicalBlockSparseMatrix<Treal> *ptr_B = i->second.b;

					//std::cout << "A:" << std::endl;
					//ptr_A->print();

				//	std::cout << "B:" << std::endl;
				//	ptr_B->print();

					const Treal *aptr = ptr_A->get_submatrix_ptr();
					const Treal *bptr = ptr_B->get_submatrix_ptr();
					Treal *cptr;
					bool tA = i->second.ta;
					bool tB = i->second.tb;

					if(C.lowest_level()){
						cptr = C.get_submatrix_ptr_for_modification();
					}
					else{
						//std::cout << "NOT lowest_level " << std::endl;
						HierarchicalBlockSparseMatrix<Treal> *ptr_C = C.get_matrix_ptr_by_position_code(i->first);
						cptr = ptr_C->get_submatrix_ptr_for_modification();
					}







					int blocksize = C.blocksize;




					if(!tA && !tB){
						gemm(Transpose::N().bt, Transpose::N().bt, &blocksize, &blocksize, &blocksize, &ONE, aptr, &blocksize, bptr, &blocksize, &ONE, cptr, &blocksize);
					}

/*
					if(!tA && tB){
						gemm(Transpose::N().bt, Transpose::T().bt, &blocksize, &blocksize, &blocksize, &ONE, aptr, &blocksize, bptr, &blocksize, &ONE, cptr, &blocksize);
					}

					if(tA && !tB){
						gemm(Transpose::T().bt, Transpose::N().bt, &blocksize, &blocksize, &blocksize, &ONE, aptr, &blocksize, bptr, &blocksize, &ONE, cptr, &blocksize);
					}

					if(tA && tB){
						gemm(Transpose::T().bt, Transpose::T().bt, &blocksize, &blocksize, &blocksize, &ONE, aptr, &blocksize, bptr, &blocksize, &ONE, cptr, &blocksize);
					}*/



				}



			}

			template<class Treal>
				void HierarchicalBlockSparseMatrix<Treal>::collide_trees(HierarchicalBlockSparseMatrix<Treal> & first, HierarchicalBlockSparseMatrix<Treal> & second){

					if(first.nRows_orig != second.nRows_orig || first.nCols_orig != second.nCols_orig ){
						throw std::runtime_error("Error in HierarchicalBlockSparseMatrix::collide_trees(): matrices to add have different sizes!");
					}

					if(first.lowest_level()){
						return;
					}

					for(int i = 0; i < 4; ++i){
						if(first.children[i] != NULL && second.children[i] == NULL){
							continue;
						}
						if(first.children[i] == NULL && second.children[i] != NULL) {
							first.children[i] = second.children[i];
							first.children[i]->parent = &first;
							continue;
						}
						if(first.children[i] != NULL && second.children[i] != NULL){
							collide_trees(*first.children[i],  *second.children[i]);
						}

					}


				}

} /* end namespace hbsm */

#endif
