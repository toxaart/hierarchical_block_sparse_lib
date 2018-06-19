/* Block-Sparse-Matrix-Lib, version 1.0. A block sparse matrix library.
 * Copyright (C) Emanuel H. Rubensson <emanuelrubensson@gmail.com>,
 *               Elias Rudberg <eliasrudberg@gmail.com>, and
 *               Anastasia Kruchinina <anastasia.kruchinina@it.uu.se>.
 * 
 * Distribution without copyright owners' explicit consent prohibited.
 * 
 * This source code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <vector>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>

static bool const verbose = false;
struct SparseMatrix {
  std::vector<int> rows;
  std::vector<int> cols;
  std::vector<double> values;
  void set_row(int rowIdx, std::vector<double> row_values) {
    int size_new = rows.size()+row_values.size();
    rows.reserve(size_new);
    cols.reserve(size_new);
    values.reserve(size_new);
    for (size_t ind = 0; ind < row_values.size(); ind++) {
      rows.push_back(rowIdx);
      cols.push_back(ind);
      values.push_back(row_values[ind]);
    }
  }
  
  template<typename MatrixType>
  void assign(MatrixType & A) const {
    A.assign_from_vectors(rows, cols, values);
  }
  void to_full(std::vector<double> & fullMatrix, unsigned int nRows, unsigned int nCols) {
    fullMatrix.resize( nRows*nCols );
    for(unsigned int ind=0; ind<nRows*nCols; ind++)
      fullMatrix[ind] = 0;
    for(unsigned int ind=0; ind<rows.size(); ind++)
      fullMatrix[rows[ind]+nRows*cols[ind]] = values[ind];
  }
};

template<typename MatrixType>
void verify_that_matrices_are_equal(const MatrixType & A,
				    const MatrixType & B) {
  if(A.get_n_rows() != B.get_n_rows() || A.get_n_cols() != B.get_n_cols())
    throw std::runtime_error("Error: matrices do not have same dimensions.");
  SparseMatrix spA;
  A.get_all_values(spA.rows, spA.cols, spA.values);
  std::vector<double> fullMatrixA;
  spA.to_full(fullMatrixA, A.get_n_rows(), A.get_n_cols());
  SparseMatrix spB;
  B.get_all_values(spB.rows, spB.cols, spB.values);
  std::vector<double> fullMatrixB;
  spB.to_full(fullMatrixB, B.get_n_rows(), B.get_n_cols());

  assert( fullMatrixA.size() == fullMatrixB.size() );
  for(unsigned int ind=0; ind<fullMatrixA.size(); ind++) {
    if (verbose)
      std::cout << std::setw(5) << ind << "  " 
		<< std::setw(14) << fullMatrixA[ind] << "  " 
		<< std::setw(14) << fullMatrixB[ind] 
		<< std::endl;
    if(fullMatrixA[ind] != fullMatrixB[ind])
      throw std::runtime_error("Error: matrices not equal.");
  }
}

template<typename MatrixType>
void verify_that_matrices_are_almost_equal(const MatrixType & A,
					   const MatrixType & B,
					   double tolerance) {
  if(A.get_n_rows() != B.get_n_rows() || A.get_n_cols() != B.get_n_cols())
    throw std::runtime_error("Error: matrices do not have same dimensions.");
  SparseMatrix spA;
  A.get_all_values(spA.rows, spA.cols, spA.values);
  std::vector<double> fullMatrixA;
  spA.to_full(fullMatrixA, A.get_n_rows(), A.get_n_cols());
  SparseMatrix spB;
  B.get_all_values(spB.rows, spB.cols, spB.values);
  std::vector<double> fullMatrixB;
  spB.to_full(fullMatrixB, B.get_n_rows(), B.get_n_cols());

  assert( fullMatrixA.size() == fullMatrixB.size() );
  for(unsigned int ind=0; ind<fullMatrixA.size(); ind++) {
    if (verbose)
      std::cout << std::setw(5) << ind << "  " 
		<< std::setw(14) << fullMatrixA[ind] << "  " 
		<< std::setw(14) << fullMatrixB[ind] 
		<< std::endl;
    if(std::fabs(fullMatrixA[ind] - fullMatrixB[ind]) > tolerance)
      throw std::runtime_error("Error: matrices not almost equal.");
  }
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6, double x7);
void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8);
