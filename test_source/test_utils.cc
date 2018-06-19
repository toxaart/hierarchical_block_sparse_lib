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
#include "test_utils.h"

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2) {
  std::vector<double> row(2);
  row[0] = x1;
  row[1] = x2;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3) {
  std::vector<double> row(3);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4) {
  std::vector<double> row(4);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  row[3] = x4;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5) {
  std::vector<double> row(5);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  row[3] = x4;
  row[4] = x5;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6) {
  std::vector<double> row(6);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  row[3] = x4;
  row[4] = x5;
  row[5] = x6;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6, double x7) {
  std::vector<double> row(7);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  row[3] = x4;
  row[4] = x5;
  row[5] = x6;
  row[6] = x7;
  A.set_row(rowIdx, row);
}

void set_row(SparseMatrix & A, int rowIdx, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8) {
  std::vector<double> row(8);
  row[0] = x1;
  row[1] = x2;
  row[2] = x3;
  row[3] = x4;
  row[4] = x5;
  row[5] = x6;
  row[6] = x7;
  row[7] = x8;
  A.set_row(rowIdx, row);
}
