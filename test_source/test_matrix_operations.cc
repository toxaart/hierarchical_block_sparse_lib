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
#include <iostream>
#include <iomanip>
#include <cmath>
#include "hierarchical_block_sparse_lib.h"
#include "test_utils.h"

template<typename MatrixType>
static void test_symm_multiply(typename MatrixType::Params const & param) {
	MatrixType A;
	A.set_params(param);
	A.resize(5, 5);
	{
		SparseMatrix As;
		set_row(As, 0, 2, 2, 3, 5, 2);
		set_row(As, 1, 0, 1, 2, 4, 3);
		set_row(As, 2, 0, 0, 3, 1, 2);
		set_row(As, 3, 0, 0, 0, 4, 5);
		set_row(As, 4, 0, 0, 0, 0, 1);
		As.assign(A);
	}
	MatrixType B;
	B.set_params(param);
	B.resize(5, 7);
	{
		SparseMatrix Bs;
		set_row(Bs, 0, 5, 3, 1, 5, 0, 3, 3);
		set_row(Bs, 1, 1, 5, 5, 4, 1, 5, 1);
		set_row(Bs, 2, 2, 1, 3, 2, 1, 1, 4);
		set_row(Bs, 3, 2, 2, 3, 3, 1, 4, 2);
		set_row(Bs, 4, 5, 1, 1, 2, 1, 2, 3);
		Bs.assign(B);
	}
	MatrixType C;
	C.set_params(param);
	C.resize(2, 5);
	{
		SparseMatrix Cs;
		set_row(Cs, 0, 5, 3, 0, 3, 3);
		set_row(Cs, 1, 1, 4, 1, 5, 1);
		Cs.assign(C);
	}
	if (verbose)
	std::cout << "Test symm_multiply 1" << std::endl;




	// A * B
	MatrixType AxB;
	MatrixType::symm_multiply(A, true, B, false, AxB);
	MatrixType AxBref;
	AxBref.set_params(param);
	AxBref.resize(5, 7);
	{
		SparseMatrix tmp;
		set_row(tmp, 0,    38,    31,    38,    43,    12,    43,    36);
		set_row(tmp, 1,    38,    24,    28,    36,    10,    35,    32);
		set_row(tmp, 2,    35,    26,    27,    36,     8,    30,    31);
		set_row(tmp, 3,    64,    49,    45,    65,    14,    62,    46);
		set_row(tmp, 4,    32,    34,    39,    43,    11,    45,    30);
		tmp.assign(AxBref);
	}


	verify_that_matrices_are_equal(AxB, AxBref);


	if (verbose)
	std::cout << "Test symm_multiply 2" << std::endl;
	// C * A
	MatrixType CxA;
	MatrixType::symm_multiply(C, false, A, true, CxA);
	MatrixType CxAref;
	CxAref.set_params(param);
	CxAref.resize(2, 5);
	{
		SparseMatrix tmp;
		set_row(tmp, 0,   37,    34,    30,    64,    37);
		set_row(tmp, 1,   40,    31,    21,    47,    42);
		tmp.assign(CxAref);
	}
	verify_that_matrices_are_equal(CxA, CxAref);
}

template<typename MatrixType>
static void test_symm_square(typename MatrixType::Params const & param) {
  MatrixType A;
  A.set_params(param);
  A.resize(5, 5);
  {
    SparseMatrix tmp;
    set_row(tmp, 0, 2, 2, 3, 5, 2);
    set_row(tmp, 1, 0, 1, 2, 4, 3);
    set_row(tmp, 2, 0, 0, 3, 1, 2);
    set_row(tmp, 3, 0, 0, 0, 4, 5);
    set_row(tmp, 4, 0, 0, 0, 0, 1);
    tmp.assign(A);
  }
  if (verbose)
    std::cout << "Test symm_square" << std::endl;
  // A * A
  MatrixType AxA;
  MatrixType::symm_square(A, AxA);

  MatrixType AxAref;
  AxAref.set_params(param);
  AxAref.resize(5, 5);
  {
    SparseMatrix tmp;
    set_row(tmp, 0,    46,    38,    28,    51,    43);
    set_row(tmp, 1,     0,    34,    24,    47,    34);
    set_row(tmp, 2,     0,     0,    27,    40,    25);
    set_row(tmp, 3,     0,     0,     0,    83,    49);
    set_row(tmp, 4,     0,     0,     0,     0,    43);
    tmp.assign(AxAref);
  }

  verify_that_matrices_are_equal(AxA, AxAref);



}

template<typename MatrixType>
static void test_symm_rk(typename MatrixType::Params const & param) {
  MatrixType A;
  A.set_params(param);
  A.resize(2, 3);
  {
    SparseMatrix tmp;
    set_row(tmp, 0, 2, 3, 5);
    set_row(tmp, 1, 0, 1, 2);
    tmp.assign(A);
  }



  if (verbose)
    std::cout << "Test symm_rk C = A * A'" << std::endl;
  // A * A'
  MatrixType AxAt;
  MatrixType::symm_rk(A, false, AxAt);
  MatrixType AxAtref;
  AxAtref.set_params(param);
  AxAtref.resize(2, 2);
  {
    SparseMatrix tmp;
    set_row(tmp, 0,    38,    13);
    set_row(tmp, 1,     0,     5);
    tmp.assign(AxAtref);
  }

  verify_that_matrices_are_equal(AxAt, AxAtref);

  if (verbose)
    std::cout << "Test symm_rk C = A' * A" << std::endl;
  // A' * A
  MatrixType AtxA;
  MatrixType::symm_rk(A, true, AtxA);
  MatrixType AtxAref;
  AtxAref.set_params(param);
  AtxAref.resize(3, 3);
  {
    SparseMatrix tmp;
    set_row(tmp, 0,     4,     6,    10);
    set_row(tmp, 1,     0,    10,    17);
    set_row(tmp, 2,     0,     0,    29);
    tmp.assign(AtxAref);
  }

  verify_that_matrices_are_equal(AtxA, AtxAref);

}

template<typename MatrixType>
static void test_invchol() {
  if (verbose)
    std::cout << "Test test_invchol()" << std::endl;
  MatrixType A;
  MatrixType Z;
  typename MatrixType::Params param;
  param.blocksize = 2;
 //set_block_size<MatrixType>(param, 2);//Has effect only for block sparse matrix
  A.set_params(param);
  A.resize(5, 5);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 5, 0, 1, 0, 0 );
      set_row(tmp, 1, 0, 6, 2, 0, 0 );
      set_row(tmp, 2, 1, 2, 5, 3, 0 );
      set_row(tmp, 3, 0, 0, 3, 8, 0 );
      set_row(tmp, 4, 0, 0, 0, 0, 7 );
      tmp.assign(A);
    }


    MatrixType::inv_chol(A, Z);
    MatrixType Z_ref;
    Z_ref.set_params(param);
    Z_ref.resize(5,5);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 0.447213595499958,                  0, -0.098373875367593,  0.060157954894827,                  0 );
      set_row(tmp, 1, 0,                  0.408248290463863, -0.163956458945988,  0.100263258158045,                  0 );
      set_row(tmp, 2, 0,                                  0,  0.491869376837965, -0.300789774474136,                  0 );
      set_row(tmp, 3, 0,                                  0,                  0,  0.414421467053253,                  0 );
      set_row(tmp, 4, 0,                                  0,                  0,                  0,  0.377964473009227 );
      tmp.assign(Z_ref);
    }
    verify_that_matrices_are_almost_equal(Z, Z_ref, 1e-10);
}

template<typename MatrixType>
static int test_operations() {
  typename MatrixType::Params param;
#if 1
	param.blocksize = 2; // Only for block sparse matrix
#endif

  // Test add()
	size_t n_resizes = 0;
	size_t n_multiplications;

	MatrixType A;
	A.set_params(param);
	A.resize(2, 3, &n_resizes);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 2, 3, 5);
		set_row(tmp, 1, 0, 1, 2);
		tmp.assign(A);
	}

	MatrixType AT;
	MatrixType::transpose(A,AT);

	MatrixType AT_ref;
	AT_ref.set_params(param);
	AT_ref.resize(3, 2);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 2, 0);
		set_row(tmp, 1, 3, 1);
		set_row(tmp, 2, 5, 2);
		tmp.assign(AT_ref);
	}

	verify_that_matrices_are_equal(AT, AT_ref);

	MatrixType B;
	B.set_params(param);
	B.resize(2, 3, &n_resizes);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 1, 3, 2);
		set_row(tmp, 1, 6, 2, 4);
		tmp.assign(B);
	}

	MatrixType C;
	C.set_params(param);
	MatrixType::add(A, B, C, &n_resizes);

	MatrixType Cref;
	Cref.set_params(param);
	Cref.resize(2, 3);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 3, 6, 7);
		set_row(tmp, 1, 6, 3, 6);
		tmp.assign(Cref);
	}
	verify_that_matrices_are_equal(C, Cref);

	// Test multiply()

	MatrixType D;
	D.set_params(param);
	D.resize(3, 2);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 2, 1);
		set_row(tmp, 1, 7, 3);
		set_row(tmp, 2, 3, 5);
		tmp.assign(D);
	}

	MatrixType AxD;
	MatrixType::multiply(A, false, D, false, AxD, &n_multiplications, &n_resizes);

	MatrixType AxDref;
	AxDref.set_params(param);
	AxDref.resize(2, 2);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 40, 36);
		set_row(tmp, 1, 13, 13);
		tmp.assign(AxDref);
	}

    verify_that_matrices_are_equal(AxD, AxDref);

	MatrixType DxA;
	MatrixType::multiply(D, false, A, false, DxA);

	MatrixType DxAref;
	DxAref.set_params(param);
	DxAref.resize(3, 3);
	{
	SparseMatrix tmp;
	set_row(tmp, 0,  4,  7, 12);
	set_row(tmp, 1, 14, 24, 41);
	set_row(tmp, 2,  6, 14, 25);
	tmp.assign(DxAref);
	}


	verify_that_matrices_are_equal(DxA, DxAref);


	if (verbose)
		std::cout << "Test multiply NT" << std::endl;
	MatrixType AxAt;
	MatrixType::multiply(A, false, A, true, AxAt);

	MatrixType AxAt_ref;
	AxAt_ref.set_params(param);
	AxAt_ref.resize(2, 2);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, 38, 13);
		set_row(tmp, 1, 13,  5);
		tmp.assign(AxAt_ref);
	}

	verify_that_matrices_are_equal(AxAt, AxAt_ref);

	if (verbose)
		std::cout << "Test multiply TN" << std::endl;

	MatrixType AtxA;
	MatrixType::multiply(A, true, A, false, AtxA);

	MatrixType AtxA_ref;
	AtxA_ref.set_params(param);
	AtxA_ref.resize(3, 3);
	{
		SparseMatrix tmp;
		set_row(tmp, 0,  4,  6, 10);
		set_row(tmp, 1,  6, 10, 17);
		set_row(tmp, 2, 10, 17, 29);
		tmp.assign(AtxA_ref);
	}
	verify_that_matrices_are_equal(AtxA, AtxA_ref);

	if (verbose)
		std::cout << "Test multiply TT" << std::endl;
	MatrixType AtxDt;
	MatrixType::multiply(A, true, D, true, AtxDt);

	MatrixType AtxDt_ref;
	AtxDt_ref.set_params(param);
	AtxDt_ref.resize(3, 3);
	{
		SparseMatrix tmp;
		set_row(tmp, 0,  4, 14,  6);
		set_row(tmp, 1,  7, 24, 14);
		set_row(tmp, 2, 12, 41, 25);
		tmp.assign(AtxDt_ref);
	}

	verify_that_matrices_are_equal(AtxDt, AtxDt_ref);



	// test matrices of different numbers levels
	{
		param.blocksize = 1 ;

		MatrixType A;
		A.set_params(param);

		A.resize(1,1);

		std::vector<int> rowsA, colsA;
		std::vector<double> valsA;

		rowsA.push_back(0);
		colsA.push_back(0);
		valsA.push_back(2);

		A.assign_from_vectors(rowsA,colsA,valsA);
		printf("A depth %d \n", A.get_depth());

		//A.print();

		MatrixType B;
		B.set_params(param);
		B.resize(1,2);
		{
			SparseMatrix tmp;
			set_row(tmp, 0,  1, 2);
			tmp.assign(B);
		}
		printf("B depth %d \n", B.get_depth());


		MatrixType AxB;
		MatrixType::multiply(A, false, B, false, AxB);

		AxB.print();

		param.blocksize = 2 ;
	}


	// test matrices of different numbers levels
	{

		typename MatrixType::Params param2;
		param2.blocksize = 3;
		MatrixType A_2level; // A will have 2 levels
		A_2level.set_params(param2);
		A_2level.resize(5, 5);
		{
			SparseMatrix As;
			set_row(As, 0, 2, 2, 3, 5, 2);
			set_row(As, 1, 2, 1, 2, 4, 3);
			set_row(As, 2, 3, 2, 3, 1, 2);
			set_row(As, 3, 5, 4, 1, 4, 5);
			set_row(As, 4, 2, 3, 2, 5, 1);
			As.assign(A_2level);
		}
		MatrixType B_3level; // B will have 3 levels
		B_3level.set_params(param2);
		B_3level.resize(5, 7);
		{
			SparseMatrix Bs;
			set_row(Bs, 0, 5, 3, 1, 5, 0, 3, 3);
			set_row(Bs, 1, 1, 5, 5, 4, 1, 5, 1);
			set_row(Bs, 2, 2, 1, 3, 2, 1, 1, 4);
			set_row(Bs, 3, 2, 2, 3, 3, 1, 4, 2);
			set_row(Bs, 4, 5, 1, 1, 2, 1, 2, 3);
			Bs.assign(B_3level);
		}

		printf("A depth %d, B depth  %d \n", A_2level.get_depth(), B_3level.get_depth());

		printf("Worth ? %d \n", MatrixType::worth_to_multiply(A_2level, false, B_3level, false));

		MatrixType A_2level_times_B_level;
		MatrixType::multiply(A_2level, false, B_3level, false, A_2level_times_B_level);

		MatrixType A_2level_times_B_level_ref;
		A_2level_times_B_level_ref.set_params(param);
		A_2level_times_B_level_ref.resize(5, 7);
		{
			SparseMatrix tmp;
			set_row(tmp, 0,    38,    31,    38,    43,    12,    43,    36);
			set_row(tmp, 1,    38,    24,    28,    36,    10,    35,    32);
			set_row(tmp, 2,    35,    26,    27,    36,     8,    30,    31);
			set_row(tmp, 3,    64,    49,    45,    65,    14,    62,    46);
			set_row(tmp, 4,    32,    34,    39,    43,    11,    45,    30);
			tmp.assign(A_2level_times_B_level_ref);
		}

		verify_that_matrices_are_equal(A_2level_times_B_level_ref, A_2level_times_B_level);
		assert(A_2level_times_B_level_ref.get_depth() == A_2level_times_B_level.get_depth());


		MatrixType BT_3level_times_A_2level_ref; // B will have 3 levels
		BT_3level_times_A_2level_ref.set_params(param2);
		BT_3level_times_A_2level_ref.resize(7, 5);
		{
			SparseMatrix Bs;
			set_row(Bs, 0, 38,    38,    35,    64,    32);
			set_row(Bs, 1, 31,    24,   26,    49 ,   34);
			set_row(Bs, 2, 38,    28,   27,    45,    39);
			set_row(Bs, 3, 43,    36,    36,    65,    43);
			set_row(Bs, 4, 12,    10,    8,   14,    11);
			set_row(Bs, 5, 43,    35,    30,    62,    45);
			set_row(Bs, 6, 36,    32,   31,    46 ,   30);

			Bs.assign(BT_3level_times_A_2level_ref);
		}

		printf("Worth ? %d \n", MatrixType::worth_to_multiply(B_3level, true, A_2level, false));

		MatrixType BT_3level_times_A_2level;
    std::cout << "BT_3level_times_A_2level n muls before = " << BT_3level_times_A_2level.get_n_block_multiplications() << std::endl;
		MatrixType::multiply(B_3level, true, A_2level, false, BT_3level_times_A_2level);
    std::cout << "BT_3level_times_A_2level n muls after = " << BT_3level_times_A_2level.get_n_block_multiplications() << std::endl;
		verify_that_matrices_are_equal(BT_3level_times_A_2level, BT_3level_times_A_2level_ref);
		assert(BT_3level_times_A_2level.get_n_block_multiplications() == 12);

	}

	// test rescale

	MatrixType minus_A;
	minus_A.rescale(A, -1.0);

	MatrixType minus_A_ref;
	minus_A_ref.set_params(param);
	minus_A_ref.resize(2, 3);
	{
		SparseMatrix tmp;
		set_row(tmp, 0, -2, -3, -5);
		set_row(tmp, 1, -0, -1, -2);
		tmp.assign(minus_A_ref);
	}

	verify_that_matrices_are_equal(minus_A, minus_A_ref);



  if (verbose)
    std::cout << "Test inv_chol" << std::endl;
  // Test inv_chol()
  {
    MatrixType A;
    MatrixType Z;
    A.set_params(param);
    A.resize(4, 4);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 5.2, 0.1 , 0.4 , 0.7 );
      set_row(tmp, 1, 0.1, 5.8 , 0.25, 0.55);
      set_row(tmp, 2, 0.4, 0.25, 6.7 , 0.6 );
      set_row(tmp, 3, 0.7, 0.55, 0.6 , 5.5 );
      tmp.assign(A);
    }
    MatrixType::inv_chol(A, Z);
    MatrixType Z_ref;
    Z_ref.set_params(param);
    Z_ref.resize(4,4);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 0.438529009653515, -0.007986466419713, -0.029497652767486, -0.055022297341530);
      set_row(tmp, 1, 0,                  0.415296253825059, -0.016194789754698, -0.038713453134371);
      set_row(tmp, 2, 0,                  0,                  0.387518183415998, -0.034114893784155);
      set_row(tmp, 3, 0,                  0,                  0,                  0.433761784290076);
      tmp.assign(Z_ref);
    }



    verify_that_matrices_are_almost_equal(Z, Z_ref, 1e-10);
  }

  test_symm_multiply<MatrixType>(param);

  test_symm_square<MatrixType>(param);

  test_symm_rk<MatrixType>(param);

  test_invchol<MatrixType>();

  if (verbose)
    std::cout << "Test get_trace" << std::endl;
  // Test get_trace()
  {
    MatrixType A;
    A.set_params(param);
    A.resize(3,3);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 2, 3, 5);
      set_row(tmp, 1, 0, 1, 2);
      set_row(tmp, 2, 5, 8, 9);
      tmp.assign(A);
    }
    if(A.get_trace() != 12.0)
      throw std::runtime_error("Error: get_trace() gave wrong value.");
  }


  if (verbose)
    std::cout << "Test set_to_identity" << std::endl;
  // Test set_to_identity()
  {
    MatrixType A;
    A.set_params(param);
    MatrixType::set_to_identity(A, 17);

    if(A.get_trace() != 17.0)
      throw std::runtime_error("Error: set_to_identity() gave wrong result.");
  }


  if (verbose)
    std::cout << "Test get_nnz_diag_lowest_level" << std::endl;
  // Test get_nnz_diag_lowest_level()
  {
    MatrixType A;
    A.set_params(param);
    A.resize(3,3);
    {
      SparseMatrix tmp;
      set_row(tmp, 0, 1, 2, 3);
      set_row(tmp, 1, 0, 6, 4);
      set_row(tmp, 2, 0, 0, 5);
      tmp.assign(A);
    }

    if(A.get_nnz_diag_lowest_level() != 8)
      throw std::runtime_error("Error: get_nnz_diag_lowest_level() gave wrong value.");
  }


  MatrixType X;
  param.blocksize = 3;
  X.set_params(param);
  X.resize(6,6);
  X.random_blocks(3);

  if(X.get_nnz() != 27) throw std::runtime_error("Error: random_blocks() gave wrong value.");



	  {//Test SpAMM

		param.blocksize = 2 ;

		MatrixType As;
		As.set_params(param);
		As.resize(4, 4, &n_resizes);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1, 2, 0.1, 0.1);
			set_row(tmp, 1, 2, 1, 0.1, 0.1);
			set_row(tmp, 2, 3, 1, 0, 0);
			set_row(tmp, 3, 5, 1, 0, 0.1);
			tmp.assign(As);
		}
		As.update_internal_info();

		MatrixType Bs;
		Bs.set_params(param);
		Bs.resize(4, 4, &n_resizes);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1, 6, 5, 0);
			set_row(tmp, 1, 0, 1, 2, 1);
			set_row(tmp, 2, 2, 1, 0.1, 0.1);
			set_row(tmp, 3, 0, 1, 0.1, 0.1);
			tmp.assign(Bs);
		}
		Bs.update_internal_info();

		MatrixType AsxBs;

		n_resizes = 0;

		MatrixType::spamm(As, false, Bs, false, AsxBs, 0.2, true, &n_multiplications, &n_resizes);

		std::cout << "SPAMM finished, n_mults =  " << n_multiplications << ", n_resizes = " << n_resizes << std::endl;


		MatrixType AsxBs_ref;
		AsxBs_ref.set_params(param);
		AsxBs_ref.resize(4, 4, &n_resizes);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1.2, 8.2, 9, 2);
			set_row(tmp, 1, 2.2, 13.2, 12, 1);
			set_row(tmp, 2, 3, 19, 17, 1);
			set_row(tmp, 3, 5, 31.1, 27, 1);
			tmp.assign(AsxBs_ref);
		}

		verify_that_matrices_are_equal(AsxBs_ref, AsxBs);

		bool apply_truncation = false;
		bool apply_spamm = true;
	  std::vector<double> taus;
		taus.push_back(0.0125);
		taus.push_back(0.025);
		taus.push_back(0.05);
		taus.push_back(0.1);
		taus.push_back(0.2);
		taus.push_back(0.4);
		taus.push_back(0.8);
		std::vector<unsigned long int> skips = MatrixType::count_skips(As, false, Bs, false, taus, apply_truncation, apply_spamm);
		std::cout << "Skips counter, in total " << skips.size() << " skips." << std::endl;
		for(int k = 0; k < skips.size(); ++k){
			std::cout << "S[" << k << "] = " << skips[k] << " ";
		}
		std::cout << std::endl;

		std::vector<double> errors = MatrixType::get_errors_of_approx_multiplication(As, false, Bs, false, taus, apply_truncation, apply_spamm);
		std::vector<double> spamm_errors = MatrixType::get_spamm_errors(As, false, Bs, false, taus);

		for(int k = 0; k < errors.size(); ++k){
			std::cout << "E[" << k << "] = " << errors[k] << " ";
			std::cout << "ESPAMM[" << k << "] = " << spamm_errors[k] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;

		for(int k = 0; k < taus.size(); ++k){
			MatrixType AsxBs_approx, AsxBs_exact, minus_AsxBs_exact, AsxBs_error;
			MatrixType::spamm(As, false, Bs, false, AsxBs_approx, taus[k], true);
			MatrixType::spamm(As, false, Bs, false, AsxBs_exact, 0.0, true);
			minus_AsxBs_exact.rescale(AsxBs_exact, -1.0);
			MatrixType::add(AsxBs_approx, minus_AsxBs_exact, AsxBs_error);
			double error_norm_squared = AsxBs_error.get_frob_squared();
			std::cout << "Error in  multiplication with tau = " << taus[k] << " is " << std::sqrt(error_norm_squared) << std::endl;
			if(errors[k] > 0) assert(std::sqrt(error_norm_squared) < errors[k]);
		}







		MatrixType AsxBsT;
		MatrixType::spamm(As, false, Bs, true, AsxBsT, 0.2, true, &n_multiplications, &n_resizes);
		std::cout << "SPAMM finished, n_mults =  " << n_multiplications << ", n_resizes = " << n_resizes << std::endl;

		MatrixType AsTxBs;
		MatrixType::spamm(As, true, Bs, false, AsTxBs, 0.2, true, &n_multiplications, &n_resizes);
		std::cout << "SPAMM finished, n_mults =  " << n_multiplications << ", n_resizes = " << n_resizes << std::endl;


		MatrixType AsTxBsT;
		MatrixType::spamm(As, true, Bs, true, AsTxBsT, 0.2, true, &n_multiplications, &n_resizes);
	  std::cout << "SPAMM finished, n_mults =  " << n_multiplications << ", n_resizes = " << n_resizes << std::endl;

  }

  {
	   // test frob_block_trunc

	  	param.blocksize = 2;
		MatrixType As;
		As.set_params(param);
		As.resize(4, 4, &n_resizes);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1, 2, 0.1, 0.1);
			set_row(tmp, 1, 2, 1, 0.1, 0.1);
			set_row(tmp, 2, 3, 1, 0, 0);
			set_row(tmp, 3, 5, 1, 0, 0.1);
			tmp.assign(As);
		}

		MatrixType Bs;
		As.frob_block_trunc(Bs, 0.21);

		MatrixType Bs_ref;
		Bs_ref.set_params(param);
		Bs_ref.resize(4, 4, &n_resizes);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1, 2);
			set_row(tmp, 1, 2, 1);
			set_row(tmp, 2, 3, 1);
			set_row(tmp, 3, 5, 1);
			tmp.assign(Bs_ref);
		}

	  verify_that_matrices_are_equal(Bs_ref, Bs);
  }



  {

		param.blocksize = 2;
		MatrixType As;
		As.set_params(param);
		As.resize(1, 4);

		MatrixType Bs;
		Bs.set_params(param);
		Bs.resize(4, 1);

		MatrixType Cs;
		Cs.set_params(param);
		Cs.resize(1, 1);

		std::vector<int> rows1, cols1, rows2, cols2, rows3, cols3;
		std::vector<double> vals1, vals2, vals3;


		rows1.push_back(0);
		cols1.push_back(0);
		vals1.push_back(1);
		rows1.push_back(0);
		cols1.push_back(1);
		vals1.push_back(2);
		rows1.push_back(0);
		cols1.push_back(2);
		vals1.push_back(3);
		rows1.push_back(0);
		cols1.push_back(3);
		vals1.push_back(4);

		As.assign_from_vectors(rows1,cols1,vals1);

		rows2.push_back(0);
		cols2.push_back(0);
		vals2.push_back(5);
		rows2.push_back(1);
		cols2.push_back(0);
		vals2.push_back(6);
		rows2.push_back(2);
		cols2.push_back(0);
		vals2.push_back(7);
		rows2.push_back(3);
		cols2.push_back(0);
		vals2.push_back(8);

    Bs.assign_from_vectors(rows2,cols2,vals2);

		MatrixType AsxBs;

		MatrixType::multiply(As, false, Bs, false, AsxBs);


		rows3.push_back(0);
		cols3.push_back(0);
		vals3.push_back(70);

		Cs.assign_from_vectors(rows3, cols3,vals3);

		verify_that_matrices_are_equal(AsxBs, Cs);

		printf("AsxBs depth %d, Cs depth %d \n", AsxBs.get_depth(), Cs.get_depth());

		assert(AsxBs.get_depth() == Cs.get_depth());

		printf("As consistent? %d \n", As.check_if_matrix_is_consistent());
		printf("Bs consistent? %d \n", B.check_if_matrix_is_consistent());
		printf("Cs consistent? %d \n", Cs.check_if_matrix_is_consistent());

		MatrixType X;
		X.set_params(param);
		X.resize(4,4);
		assert(!X.check_if_matrix_is_consistent());

		MatrixType Y;
		Y.set_params(param);
		Y.resize(2,2);
		assert(Y.check_if_matrix_is_consistent());

  }


  {

		param.blocksize = 1;
		MatrixType As;
		As.set_params(param);
		As.resize(2, 6);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 1, 2, 3, 4, 5, 6);
			set_row(tmp, 1, 7, 8, 9, 10, 11, 12);
			tmp.assign(As);
		}

		MatrixType Bs;
		Bs.set_params(param);
		Bs.resize(6, 2);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 0, 0);
			set_row(tmp, 1, 0, 0);
			set_row(tmp, 2, 0, 0);
			set_row(tmp, 3, 0, 0);
			set_row(tmp, 4, 1, 1);
			set_row(tmp, 5, 1, 1);
			tmp.assign(Bs);
		}

		MatrixType Cs;
		Cs.set_params(param);
		Cs.resize(2, 2);
		{
			SparseMatrix tmp;
			set_row(tmp, 0, 11, 11);
			set_row(tmp, 1, 23, 23);
			tmp.assign(Cs);
		}


		MatrixType AsxBs;

		MatrixType::multiply(As, false, Bs, false, AsxBs);

    verify_that_matrices_are_equal(AsxBs, Cs);

		assert(AsxBs.get_depth() == Cs.get_depth());


  }

  std::cout << "Matrix library matrix operations test finished OK." << std::endl;

  return 0;
}

int main() {

  return test_operations<hbsm::HierarchicalBlockSparseMatrix<double> >();

}
