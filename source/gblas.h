/* Block-Sparse-Matrix-Lib, version 1.0. A block sparse matrix library.
 * Copyright (C) Emanuel H. Rubensson <emanuelrubensson@gmail.com>,
 *               Elias Rudberg <eliasrudberg@gmail.com>, 
 *               Anastasia Kruchinina <anastasia.kruchinina@it.uu.se>, and
 *               Anton Artemov anton.artemov@it.uu.se.
 * 
 * Distribution without copyright owners' explicit consent prohibited.
 * 
 * This source code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef GBLAS_HEADER
#define GBLAS_HEADER


extern "C" void dgemm_(const char *ta,const char *tb,
                       const int *n, const int *k, const int *l,
                       const double *alpha,const double *A,const int *lda,
                       const double *B, const int *ldb,
                       const double *beta, double *C, const int *ldc);
extern "C" void sgemm_(const char *ta,const char *tb,
                       const int *n, const int *k, const int *l,
                       const float *alpha,const float *A,const int *lda,
                       const float *B, const int *ldb,
                       const float *beta, float *C, const int *ldc);
extern "C" void dsymm_(const char *side,const char *uplo,
                       const int *m,const int *n,
                       const double *alpha,const double *A,const int *lda,
                       const double *B,const int *ldb, const double* beta,
                       double *C,const int *ldc);
extern "C" void ssymm_(const char *side,const char *uplo,
                       const int *m,const int *n,
                       const float *alpha,const float *A,const int *lda,
                       const float *B,const int *ldb, const float* beta,
                       float *C,const int *ldc);
extern "C" void dsyrk_(const char *uplo, const char *trans, const int *n,
                       const int *k, const double *alpha, const double *A,
                       const int *lda, const double *beta,
                       double *C, const int *ldc);
extern "C" void ssyrk_(const char *uplo, const char *trans, const int *n,
                       const int *k, const float *alpha, const float *A,
                       const int *lda, const float *beta,
                       float *C, const int *ldc);

extern "C" void dtrmm_(const char *side,const char *uplo,const char *transa,
                       const char *diag,const int *m,const int *n,
                       const double *alpha,const double *A,const int *lda,
                       double *B,const int *ldb);
extern "C" void strmm_(const char *side,const char *uplo,const char *transa,
                       const char *diag,const int *m,const int *n,
                       const float *alpha,const float *A,const int *lda,
                       float *B,const int *ldb);
extern "C" void daxpy_(const int* n, const double* da, const double* dx,
		       const int* incx, double* dy,const int* incy);
extern "C" void saxpy_(const int* n, const float* da, const float* dx,
		       const int* incx, float* dy,const int* incy);
extern "C" void dsyev_(const char *jobz, const char *uplo, const int *n,
		       double *a, const int *lda, double *w, double *work,
		       const int *lwork, int *info);
extern "C" void dgemv_(const char *ta, const int *m, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy);
extern "C" void sgemv_(const char *ta, const int *m, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy);

extern "C" void dsymv_(const char *uplo, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy);
extern "C" void ssymv_(const char *uplo, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy);

extern "C" double ddot_(const int *n, const double *dx, const int *incx,
                      const double *dy, const int *incy);

extern "C" float sdot_(const int *n, const float *dx, const int *incx,
                      const float *dy, const int *incy);

inline void gemm(const char *ta,const char *tb,
		 const int *n, const int *k, const int *l,
		 const double *alpha,const double *A,const int *lda,
		 const double *B, const int *ldb,
		 const double *beta, double *C, const int *ldc) {
  dgemm_(ta,tb,n,k,l,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void gemm(const char *ta,const char *tb,
		 const int *n, const int *k, const int *l,
		 const float *alpha,const float *A,const int *lda,
		 const float *B, const int *ldb,
		 const float *beta, float *C, const int *ldc) {
  sgemm_(ta,tb,n,k,l,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void symm(const char *side,const char *uplo,
		 const int *m,const int *n,
		 const double *alpha,const double *A,const int *lda,
		 const double *B,const int *ldb, const double* beta,
		 double *C,const int *ldc) {
  dsymm_(side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void symm(const char *side,const char *uplo,
		 const int *m,const int *n,
		 const float *alpha,const float *A,const int *lda,
		 const float *B,const int *ldb, const float* beta,
		 float *C,const int *ldc) {
  ssymm_(side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc);
}
inline void syrk(const char *uplo, const char *trans, const int *n,
		 const int *k, const double *alpha, const double *A,
		 const int *lda, const double *beta,
		 double *C, const int *ldc) {
  dsyrk_(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void syrk(const char *uplo, const char *trans, const int *n,
		 const int *k, const float *alpha, const float *A,
		 const int *lda, const float *beta,
		 float *C, const int *ldc) {
  ssyrk_(uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
inline void trmm(const char *side,const char *uplo,const char *transa,
		 const char *diag,const int *m,const int *n,
		 const double *alpha,const double *A,const int *lda,
		 double *B,const int *ldb) {
  dtrmm_(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb);
}
inline void trmm(const char *side,const char *uplo,const char *transa,
		 const char *diag,const int *m,const int *n,
		 const float *alpha,const float *A,const int *lda,
		 float *B,const int *ldb) {
  strmm_(side,uplo,transa,diag,m,n,alpha,A,lda,B,ldb);
}
inline void axpy(const int* n, const double* da, const double* dx,
		 const int* incx, double* dy,const int* incy) {
  daxpy_(n, da, dx, incx, dy, incy);
}
inline void axpy(const int* n, const float* da, const float* dx,
		 const int* incx, float* dy,const int* incy) {
  saxpy_(n, da, dx, incx, dy, incy);
}


inline void syev(const char *jobz, const char *uplo, const int *n,
		       double *a, const int *lda, double *w, double *work,
		       const int *lwork, int *info)
{
  dsyev_(jobz, uplo, n, a, lda, w, work, lwork, info);
}


inline void gemv(const char *ta, const int *m, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy){
    dgemv_(ta, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void gemv(const char *ta, const int *m, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy){
    sgemv_(ta, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void symv(const char *uplo, const int *n,
		       const double *alpha, const double *A, const int *lda,
		       const double *x, const int *incx, const double *beta,
		       double *y, const int *incy){
   dsymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline void symv(const char *uplo, const int *n,
		       const float *alpha, const float *A, const int *lda,
		       const float *x, const int *incx, const float *beta,
		       float *y, const int *incy){
    ssymv_(uplo, n, alpha, A, lda, x, incx, beta, y, incy);
}

inline double ddot(const int *n, const double *dx, const int *incx,
                      const double *dy, const int *incy){
    return ddot_(n,dx,incx,dy,incy);
}

inline float ddot(const int *n, const float *dx, const int *incx,
                      const float *dy, const int *incy){
    return sdot_(n,dx,incx,dy,incy);
}



#endif
