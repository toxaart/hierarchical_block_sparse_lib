# hierarchical_block_sparse_lib
Hierarchical block-sparse matrix library

This matrix library is developed to effeiciently perform approximate multiplication of sparse matrices with decay in shared memory. 

It has been used in conjunction with the Chunk and Tasks matrix library.

The library is compeltely serial, but threading can be added using C++17 features if so desired. 

References:
1. [Fast Multiplication of Matrices with Deca](https://arxiv.org/abs/1011.3534)
2. [Sparse approximate matrix multiplication in a fully recursive distributed task-based parallel framework](https://arxiv.org/abs/1906.08148)
3. [Sparse approximate matrix-matrix multiplication with error control](https://arxiv.org/abs/2005.10680)
4. [Locality-aware parallel block-sparse matrix-matrix multiplication using the Chunks and Tasks programming model](https://arxiv.org/abs/1501.07800)
