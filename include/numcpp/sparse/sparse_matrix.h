#ifndef NUMPCPP_SPARSE_SPARSE_MATRIX_H
#define NUMPCPP_SPARSE_SPARSE_MATRIX_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"
#include <vector>
#include <tuple>
#include <ostream>

namespace numcpp {

class SparseMatrix {
public:
    SparseMatrix();
    SparseMatrix(size_t rows, size_t cols);
    SparseMatrix(size_t rows, size_t cols,
                 const std::vector<size_t>& row_ptr,
                 const std::vector<size_t>& col_idx,
                 const std::vector<real_t>& values);
    SparseMatrix(const Matrix& dense);

    size_t rows() const;
    size_t cols() const;
    size_t nnz() const;
    bool empty() const;
    void clear();

    const std::vector<size_t>& rowPtr() const;
    const std::vector<size_t>& colIdx() const;
    const std::vector<real_t>& values() const;

    real_t operator()(size_t i, size_t j) const;
    void insert(size_t i, size_t j, real_t value);
    void set(size_t i, size_t j, real_t value);

    SparseMatrix operator+(const SparseMatrix& other) const;
    SparseMatrix operator-(const SparseMatrix& other) const;
    SparseMatrix operator*(real_t scalar) const;
    Vector operator*(const Vector& v) const;

    SparseMatrix& operator+=(const SparseMatrix& other);
    SparseMatrix& operator-=(const SparseMatrix& other);
    SparseMatrix& operator*=(real_t scalar);

    SparseMatrix transpose() const;
    real_t trace() const;

    Matrix toDense() const;
    void fromDense(const Matrix& dense);

    static SparseMatrix identity(size_t n);
    static SparseMatrix diagonal(const Vector& diag);

    friend std::ostream& operator<<(std::ostream& os, const SparseMatrix& m);

private:
    size_t rows_;
    size_t cols_;
    std::vector<size_t> row_ptr_;
    std::vector<size_t> col_idx_;
    std::vector<real_t> values_;

    void compress();
    void reserve(size_t n);
    size_t findIndex(size_t row, size_t col) const;
};

SparseMatrix operator*(real_t scalar, const SparseMatrix& m);

Vector spmv(const SparseMatrix& A, const Vector& x);
Vector spmvParallel(const SparseMatrix& A, const Vector& x);

SparseMatrix spAdd(const SparseMatrix& A, const SparseMatrix& B);
SparseMatrix spScalarMult(const SparseMatrix& A, real_t scalar);

}

#endif
