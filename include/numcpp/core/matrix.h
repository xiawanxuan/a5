#ifndef NUMPCPP_CORE_MATRIX_H
#define NUMPCPP_CORE_MATRIX_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include <vector>
#include <initializer_list>
#include <ostream>

namespace numcpp {

class Matrix {
public:
    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, real_t value);
    Matrix(const std::vector<std::vector<real_t>>& data);
    Matrix(std::initializer_list<std::initializer_list<real_t>> values);

    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    bool empty() const;
    void resize(size_t rows, size_t cols);
    void fill(real_t value);
    void clear();

    real_t& operator()(size_t i, size_t j);
    const real_t& operator()(size_t i, size_t j) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(real_t scalar) const;
    Matrix operator/(real_t scalar) const;
    Vector operator*(const Vector& v) const;
    Matrix operator*(const Matrix& other) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(real_t scalar);
    Matrix& operator/=(real_t scalar);

    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;

    Matrix transpose() const;
    real_t trace() const;
    real_t determinant() const;
    Matrix inverse() const;

    Vector row(size_t i) const;
    Vector col(size_t j) const;
    void setRow(size_t i, const Vector& v);
    void setCol(size_t j, const Vector& v);

    Matrix submatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const;

    bool isSquare() const;
    bool isSymmetric(real_t tol = EPS) const;
    bool isDiagonal(real_t tol = EPS) const;

    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t n);
    static Matrix diagonal(const Vector& diag);
    static Matrix random(size_t rows, size_t cols, real_t min = 0.0, real_t max = 1.0);

    const std::vector<real_t>& data() const;
    std::vector<real_t>& data();

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);

private:
    size_t rows_;
    size_t cols_;
    std::vector<real_t> data_;

    size_t index(size_t i, size_t j) const;
};

Matrix operator*(real_t scalar, const Matrix& m);

}

#endif
