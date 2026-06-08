#include "numcpp/core/matrix.h"
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

namespace numcpp {

Matrix::Matrix() : rows_(0), cols_(0), data_() {}

Matrix::Matrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, real_t value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(const std::vector<std::vector<real_t>>& data) {
    rows_ = data.size();
    cols_ = rows_ > 0 ? data[0].size() : 0;
    data_.resize(rows_ * cols_);
    for (size_t i = 0; i < rows_; ++i) {
        if (data[i].size() != cols_)
            throw std::invalid_argument("All rows must have the same size");
        for (size_t j = 0; j < cols_; ++j)
            data_[index(i, j)] = data[i][j];
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<real_t>> values) {
    rows_ = values.size();
    cols_ = rows_ > 0 ? values.begin()->size() : 0;
    data_.resize(rows_ * cols_);
    size_t i = 0;
    for (const auto& row : values) {
        if (row.size() != cols_)
            throw std::invalid_argument("All rows must have the same size");
        size_t j = 0;
        for (auto val : row) {
            data_[index(i, j)] = val;
            ++j;
        }
        ++i;
    }
}

size_t Matrix::rows() const { return rows_; }

size_t Matrix::cols() const { return cols_; }

size_t Matrix::size() const { return rows_ * cols_; }

bool Matrix::empty() const { return rows_ == 0 || cols_ == 0; }

void Matrix::resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;
    data_.resize(rows * cols, 0.0);
}

void Matrix::fill(real_t value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::clear() {
    rows_ = 0;
    cols_ = 0;
    data_.clear();
}

size_t Matrix::index(size_t i, size_t j) const { return i * cols_ + j; }

real_t& Matrix::operator()(size_t i, size_t j) {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[index(i, j)];
}

const real_t& Matrix::operator()(size_t i, size_t j) const {
    if (i >= rows_ || j >= cols_)
        throw std::out_of_range("Matrix index out of range");
    return data_[index(i, j)];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match for addition");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] + other.data_[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] - other.data_[i];
    return result;
}

Matrix Matrix::operator*(real_t scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] * scalar;
    return result;
}

Matrix Matrix::operator/(real_t scalar) const {
    if (std::abs(scalar) < EPS)
        throw std::invalid_argument("Division by zero");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < size(); ++i)
        result.data_[i] = data_[i] / scalar;
    return result;
}

Vector Matrix::operator*(const Vector& v) const {
    if (cols_ != v.size())
        throw std::invalid_argument("Matrix columns must match vector size");
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        real_t sum = 0.0;
        for (size_t k = 0; k < cols_; ++k)
            sum += data_[index(i, k)] * v[k];
        result[i] = sum;
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            real_t sum = 0.0;
            for (size_t k = 0; k < cols_; ++k)
                sum += data_[index(i, k)] * other.data_[other.index(k, j)];
            result.data_[result.index(i, j)] = sum;
        }
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match for addition");
    for (size_t i = 0; i < size(); ++i)
        data_[i] += other.data_[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_)
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    for (size_t i = 0; i < size(); ++i)
        data_[i] -= other.data_[i];
    return *this;
}

Matrix& Matrix::operator*=(real_t scalar) {
    for (auto& x : data_) x *= scalar;
    return *this;
}

Matrix& Matrix::operator/=(real_t scalar) {
    if (std::abs(scalar) < EPS)
        throw std::invalid_argument("Division by zero");
    for (auto& x : data_) x /= scalar;
    return *this;
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) return false;
    for (size_t i = 0; i < size(); ++i)
        if (std::abs(data_[i] - other.data_[i]) > EPS) return false;
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.data_[result.index(j, i)] = data_[index(i, j)];
    return result;
}

real_t Matrix::trace() const {
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square for trace");
    real_t sum = 0.0;
    for (size_t i = 0; i < rows_; ++i)
        sum += data_[index(i, i)];
    return sum;
}

real_t Matrix::determinant() const {
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square for determinant");
    if (rows_ == 1) return data_[0];
    if (rows_ == 2)
        return data_[0] * data_[3] - data_[1] * data_[2];
    
    Matrix temp = *this;
    real_t det = 1.0;
    for (size_t k = 0; k < rows_ - 1; ++k) {
        size_t pivot = k;
        real_t maxVal = std::abs(temp.data_[temp.index(k, k)]);
        for (size_t i = k + 1; i < rows_; ++i) {
            if (std::abs(temp.data_[temp.index(i, k)]) > maxVal) {
                maxVal = std::abs(temp.data_[temp.index(i, k)]);
                pivot = i;
            }
        }
        if (maxVal < EPS) return 0.0;
        if (pivot != k) {
            for (size_t j = 0; j < cols_; ++j)
                std::swap(temp.data_[temp.index(k, j)], temp.data_[temp.index(pivot, j)]);
            det *= -1.0;
        }
        for (size_t i = k + 1; i < rows_; ++i) {
            real_t factor = temp.data_[temp.index(i, k)] / temp.data_[temp.index(k, k)];
            for (size_t j = k; j < cols_; ++j)
                temp.data_[temp.index(i, j)] -= factor * temp.data_[temp.index(k, j)];
        }
    }
    for (size_t i = 0; i < rows_; ++i)
        det *= temp.data_[temp.index(i, i)];
    return det;
}

Matrix Matrix::inverse() const {
    if (!isSquare())
        throw std::invalid_argument("Matrix must be square for inverse");
    size_t n = rows_;
    Matrix aug(n, 2 * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            aug.data_[aug.index(i, j)] = data_[index(i, j)];
        aug.data_[aug.index(i, n + i)] = 1.0;
    }
    
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        real_t maxVal = std::abs(aug.data_[aug.index(k, k)]);
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(aug.data_[aug.index(i, k)]) > maxVal) {
                maxVal = std::abs(aug.data_[aug.index(i, k)]);
                pivot = i;
            }
        }
        if (maxVal < EPS)
            throw std::runtime_error("Matrix is singular");
        if (pivot != k) {
            for (size_t j = 0; j < 2 * n; ++j)
                std::swap(aug.data_[aug.index(k, j)], aug.data_[aug.index(pivot, j)]);
        }
        real_t pivotVal = aug.data_[aug.index(k, k)];
        for (size_t j = k; j < 2 * n; ++j)
            aug.data_[aug.index(k, j)] /= pivotVal;
        for (size_t i = 0; i < n; ++i) {
            if (i != k) {
                real_t factor = aug.data_[aug.index(i, k)];
                for (size_t j = k; j < 2 * n; ++j)
                    aug.data_[aug.index(i, j)] -= factor * aug.data_[aug.index(k, j)];
            }
        }
    }
    
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            result.data_[result.index(i, j)] = aug.data_[aug.index(i, n + j)];
    return result;
}

Vector Matrix::row(size_t i) const {
    if (i >= rows_)
        throw std::out_of_range("Row index out of range");
    Vector result(cols_);
    for (size_t j = 0; j < cols_; ++j)
        result[j] = data_[index(i, j)];
    return result;
}

Vector Matrix::col(size_t j) const {
    if (j >= cols_)
        throw std::out_of_range("Column index out of range");
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i)
        result[i] = data_[index(i, j)];
    return result;
}

void Matrix::setRow(size_t i, const Vector& v) {
    if (i >= rows_)
        throw std::out_of_range("Row index out of range");
    if (v.size() != cols_)
        throw std::invalid_argument("Vector size must match number of columns");
    for (size_t j = 0; j < cols_; ++j)
        data_[index(i, j)] = v[j];
}

void Matrix::setCol(size_t j, const Vector& v) {
    if (j >= cols_)
        throw std::out_of_range("Column index out of range");
    if (v.size() != rows_)
        throw std::invalid_argument("Vector size must match number of rows");
    for (size_t i = 0; i < rows_; ++i)
        data_[index(i, j)] = v[i];
}

Matrix Matrix::submatrix(size_t startRow, size_t startCol, size_t numRows, size_t numCols) const {
    if (startRow + numRows > rows_ || startCol + numCols > cols_)
        throw std::out_of_range("Submatrix bounds out of range");
    Matrix result(numRows, numCols);
    for (size_t i = 0; i < numRows; ++i)
        for (size_t j = 0; j < numCols; ++j)
            result.data_[result.index(i, j)] = data_[index(startRow + i, startCol + j)];
    return result;
}

bool Matrix::isSquare() const { return rows_ == cols_; }

bool Matrix::isSymmetric(real_t tol) const {
    if (!isSquare()) return false;
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = i + 1; j < cols_; ++j)
            if (std::abs(data_[index(i, j)] - data_[index(j, i)]) > tol)
                return false;
    return true;
}

bool Matrix::isDiagonal(real_t tol) const {
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            if (i != j && std::abs(data_[index(i, j)]) > tol)
                return false;
    return true;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t n) {
    Matrix result(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        result.data_[result.index(i, i)] = 1.0;
    return result;
}

Matrix Matrix::diagonal(const Vector& diag) {
    size_t n = diag.size();
    Matrix result(n, n, 0.0);
    for (size_t i = 0; i < n; ++i)
        result.data_[result.index(i, i)] = diag[i];
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, real_t min, real_t max) {
    Matrix result(rows, cols);
    static std::mt19937 gen(42);
    std::uniform_real_distribution<real_t> dist(min, max);
    for (size_t i = 0; i < result.size(); ++i)
        result.data_[i] = dist(gen);
    return result;
}

const std::vector<real_t>& Matrix::data() const { return data_; }

std::vector<real_t>& Matrix::data() { return data_; }

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    os << "[";
    for (size_t i = 0; i < m.rows_; ++i) {
        if (i > 0) os << " ";
        os << "[";
        for (size_t j = 0; j < m.cols_; ++j) {
            os << std::setw(10) << std::setprecision(6) << m.data_[m.index(i, j)];
            if (j < m.cols_ - 1) os << ", ";
        }
        os << "]";
        if (i < m.rows_ - 1) os << ",\n";
    }
    os << "]";
    return os;
}

Matrix operator*(real_t scalar, const Matrix& m) { return m * scalar; }

}
