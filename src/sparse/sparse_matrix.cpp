#include "numcpp/sparse/sparse_matrix.h"
#include "numcpp/parallel/parallel.h"
#include <algorithm>
#include <stdexcept>
#include <iomanip>

#ifdef NUMPCPP_HAVE_THREADS
#include <thread>
#endif

namespace numcpp {

SparseMatrix::SparseMatrix()
    : rows_(0), cols_(0), row_ptr_(1, 0) {}

SparseMatrix::SparseMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

SparseMatrix::SparseMatrix(size_t rows, size_t cols,
                           const std::vector<size_t>& row_ptr,
                           const std::vector<size_t>& col_idx,
                           const std::vector<real_t>& values)
    : rows_(rows), cols_(cols), row_ptr_(row_ptr), col_idx_(col_idx), values_(values) {}

SparseMatrix::SparseMatrix(const Matrix& dense) {
    fromDense(dense);
}

size_t SparseMatrix::rows() const { return rows_; }
size_t SparseMatrix::cols() const { return cols_; }
size_t SparseMatrix::nnz() const { return values_.size(); }
bool SparseMatrix::empty() const { return rows_ == 0 || cols_ == 0; }

void SparseMatrix::clear() {
    rows_ = 0;
    cols_ = 0;
    row_ptr_.assign(1, 0);
    col_idx_.clear();
    values_.clear();
}

const std::vector<size_t>& SparseMatrix::rowPtr() const { return row_ptr_; }
const std::vector<size_t>& SparseMatrix::colIdx() const { return col_idx_; }
const std::vector<real_t>& SparseMatrix::values() const { return values_; }

void SparseMatrix::reserve(size_t n) {
    col_idx_.reserve(n);
    values_.reserve(n);
}

size_t SparseMatrix::findIndex(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("SparseMatrix index out of range");
    }
    
    size_t start = row_ptr_[row];
    size_t end = row_ptr_[row + 1];
    
    for (size_t i = start; i < end; ++i) {
        if (col_idx_[i] == col) {
            return i;
        }
    }
    return end;
}

real_t SparseMatrix::operator()(size_t i, size_t j) const {
    size_t idx = findIndex(i, j);
    if (idx < row_ptr_[i + 1] && col_idx_[idx] == j) {
        return values_[idx];
    }
    return 0.0;
}

void SparseMatrix::insert(size_t i, size_t j, real_t value) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("SparseMatrix insert index out of range");
    }
    
    size_t pos = findIndex(i, j);
    if (pos < row_ptr_[i + 1] && col_idx_[pos] == j) {
        values_[pos] = value;
    } else {
        col_idx_.insert(col_idx_.begin() + pos, j);
        values_.insert(values_.begin() + pos, value);
        for (size_t k = i + 1; k <= rows_; ++k) {
            row_ptr_[k]++;
        }
    }
}

void SparseMatrix::set(size_t i, size_t j, real_t value) {
    if (std::abs(value) < EPS) {
        size_t pos = findIndex(i, j);
        if (pos < row_ptr_[i + 1] && col_idx_[pos] == j) {
            col_idx_.erase(col_idx_.begin() + pos);
            values_.erase(values_.begin() + pos);
            for (size_t k = i + 1; k <= rows_; ++k) {
                row_ptr_[k]--;
            }
        }
    } else {
        insert(i, j, value);
    }
}

SparseMatrix SparseMatrix::operator+(const SparseMatrix& other) const {
    return spAdd(*this, other);
}

SparseMatrix SparseMatrix::operator-(const SparseMatrix& other) const {
    return spAdd(*this, spScalarMult(other, -1.0));
}

SparseMatrix SparseMatrix::operator*(real_t scalar) const {
    return spScalarMult(*this, scalar);
}

Vector SparseMatrix::operator*(const Vector& v) const {
    return spmv(*this, v);
}

SparseMatrix& SparseMatrix::operator+=(const SparseMatrix& other) {
    *this = *this + other;
    return *this;
}

SparseMatrix& SparseMatrix::operator-=(const SparseMatrix& other) {
    *this = *this - other;
    return *this;
}

SparseMatrix& SparseMatrix::operator*=(real_t scalar) {
    for (auto& v : values_) {
        v *= scalar;
    }
    return *this;
}

SparseMatrix SparseMatrix::transpose() const {
    SparseMatrix result(cols_, rows_);
    result.row_ptr_.assign(cols_ + 1, 0);
    
    for (size_t i = 0; i < nnz(); ++i) {
        result.row_ptr_[col_idx_[i] + 1]++;
    }
    
    for (size_t i = 1; i <= cols_; ++i) {
        result.row_ptr_[i] += result.row_ptr_[i - 1];
    }
    
    result.col_idx_.resize(nnz());
    result.values_.resize(nnz());
    
    std::vector<size_t> next = result.row_ptr_;
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
            size_t col = col_idx_[j];
            size_t pos = next[col]++;
            result.col_idx_[pos] = i;
            result.values_[pos] = values_[j];
        }
    }
    
    return result;
}

real_t SparseMatrix::trace() const {
    if (rows_ != cols_) {
        throw std::invalid_argument("Matrix must be square for trace");
    }
    
    real_t sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        sum += (*this)(i, i);
    }
    return sum;
}

Matrix SparseMatrix::toDense() const {
    Matrix result(rows_, cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = row_ptr_[i]; j < row_ptr_[i + 1]; ++j) {
            result(i, col_idx_[j]) = values_[j];
        }
    }
    return result;
}

void SparseMatrix::fromDense(const Matrix& dense) {
    rows_ = dense.rows();
    cols_ = dense.cols();
    row_ptr_.assign(rows_ + 1, 0);
    col_idx_.clear();
    values_.clear();
    
    size_t nnz_count = 0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(dense(i, j)) > EPS) {
                nnz_count++;
            }
        }
        row_ptr_[i + 1] = nnz_count;
    }
    
    col_idx_.reserve(nnz_count);
    values_.reserve(nnz_count);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(dense(i, j)) > EPS) {
                col_idx_.push_back(j);
                values_.push_back(dense(i, j));
            }
        }
    }
}

SparseMatrix SparseMatrix::identity(size_t n) {
    SparseMatrix result(n, n);
    result.row_ptr_.resize(n + 1);
    result.col_idx_.resize(n);
    result.values_.resize(n, 1.0);
    
    for (size_t i = 0; i < n; ++i) {
        result.row_ptr_[i] = i;
        result.col_idx_[i] = i;
    }
    result.row_ptr_[n] = n;
    
    return result;
}

SparseMatrix SparseMatrix::diagonal(const Vector& diag) {
    size_t n = diag.size();
    SparseMatrix result(n, n);
    result.row_ptr_.resize(n + 1);
    result.col_idx_.resize(n);
    result.values_.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        result.row_ptr_[i] = i;
        result.col_idx_[i] = i;
        result.values_[i] = diag[i];
    }
    result.row_ptr_[n] = n;
    
    return result;
}

SparseMatrix operator*(real_t scalar, const SparseMatrix& m) {
    return m * scalar;
}

Vector spmv(const SparseMatrix& A, const Vector& x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("Matrix and vector dimensions incompatible for multiplication");
    }
    
    Vector result(A.rows(), 0.0);
    const auto& row_ptr = A.rowPtr();
    const auto& col_idx = A.colIdx();
    const auto& values = A.values();
    
    for (size_t i = 0; i < A.rows(); ++i) {
        real_t sum = 0.0;
        for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        result[i] = sum;
    }
    
    return result;
}

Vector spmvParallel(const SparseMatrix& A, const Vector& x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("Matrix and vector dimensions incompatible for multiplication");
    }
    
    size_t nrows = A.rows();
    Vector result(nrows, 0.0);
    
#ifdef NUMPCPP_HAVE_THREADS
    unsigned int nthreads = getNumThreads();
    if (nthreads <= 1 || nrows < nthreads * 2) {
        return spmv(A, x);
    }
    
    const auto& row_ptr = A.rowPtr();
    const auto& col_idx = A.colIdx();
    const auto& values = A.values();
    
    size_t chunk = nrows / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    
    for (unsigned int t = 0; t < nthreads; ++t) {
        size_t start = t * chunk;
        size_t stop = (t == nthreads - 1) ? nrows : (t + 1) * chunk;
        
        threads.emplace_back([start, stop, &row_ptr, &col_idx, &values, &x, &result]() {
            for (size_t i = start; i < stop; ++i) {
                real_t sum = 0.0;
                for (size_t j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
                    sum += values[j] * x[col_idx[j]];
                }
                result[i] = sum;
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
#else
    result = spmv(A, x);
#endif
    
    return result;
}

SparseMatrix spAdd(const SparseMatrix& A, const SparseMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions incompatible for addition");
    }
    
    SparseMatrix result(A.rows(), A.cols());
    result.row_ptr_.resize(A.rows() + 1, 0);
    
    const auto& A_row_ptr = A.rowPtr();
    const auto& A_col_idx = A.colIdx();
    const auto& A_values = A.values();
    
    const auto& B_row_ptr = B.rowPtr();
    const auto& B_col_idx = B.colIdx();
    const auto& B_values = B.values();
    
    std::vector<size_t> temp_col;
    std::vector<real_t> temp_val;
    
    for (size_t i = 0; i < A.rows(); ++i) {
        temp_col.clear();
        temp_val.clear();
        
        size_t ja = A_row_ptr[i];
        size_t jb = B_row_ptr[i];
        size_t ea = A_row_ptr[i + 1];
        size_t eb = B_row_ptr[i + 1];
        
        while (ja < ea && jb < eb) {
            if (A_col_idx[ja] == B_col_idx[jb]) {
                real_t sum = A_values[ja] + B_values[jb];
                if (std::abs(sum) > EPS) {
                    temp_col.push_back(A_col_idx[ja]);
                    temp_val.push_back(sum);
                }
                ja++;
                jb++;
            } else if (A_col_idx[ja] < B_col_idx[jb]) {
                temp_col.push_back(A_col_idx[ja]);
                temp_val.push_back(A_values[ja]);
                ja++;
            } else {
                temp_col.push_back(B_col_idx[jb]);
                temp_val.push_back(B_values[jb]);
                jb++;
            }
        }
        
        while (ja < ea) {
            temp_col.push_back(A_col_idx[ja]);
            temp_val.push_back(A_values[ja]);
            ja++;
        }
        
        while (jb < eb) {
            temp_col.push_back(B_col_idx[jb]);
            temp_val.push_back(B_values[jb]);
            jb++;
        }
        
        result.row_ptr_[i + 1] = result.row_ptr_[i] + temp_col.size();
        result.col_idx_.insert(result.col_idx_.end(), temp_col.begin(), temp_col.end());
        result.values_.insert(result.values_.end(), temp_val.begin(), temp_val.end());
    }
    
    return result;
}

SparseMatrix spScalarMult(const SparseMatrix& A, real_t scalar) {
    SparseMatrix result = A;
    if (std::abs(scalar) < EPS) {
        result.values_.clear();
        result.col_idx_.clear();
        result.row_ptr_.assign(A.rows() + 1, 0);
    } else {
        for (auto& v : result.values_) {
            v *= scalar;
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os, const SparseMatrix& m) {
    os << "SparseMatrix " << m.rows() << "x" << m.cols() 
       << " (" << m.nnz() << " non-zeros):" << std::endl;
    
    size_t print_rows = std::min(m.rows(), static_cast<size_t>(10));
    
    for (size_t i = 0; i < print_rows; ++i) {
        os << "  row " << i << ": ";
        size_t count = 0;
        for (size_t j = m.rowPtr()[i]; j < m.rowPtr()[i + 1] && count < 5; ++j) {
            os << "(" << m.colIdx()[j] << ", " << std::setprecision(4) << m.values()[j] << ") ";
            count++;
        }
        if (m.rowPtr()[i + 1] - m.rowPtr()[i] > 5) {
            os << "...";
        }
        os << std::endl;
    }
    
    if (m.rows() > 10) {
        os << "  ..." << std::endl;
    }
    
    return os;
}

}
