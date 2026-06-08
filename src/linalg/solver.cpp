#include "numcpp/linalg/solver.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#ifdef NUMPCPP_HAVE_OPENMP
#include <omp.h>
#endif

namespace numcpp {

LUFactorization luDecompose(const Matrix& A) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for LU decomposition");
    
    size_t n = A.rows();
    LUFactorization result;
    result.L = Matrix::zeros(n, n);
    result.U = Matrix::zeros(n, n);
    result.pivots.resize(n);
    for (size_t i = 0; i < n; ++i) result.pivots[i] = i;
    
    Matrix temp = A;
    
    for (size_t k = 0; k < n - 1; ++k) {
        size_t pivot = k;
        real_t maxVal = std::abs(temp(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(temp(i, k)) > maxVal) {
                maxVal = std::abs(temp(i, k));
                pivot = i;
            }
        }
        if (maxVal < EPS)
            throw std::runtime_error("Matrix is singular or nearly singular");
        
        if (pivot != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(temp(k, j), temp(pivot, j));
            }
            std::swap(result.pivots[k], result.pivots[pivot]);
        }
        
        real_t pivotVal = temp(k, k);
        for (size_t i = k + 1; i < n; ++i) {
            real_t factor = temp(i, k) / pivotVal;
            temp(i, k) = factor;
            for (size_t j = k + 1; j < n; ++j) {
                temp(i, j) -= factor * temp(k, j);
            }
        }
    }
    
    for (size_t i = 0; i < n; ++i) {
        result.L(i, i) = 1.0;
        for (size_t j = 0; j < i; ++j) {
            result.L(i, j) = temp(i, j);
        }
        for (size_t j = i; j < n; ++j) {
            result.U(i, j) = temp(i, j);
        }
    }
    
    return result;
}

Vector solveLU(const LUFactorization& lu, const Vector& b) {
    size_t n = lu.L.rows();
    if (b.size() != n)
        throw std::invalid_argument("Vector size must match matrix dimensions");
    
    Vector pb(n);
    for (size_t i = 0; i < n; ++i) {
        pb[i] = b[lu.pivots[i]];
    }
    
    Vector y(n);
    for (size_t i = 0; i < n; ++i) {
        real_t sum = pb[i];
        for (size_t j = 0; j < i; ++j) {
            sum -= lu.L(i, j) * y[j];
        }
        y[i] = sum / lu.L(i, i);
    }
    
    Vector x(n);
    for (size_t i = n; i-- > 0; ) {
        real_t sum = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= lu.U(i, j) * x[j];
        }
        x[i] = sum / lu.U(i, i);
    }
    
    return x;
}

Vector solveLU(const Matrix& A, const Vector& b) {
    LUFactorization lu = luDecompose(A);
    return solveLU(lu, b);
}

Vector solveGaussElimination(const Matrix& A, const Vector& b) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for Gaussian elimination");
    if (A.rows() != b.size())
        throw std::invalid_argument("Vector size must match matrix dimensions");
    
    size_t n = A.rows();
    Matrix aug(n, n + 1);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug(i, j) = A(i, j);
        }
        aug(i, n) = b[i];
    }
    
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        real_t maxVal = std::abs(aug(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(aug(i, k)) > maxVal) {
                maxVal = std::abs(aug(i, k));
                pivot = i;
            }
        }
        if (maxVal < EPS)
            throw std::runtime_error("Matrix is singular or nearly singular");
        
        if (pivot != k) {
            for (size_t j = 0; j <= n; ++j) {
                std::swap(aug(k, j), aug(pivot, j));
            }
        }
        
        real_t pivotVal = aug(k, k);
        for (size_t i = k + 1; i < n; ++i) {
            real_t factor = aug(i, k) / pivotVal;
            for (size_t j = k; j <= n; ++j) {
                aug(i, j) -= factor * aug(k, j);
            }
        }
    }
    
    Vector x(n);
    for (size_t i = n; i-- > 0; ) {
        real_t sum = aug(i, n);
        for (size_t j = i + 1; j < n; ++j) {
            sum -= aug(i, j) * x[j];
        }
        x[i] = sum / aug(i, i);
    }
    
    return x;
}

Vector solveJacobi(const Matrix& A, const Vector& b, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for Jacobi method");
    if (A.rows() != b.size())
        throw std::invalid_argument("Vector size must match matrix dimensions");
    
    size_t n = A.rows();
    Vector x = Vector::zeros(n);
    Vector x_new = Vector::zeros(n);
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            real_t sum = b[i];
            for (size_t j = 0; j < n; ++j) {
                if (j != i) {
                    sum -= A(i, j) * x[j];
                }
            }
            if (std::abs(A(i, i)) < EPS)
                throw std::runtime_error("Zero diagonal element encountered");
            x_new[i] = sum / A(i, i);
        }
        
        real_t error = (x_new - x).norm();
        x = x_new;
        
        if (error < tol) {
            return x;
        }
    }
    
    std::cerr << "Warning: Jacobi method did not converge within " << maxIter << " iterations" << std::endl;
    return x;
}

Vector solveGaussSeidel(const Matrix& A, const Vector& b, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for Gauss-Seidel method");
    if (A.rows() != b.size())
        throw std::invalid_argument("Vector size must match matrix dimensions");
    
    size_t n = A.rows();
    Vector x = Vector::zeros(n);
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        Vector x_old = x;
        
        for (size_t i = 0; i < n; ++i) {
            real_t sum = b[i];
            for (size_t j = 0; j < i; ++j) {
                sum -= A(i, j) * x[j];
            }
            for (size_t j = i + 1; j < n; ++j) {
                sum -= A(i, j) * x[j];
            }
            if (std::abs(A(i, i)) < EPS)
                throw std::runtime_error("Zero diagonal element encountered");
            x[i] = sum / A(i, i);
        }
        
        real_t error = (x - x_old).norm();
        if (error < tol) {
            return x;
        }
    }
    
    std::cerr << "Warning: Gauss-Seidel method did not converge within " << maxIter << " iterations" << std::endl;
    return x;
}

Vector solveConjugateGradient(const Matrix& A, const Vector& b, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for conjugate gradient method");
    if (A.rows() != b.size())
        throw std::invalid_argument("Vector size must match matrix dimensions");
    if (!A.isSymmetric())
        std::cerr << "Warning: Conjugate gradient works best with symmetric positive definite matrices" << std::endl;
    
    size_t n = A.rows();
    Vector x = Vector::zeros(n);
    Vector r = b - A * x;
    Vector p = r;
    real_t r_dot = r.dot(r);
    real_t b_norm = b.norm();
    
    if (b_norm < EPS) return x;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        Vector Ap = A * p;
        real_t p_dot_Ap = p.dot(Ap);
        
        if (std::abs(p_dot_Ap) < EPS) break;
        
        real_t alpha = r_dot / p_dot_Ap;
        x = x + alpha * p;
        r = r - alpha * Ap;
        
        real_t r_dot_new = r.dot(r);
        
        if (std::sqrt(r_dot_new) / b_norm < tol) {
            return x;
        }
        
        real_t beta = r_dot_new / r_dot;
        p = r + beta * p;
        r_dot = r_dot_new;
    }
    
    std::cerr << "Warning: Conjugate gradient did not converge within " << maxIter << " iterations" << std::endl;
    return x;
}

Matrix solveMultiple(const Matrix& A, const Matrix& B, DecompType type) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix A must be square");
    if (A.rows() != B.rows())
        throw std::invalid_argument("Matrix B rows must match A dimensions");
    
    size_t n = A.rows();
    size_t numRHS = B.cols();
    Matrix result(n, numRHS);
    
    if (type == DecompType::LU) {
        LUFactorization lu = luDecompose(A);
        for (size_t j = 0; j < numRHS; ++j) {
            Vector b = B.col(j);
            Vector x = solveLU(lu, b);
            result.setCol(j, x);
        }
    } else {
        for (size_t j = 0; j < numRHS; ++j) {
            Vector b = B.col(j);
            Vector x = solveGaussElimination(A, b);
            result.setCol(j, x);
        }
    }
    
    return result;
}

Vector solveThomas(const Vector& a, const Vector& b, const Vector& c, const Vector& d) {
    size_t n = b.size();
    if (a.size() != n - 1 || c.size() != n - 1 || d.size() != n)
        throw std::invalid_argument("Invalid vector sizes for Thomas algorithm");
    
    Vector c_prime(n);
    Vector d_prime(n);
    Vector x(n);
    
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    
    for (size_t i = 1; i < n - 1; ++i) {
        real_t m = a[i - 1] / (b[i] - a[i - 1] * c_prime[i - 1]);
        c_prime[i] = c[i] * m / -1.0;
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / (b[i] - a[i - 1] * c_prime[i - 1]);
    }
    
    size_t i = n - 1;
    d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / (b[i] - a[i - 1] * c_prime[i - 1]);
    
    x[n - 1] = d_prime[n - 1];
    for (size_t j = n - 2; j-- > 0; ) {
        x[j] = d_prime[j] - c_prime[j] * x[j + 1];
    }
    
    return x;
}

}
