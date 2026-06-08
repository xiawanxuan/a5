#include "numcpp/linalg/eigenvalue.h"
#include "numcpp/linalg/solver.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace numcpp {

real_t powerMethod(const Matrix& A, Vector& eigenvector, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for power method");
    
    size_t n = A.rows();
    Vector v = Vector::ones(n);
    real_t eigenvalue = 0.0;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        Vector Av = A * v;
        real_t new_eigenvalue = v.dot(Av) / v.dot(v);
        
        v = Av.normalize();
        
        if (std::abs(new_eigenvalue - eigenvalue) < tol) {
            eigenvector = v;
            return new_eigenvalue;
        }
        
        eigenvalue = new_eigenvalue;
    }
    
    std::cerr << "Warning: Power method did not fully converge" << std::endl;
    eigenvector = v;
    return eigenvalue;
}

real_t inversePowerMethod(const Matrix& A, Vector& eigenvector, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for inverse power method");
    
    size_t n = A.rows();
    Vector v = Vector::ones(n);
    real_t eigenvalue = 0.0;
    
    LUFactorization lu = luDecompose(A);
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        Vector y = solveLU(lu, v);
        real_t new_eigenvalue = v.dot(y) / v.dot(v);
        
        v = y.normalize();
        
        if (std::abs(1.0 / new_eigenvalue - eigenvalue) < tol) {
            eigenvector = v;
            return 1.0 / new_eigenvalue;
        }
        
        eigenvalue = 1.0 / new_eigenvalue;
    }
    
    std::cerr << "Warning: Inverse power method did not fully converge" << std::endl;
    eigenvector = v;
    return eigenvalue;
}

QRResult qrDecompose(const Matrix& A) {
    size_t m = A.rows();
    size_t n = A.cols();
    
    QRResult result;
    result.Q = Matrix::identity(m);
    result.R = A;
    
    for (size_t k = 0; k < std::min(m - 1, n); ++k) {
        Vector x(m - k);
        for (size_t i = 0; i < m - k; ++i) {
            x[i] = result.R(k + i, k);
        }
        
        real_t alpha = x.norm();
        if (x[0] < 0) alpha = -alpha;
        
        Vector v = x;
        v[0] += alpha;
        real_t v_norm = v.norm();
        if (v_norm < EPS) continue;
        v = v / v_norm;
        
        for (size_t j = k; j < n; ++j) {
            real_t dot = 0.0;
            for (size_t i = 0; i < m - k; ++i) {
                dot += v[i] * result.R(k + i, j);
            }
            for (size_t i = 0; i < m - k; ++i) {
                result.R(k + i, j) -= 2.0 * v[i] * dot;
            }
        }
        
        for (size_t j = 0; j < m; ++j) {
            real_t dot = 0.0;
            for (size_t i = 0; i < m - k; ++i) {
                dot += v[i] * result.Q(k + i, j);
            }
            for (size_t i = 0; i < m - k; ++i) {
                result.Q(k + i, j) -= 2.0 * v[i] * dot;
            }
        }
    }
    
    result.Q = result.Q.transpose();
    
    return result;
}

EigenResult qrEigenvalues(const Matrix& A, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    
    size_t n = A.rows();
    Matrix Ak = A;
    Matrix eigenvectors = Matrix::identity(n);
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        QRResult qr = qrDecompose(Ak);
        Ak = qr.R * qr.Q;
        eigenvectors = eigenvectors * qr.Q;
        
        real_t off_diag = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < i; ++j) {
                off_diag += Ak(i, j) * Ak(i, j);
            }
        }
        
        if (std::sqrt(off_diag) < tol) {
            break;
        }
    }
    
    EigenResult result;
    result.eigenvalues = Vector(n);
    for (size_t i = 0; i < n; ++i) {
        result.eigenvalues[i] = Ak(i, i);
    }
    result.eigenvectors = eigenvectors;
    
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (result.eigenvalues[j] > result.eigenvalues[i]) {
                std::swap(result.eigenvalues[i], result.eigenvalues[j]);
                for (size_t k = 0; k < n; ++k) {
                    std::swap(result.eigenvectors(k, i), result.eigenvectors(k, j));
                }
            }
        }
    }
    
    return result;
}

EigenResult jacobiEigenvalues(const Matrix& A, real_t tol, size_t maxIter) {
    if (!A.isSquare())
        throw std::invalid_argument("Matrix must be square for Jacobi method");
    if (!A.isSymmetric())
        std::cerr << "Warning: Jacobi method is designed for symmetric matrices" << std::endl;
    
    size_t n = A.rows();
    Matrix M = A;
    Matrix V = Matrix::identity(n);
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        size_t p = 0, q = 1;
        real_t max_val = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                if (std::abs(M(i, j)) > max_val) {
                    max_val = std::abs(M(i, j));
                    p = i;
                    q = j;
                }
            }
        }
        
        if (max_val < tol) break;
        
        real_t theta = (M(q, q) - M(p, p)) / (2.0 * M(p, q));
        real_t t;
        if (std::abs(theta) < EPS) {
            t = 1.0;
        } else {
            t = (theta > 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        }
        real_t c = 1.0 / std::sqrt(1.0 + t * t);
        real_t s = t * c;
        
        real_t mpp = M(p, p);
        real_t mqq = M(q, q);
        real_t mpq = M(p, q);
        
        M(p, p) = c * c * mpp - 2.0 * s * c * mpq + s * s * mqq;
        M(q, q) = s * s * mpp + 2.0 * s * c * mpq + c * c * mqq;
        M(p, q) = 0.0;
        M(q, p) = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            if (i != p && i != q) {
                real_t mip = M(i, p);
                real_t miq = M(i, q);
                M(i, p) = c * mip - s * miq;
                M(p, i) = M(i, p);
                M(i, q) = s * mip + c * miq;
                M(q, i) = M(i, q);
            }
        }
        
        for (size_t i = 0; i < n; ++i) {
            real_t vip = V(i, p);
            real_t viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
        }
    }
    
    EigenResult result;
    result.eigenvalues = Vector(n);
    for (size_t i = 0; i < n; ++i) {
        result.eigenvalues[i] = M(i, i);
    }
    result.eigenvectors = V;
    
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (result.eigenvalues[j] > result.eigenvalues[i]) {
                std::swap(result.eigenvalues[i], result.eigenvalues[j]);
                for (size_t k = 0; k < n; ++k) {
                    std::swap(result.eigenvectors(k, i), result.eigenvectors(k, j));
                }
            }
        }
    }
    
    return result;
}

EigenResult eigenvalues(const Matrix& A, EigenMethod method, real_t tol, size_t maxIter) {
    switch (method) {
        case EigenMethod::QR:
            return qrEigenvalues(A, tol, maxIter);
        case EigenMethod::Jacobi:
            return jacobiEigenvalues(A, tol, maxIter);
        case EigenMethod::Power: {
            if (!A.isSquare())
                throw std::invalid_argument("Matrix must be square for power method");
            size_t n = A.rows();
            EigenResult result;
            result.eigenvalues = Vector(n);
            result.eigenvectors = Matrix::zeros(n, n);
            Vector ev;
            result.eigenvalues[0] = powerMethod(A, ev, tol, maxIter);
            result.eigenvectors.setCol(0, ev);
            return result;
        }
        default:
            return qrEigenvalues(A, tol, maxIter);
    }
}

Vector eigenvaluesOnly(const Matrix& A, EigenMethod method, real_t tol, size_t maxIter) {
    EigenResult result = eigenvalues(A, method, tol, maxIter);
    return result.eigenvalues;
}

}
