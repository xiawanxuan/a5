#ifndef NUMPCPP_LINALG_EIGENVALUE_H
#define NUMPCPP_LINALG_EIGENVALUE_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"

namespace numcpp {

struct EigenResult {
    Vector eigenvalues;
    Matrix eigenvectors;
};

struct QRResult {
    Matrix Q;
    Matrix R;
};

real_t powerMethod(const Matrix& A, Vector& eigenvector,
                   real_t tol = 1e-8, size_t maxIter = 1000);

real_t inversePowerMethod(const Matrix& A, Vector& eigenvector,
                          real_t tol = 1e-8, size_t maxIter = 1000);

QRResult qrDecompose(const Matrix& A);

EigenResult qrEigenvalues(const Matrix& A, real_t tol = 1e-8, size_t maxIter = 500);

EigenResult jacobiEigenvalues(const Matrix& A, real_t tol = 1e-8, size_t maxIter = 100);

EigenResult eigenvalues(const Matrix& A, EigenMethod method = EigenMethod::QR,
                        real_t tol = 1e-8, size_t maxIter = 500);

Vector eigenvaluesOnly(const Matrix& A, EigenMethod method = EigenMethod::QR,
                       real_t tol = 1e-8, size_t maxIter = 500);

}

#endif
