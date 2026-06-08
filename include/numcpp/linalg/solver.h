#ifndef NUMPCPP_LINALG_SOLVER_H
#define NUMPCPP_LINALG_SOLVER_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"

namespace numcpp {

struct LUFactorization {
    Matrix L;
    Matrix U;
    std::vector<size_t> pivots;
};

LUFactorization luDecompose(const Matrix& A);

Vector solveLU(const Matrix& A, const Vector& b);
Vector solveLU(const LUFactorization& lu, const Vector& b);

Vector solveGaussElimination(const Matrix& A, const Vector& b);

Vector solveJacobi(const Matrix& A, const Vector& b,
                   real_t tol = 1e-8, size_t maxIter = 1000);

Vector solveGaussSeidel(const Matrix& A, const Vector& b,
                        real_t tol = 1e-8, size_t maxIter = 1000);

Vector solveConjugateGradient(const Matrix& A, const Vector& b,
                              real_t tol = 1e-8, size_t maxIter = 1000);

Matrix solveMultiple(const Matrix& A, const Matrix& B, DecompType type = DecompType::LU);

Vector solveThomas(const Vector& a, const Vector& b, const Vector& c, const Vector& d);

}

#endif
