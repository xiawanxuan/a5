#ifndef NUMPCPP_PARALLEL_PARALLEL_H
#define NUMPCPP_PARALLEL_PARALLEL_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"
#include <functional>

namespace numcpp {

unsigned int getNumThreads();
void setNumThreads(unsigned int n);

void parallelFor(size_t begin, size_t end, std::function<void(size_t)> func);

Matrix matMulParallel(const Matrix& A, const Matrix& B);

Vector vecAddParallel(const Vector& a, const Vector& b);
Vector vecMulParallel(const Vector& a, real_t scalar);

real_t integrateParallel(real_t (*f)(real_t), real_t a, real_t b, size_t n = 100000);
real_t integrateTrapezoidalParallel(real_t (*f)(real_t), real_t a, real_t b, size_t n = 100000);

}

#endif
