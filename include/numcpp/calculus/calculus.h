#ifndef NUMPCPP_CALCULUS_CALCULUS_H
#define NUMPCPP_CALCULUS_CALCULUS_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include <functional>

namespace numcpp {

using MultiVarFunction = std::function<real_t(const Vector&)>;

real_t differentiate(ScalarFunction f, real_t x, DiffMethod method = DiffMethod::Central, real_t h = 1e-5);

real_t differentiateSecond(ScalarFunction f, real_t x, real_t h = 1e-4);

Vector gradient(MultiVarFunction f, const Vector& x, real_t h = 1e-5);

Matrix hessian(MultiVarFunction f, const Vector& x, real_t h = 1e-4);

real_t integrateTrapezoidal(ScalarFunction f, real_t a, real_t b, size_t n = 1000);

real_t integrateSimpson(ScalarFunction f, real_t a, real_t b, size_t n = 1000);

real_t integrateRomberg(ScalarFunction f, real_t a, real_t b, real_t tol = 1e-8, size_t maxIter = 20);

real_t integrateAdaptiveSimpson(ScalarFunction f, real_t a, real_t b, real_t tol = 1e-8, size_t maxRecursion = 20);

real_t integrateGaussLegendre(ScalarFunction f, real_t a, real_t b, size_t n = 10);

real_t integrateMonteCarlo(ScalarFunction f, real_t a, real_t b, size_t samples = 100000, unsigned int seed = 42);

std::vector<real_t> gaussLegendreNodes(size_t n);

std::vector<real_t> gaussLegendreWeights(size_t n);

}

#endif
