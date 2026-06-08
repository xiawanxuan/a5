#ifndef NUMPCPP_NONLINEAR_ROOT_FINDING_H
#define NUMPCPP_NONLINEAR_ROOT_FINDING_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"
#include <functional>

namespace numcpp {

using VectorFunction = std::function<Vector(const Vector&)>;
using JacobianFunction = std::function<Matrix(const Vector&)>;

struct RootResult {
    real_t root;
    size_t iterations;
    bool converged;
    real_t residual;
};

struct VectorRootResult {
    Vector root;
    size_t iterations;
    bool converged;
    real_t residual;
};

RootResult bisection(ScalarFunction f, real_t a, real_t b,
                     real_t tol = 1e-8, size_t maxIter = 100);

RootResult newtonMethod(ScalarFunction f, ScalarFunction df, real_t x0,
                        real_t tol = 1e-8, size_t maxIter = 100);

RootResult newtonMethod(ScalarFunction f, real_t x0,
                        real_t tol = 1e-8, size_t maxIter = 100,
                        real_t h = 1e-5);

RootResult secantMethod(ScalarFunction f, real_t x0, real_t x1,
                        real_t tol = 1e-8, size_t maxIter = 100);

RootResult fixedPointIteration(ScalarFunction g, real_t x0,
                               real_t tol = 1e-8, size_t maxIter = 100);

RootResult brentMethod(ScalarFunction f, real_t a, real_t b,
                       real_t tol = 1e-8, size_t maxIter = 100);

VectorRootResult newtonSystem(VectorFunction F, JacobianFunction J, const Vector& x0,
                              real_t tol = 1e-8, size_t maxIter = 50);

VectorRootResult newtonSystem(VectorFunction F, const Vector& x0,
                              real_t tol = 1e-8, size_t maxIter = 50,
                              real_t h = 1e-5);

}

#endif
