#ifndef NUMPCPP_ODE_ODE_SOLVER_H
#define NUMPCPP_ODE_ODE_SOLVER_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include <functional>
#include <vector>

namespace numcpp {

using ODEFunction = std::function<Vector(real_t, const Vector&)>;
using ODEFunction1D = std::function<real_t(real_t, real_t)>;

enum class ODEMethod {
    Euler,
    ImprovedEuler,
    RK4,
    RK45
};

struct ODEResult {
    std::vector<real_t> t;
    std::vector<Vector> y;
};

struct ODE1DResult {
    std::vector<real_t> t;
    std::vector<real_t> y;
};

real_t odeEuler(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h);

real_t odeImprovedEuler(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h);

real_t odeRK4(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h);

real_t odeRK45(ODEFunction1D f, real_t t0, real_t y0, real_t t_end,
                 real_t tol = 1e-6, real_t h_init = 0.01, real_t h_min = 1e-10);

ODE1DResult odeSolve1D(ODEFunction1D f, real_t t0, real_t y0, real_t t_end,
                       real_t h = 0.01, ODEMethod method = ODEMethod::RK4);

Vector odeEulerSystem(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h);

Vector odeImprovedEulerSystem(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h);

Vector odeRK4System(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h);

Vector odeRK45System(ODEFunction f, real_t t0, const Vector& y0, real_t t_end,
                      real_t tol = 1e-6, real_t h_init = 0.01, real_t h_min = 1e-10);

ODEResult odeSolve(ODEFunction f, real_t t0, const Vector& y0, real_t t_end,
                    real_t h = 0.01, ODEMethod method = ODEMethod::RK4);

std::vector<Vector> odeSolveMultiple(const std::vector<ODEFunction>& systems,
                               real_t t0, const std::vector<Vector>& y0s,
                               real_t t_end, real_t h = 0.01,
                               ODEMethod method = ODEMethod::RK4);

}

#endif
