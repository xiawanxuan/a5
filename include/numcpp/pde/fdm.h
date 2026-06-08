#ifndef NUMPCPP_PDE_FDM_H
#define NUMPCPP_PDE_FDM_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"
#include <functional>

namespace numcpp {

using BCFunction = std::function<real_t(real_t)>;
using SourceFunction1D = std::function<real_t(real_t)>;
using SourceFunction2D = std::function<real_t(real_t, real_t)>;
using InitialCondition = std::function<real_t(real_t)>;
using InitialCondition2D = std::function<real_t(real_t, real_t)>;

struct FDMResult1D {
    Vector x;
    Vector u;
};

struct FDMResult2D {
    Vector x;
    Vector y;
    Matrix u;
};

struct HeatResult1D {
    Vector x;
    std::vector<real_t> t;
    std::vector<Vector> u;
};

struct HeatResult2D {
    Vector x;
    Vector y;
    std::vector<real_t> t;
    std::vector<Matrix> u;
};

FDMResult1D solvePoisson1D(real_t a, real_t b, size_t nx,
                           SourceFunction1D f,
                           BCFunction bc_left, BCFunction bc_right);

FDMResult1D solveLaplace1D(real_t a, real_t b, size_t nx,
                            BCFunction bc_left, BCFunction bc_right);

FDMResult2D solvePoisson2D(real_t ax, real_t bx, size_t nx,
                           real_t ay, real_t by, size_t ny,
                           SourceFunction2D f,
                           BCFunction bc_left, BCFunction bc_right,
                           BCFunction bc_bottom, BCFunction bc_top,
                           real_t tol = 1e-8, size_t maxIter = 10000);

FDMResult2D solveLaplace2D(real_t ax, real_t bx, size_t nx,
                           real_t ay, real_t by, size_t ny,
                           BCFunction bc_left, BCFunction bc_right,
                           BCFunction bc_bottom, BCFunction bc_top,
                           real_t tol = 1e-8, size_t maxIter = 10000);

HeatResult1D solveHeatEquation1D(real_t a, real_t b, size_t nx,
                                 real_t t0, real_t t_end, size_t nt,
                                 real_t alpha,
                                 InitialCondition u0,
                                 BCFunction bc_left, BCFunction bc_right);

HeatResult2D solveHeatEquation2D(real_t ax, real_t bx, size_t nx,
                                 real_t ay, real_t by, size_t ny,
                                 real_t t0, real_t t_end, size_t nt,
                                 real_t alpha,
                                 InitialCondition2D u0,
                                 BCFunction bc_left, BCFunction bc_right,
                                 BCFunction bc_bottom, BCFunction bc_top);

FDMResult1D solveConvectionDiffusion1D(real_t a, real_t b, size_t nx,
                                       real_t v, real_t D,
                                       SourceFunction1D f,
                                       BCFunction bc_left, BCFunction bc_right);

}

#endif
