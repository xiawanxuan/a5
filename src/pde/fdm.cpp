#include "numcpp/pde/fdm.h"
#include "numcpp/linalg/solver.h"
#include "numcpp/parallel/parallel.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef NUMPCPP_HAVE_THREADS
#include <thread>
#endif

namespace numcpp {

FDMResult1D solvePoisson1D(real_t a, real_t b, size_t nx,
                           SourceFunction1D f,
                           BCFunction bc_left, BCFunction bc_right) {
    if (nx < 3) {
        throw std::invalid_argument("Number of grid points must be at least 3");
    }
    
    FDMResult1D result;
    result.x = Vector::linspace(a, b, nx);
    result.u.resize(nx);
    
    real_t dx = (b - a) / (nx - 1);
    size_t n_interior = nx - 2;
    
    Vector diag(n_interior, 2.0);
    Vector off_diag(n_interior - 1, -1.0);
    Vector rhs(n_interior);
    
    real_t dx2 = dx * dx;
    for (size_t i = 0; i < n_interior; ++i) {
        real_t x = result.x[i + 1];
        rhs[i] = -dx2 * f(x);
    }
    
    rhs[0] += bc_left(a);
    rhs[n_interior - 1] += bc_right(b);
    
    Vector u_interior = solveThomas(off_diag, diag, off_diag, rhs);
    
    result.u[0] = bc_left(a);
    result.u[nx - 1] = bc_right(b);
    for (size_t i = 0; i < n_interior; ++i) {
        result.u[i + 1] = u_interior[i];
    }
    
    return result;
}

FDMResult1D solveLaplace1D(real_t a, real_t b, size_t nx,
                            BCFunction bc_left, BCFunction bc_right) {
    return solvePoisson1D(a, b, nx,
                          [](real_t) { return 0.0; },
                          bc_left, bc_right);
}

FDMResult2D solvePoisson2D(real_t ax, real_t bx, size_t nx,
                           real_t ay, real_t by, size_t ny,
                           SourceFunction2D f,
                           BCFunction bc_left, BCFunction bc_right,
                           BCFunction bc_bottom, BCFunction bc_top,
                           real_t tol, size_t maxIter) {
    if (nx < 3 || ny < 3) {
        throw std::invalid_argument("Number of grid points must be at least 3");
    }
    
    FDMResult2D result;
    result.x = Vector::linspace(ax, bx, nx);
    result.y = Vector::linspace(ay, by, ny);
    result.u = Matrix::zeros(ny, nx);
    
    real_t dx = (bx - ax) / (nx - 1);
    real_t dy = (by - ay) / (ny - 1);
    real_t dx2 = dx * dx;
    real_t dy2 = dy * dy;
    
    for (size_t i = 0; i < nx; ++i) {
        result.u(0, i) = bc_bottom(result.x[i]);
        result.u(ny - 1, i) = bc_top(result.x[i]);
    }
    for (size_t j = 0; j < ny; ++j) {
        result.u(j, 0) = bc_left(result.y[j]);
        result.u(j, nx - 1) = bc_right(result.y[j]);
    }
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        real_t max_diff = 0.0;
        
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                real_t u_old = result.u(j, i);
                real_t u_new = (
                    (result.u(j, i - 1) + result.u(j, i + 1)) / dx2 +
                    (result.u(j - 1, i) + result.u(j + 1, i)) / dy2 -
                    f(result.x[i], result.y[j])
                ) / (2.0 / dx2 + 2.0 / dy2);
                
                result.u(j, i) = u_new;
                real_t diff = std::abs(u_new - u_old);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        
        if (max_diff < tol) {
            break;
        }
    }
    
    return result;
}

FDMResult2D solveLaplace2D(real_t ax, real_t bx, size_t nx,
                           real_t ay, real_t by, size_t ny,
                           BCFunction bc_left, BCFunction bc_right,
                           BCFunction bc_bottom, BCFunction bc_top,
                           real_t tol, size_t maxIter) {
    return solvePoisson2D(ax, bx, nx, ay, by, ny,
                          [](real_t, real_t) { return 0.0; },
                          bc_left, bc_right, bc_bottom, bc_top,
                          tol, maxIter);
}

HeatResult1D solveHeatEquation1D(real_t a, real_t b, size_t nx,
                                 real_t t0, real_t t_end, size_t nt,
                                 real_t alpha,
                                 InitialCondition u0,
                                 BCFunction bc_left, BCFunction bc_right) {
    if (nx < 3) {
        throw std::invalid_argument("Number of grid points must be at least 3");
    }
    if (nt < 1) {
        throw std::invalid_argument("Number of time steps must be at least 1");
    }
    
    HeatResult1D result;
    result.x = Vector::linspace(a, b, nx);
    result.t.reserve(nt + 1);
    result.u.reserve(nt + 1);
    
    real_t dx = (b - a) / (nx - 1);
    real_t dt = (t_end - t0) / nt;
    
    real_t r = alpha * dt / (dx * dx);
    
    if (r > 0.5) {
        throw std::runtime_error("Unstable: r = alpha*dt/dx^2 must be <= 0.5 for explicit method");
    }
    
    Vector u_current(nx);
    for (size_t i = 0; i < nx; ++i) {
        u_current[i] = u0(result.x[i]);
    }
    
    result.t.push_back(t0);
    result.u.push_back(u_current);
    
    real_t t = t0;
    for (size_t k = 0; k < nt; ++k) {
        Vector u_next = u_current;
        
        u_next[0] = bc_left(t + dt);
        u_next[nx - 1] = bc_right(t + dt);
        
        for (size_t i = 1; i < nx - 1; ++i) {
            u_next[i] = u_current[i] + r * (u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1]);
        }
        
        t += dt;
        u_current = u_next;
        result.t.push_back(t);
        result.u.push_back(u_current);
    }
    
    return result;
}

HeatResult2D solveHeatEquation2D(real_t ax, real_t bx, size_t nx,
                                 real_t ay, real_t by, size_t ny,
                                 real_t t0, real_t t_end, size_t nt,
                                 real_t alpha,
                                 InitialCondition2D u0,
                                 BCFunction bc_left, BCFunction bc_right,
                                 BCFunction bc_bottom, BCFunction bc_top) {
    if (nx < 3 || ny < 3) {
        throw std::invalid_argument("Number of grid points must be at least 3");
    }
    if (nt < 1) {
        throw std::invalid_argument("Number of time steps must be at least 1");
    }
    
    HeatResult2D result;
    result.x = Vector::linspace(ax, bx, nx);
    result.y = Vector::linspace(ay, by, ny);
    result.t.reserve(nt + 1);
    result.u.reserve(nt + 1);
    
    real_t dx = (bx - ax) / (nx - 1);
    real_t dy = (by - ay) / (ny - 1);
    real_t dt = (t_end - t0) / nt;
    
    real_t rx = alpha * dt / (dx * dx);
    real_t ry = alpha * dt / (dy * dy);
    
    if (rx + ry > 0.5) {
        throw std::runtime_error("Unstable: rx + ry must be <= 0.5 for explicit 2D method");
    }
    
    Matrix u_current(ny, nx);
    for (size_t j = 0; j < ny; ++j) {
        for (size_t i = 0; i < nx; ++i) {
            u_current(j, i) = u0(result.x[i], result.y[j]);
        }
    }
    
    for (size_t i = 0; i < nx; ++i) {
        u_current(0, i) = bc_bottom(result.x[i]);
        u_current(ny - 1, i) = bc_top(result.x[i]);
    }
    for (size_t j = 0; j < ny; ++j) {
        u_current(j, 0) = bc_left(result.y[j]);
        u_current(j, nx - 1) = bc_right(result.y[j]);
    }
    
    result.t.push_back(t0);
    result.u.push_back(u_current);
    
    real_t t = t0;
    for (size_t k = 0; k < nt; ++k) {
        Matrix u_next = u_current;
        
        for (size_t i = 0; i < nx; ++i) {
            u_next(0, i) = bc_bottom(result.x[i]);
            u_next(ny - 1, i) = bc_top(result.x[i]);
        }
        for (size_t j = 0; j < ny; ++j) {
            u_next(j, 0) = bc_left(result.y[j]);
            u_next(j, nx - 1) = bc_right(result.y[j]);
        }
        
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                u_next(j, i) = u_current(j, i) +
                    rx * (u_current(j, i + 1) - 2.0 * u_current(j, i) + u_current(j, i - 1)) +
                    ry * (u_current(j + 1, i) - 2.0 * u_current(j, i) + u_current(j - 1, i));
            }
        }
        
        t += dt;
        u_current = u_next;
        result.t.push_back(t);
        result.u.push_back(u_current);
    }
    
    return result;
}

FDMResult1D solveConvectionDiffusion1D(real_t a, real_t b, size_t nx,
                                       real_t v, real_t D,
                                       SourceFunction1D f,
                                       BCFunction bc_left, BCFunction bc_right) {
    if (nx < 3) {
        throw std::invalid_argument("Number of grid points must be at least 3");
    }
    
    FDMResult1D result;
    result.x = Vector::linspace(a, b, nx);
    result.u.resize(nx);
    
    real_t dx = (b - a) / (nx - 1);
    size_t n_interior = nx - 2;
    
    real_t Pe = std::abs(v) * dx / (2.0 * D);
    
    Vector diag(n_interior);
    Vector lower(n_interior - 1);
    Vector upper(n_interior - 1);
    Vector rhs(n_interior);
    
    for (size_t i = 0; i < n_interior; ++i) {
        diag[i] = 2.0 * D / (dx * dx);
    }
    
    for (size_t i = 0; i < n_interior - 1; ++i) {
        lower[i] = -D / (dx * dx) - v / (2.0 * dx);
        upper[i] = -D / (dx * dx) + v / (2.0 * dx);
    }
    
    for (size_t i = 0; i < n_interior; ++i) {
        real_t x = result.x[i + 1];
        rhs[i] = f(x);
    }
    
    real_t left_val = bc_left(a);
    real_t right_val = bc_right(b);
    
    rhs[0] -= lower[0] * left_val;
    rhs[n_interior - 1] -= upper[n_interior - 2] * right_val;
    
    Vector u_interior = solveThomas(lower, diag, upper, rhs);
    
    result.u[0] = left_val;
    result.u[nx - 1] = right_val;
    for (size_t i = 0; i < n_interior; ++i) {
        result.u[i + 1] = u_interior[i];
    }
    
    return result;
}

}
