#include "numcpp/ode/ode_solver.h"
#include "numcpp/parallel/parallel.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef NUMPCPP_HAVE_THREADS
#include <thread>
#endif

namespace numcpp {

namespace {

real_t rk45_coeffs_a2 = 1.0 / 4.0;
real_t rk45_coeffs_a3[] = {3.0 / 32.0, 9.0 / 32.0};
real_t rk45_coeffs_a4[] = {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0};
real_t rk45_coeffs_a5[] = {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0};
real_t rk45_coeffs_a6[] = {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0};

real_t rk45_b4[] = {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0};
real_t rk45_b5[] = {16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};

}

real_t odeEuler(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    real_t y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        y += h * f(t, y);
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        y += remaining * f(t, y);
    }
    
    return y;
}

real_t odeImprovedEuler(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    real_t y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        real_t k1 = f(t, y);
        real_t k2 = f(t + h, y + h * k1);
        y += 0.5 * h * (k1 + k2);
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        real_t k1 = f(t, y);
        real_t k2 = f(t + remaining, y + remaining * k1);
        y += 0.5 * remaining * (k1 + k2);
    }
    
    return y;
}

real_t odeRK4(ODEFunction1D f, real_t t0, real_t y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    real_t y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        real_t k1 = f(t, y);
        real_t k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        real_t k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        real_t k4 = f(t + h, y + h * k3);
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        real_t k1 = f(t, y);
        real_t k2 = f(t + 0.5 * remaining, y + 0.5 * remaining * k1);
        real_t k3 = f(t + 0.5 * remaining, y + 0.5 * remaining * k2);
        real_t k4 = f(t + remaining, y + remaining * k3);
        y += (remaining / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    
    return y;
}

real_t odeRK45(ODEFunction1D f, real_t t0, real_t y0, real_t t_end,
               real_t tol, real_t h_init, real_t h_min) {
    if (h_init <= 0.0) {
        throw std::invalid_argument("Initial step size must be positive");
    }
    if (tol <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    
    real_t t = t0;
    real_t y = y0;
    real_t h = h_init;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    while ((direction > 0 && t < t_end) || (direction < 0 && t > t_end)) {
        if (std::abs(h) > std::abs(t_end - t)) {
            h = t_end - t;
        }
        
        if (std::abs(h) < h_min) {
            h = h_min * direction;
        }
        
        real_t k1 = f(t, y);
        real_t k2 = f(t + rk45_coeffs_a2 * h, y + rk45_coeffs_a2 * h * k1);
        real_t k3 = f(t + (rk45_coeffs_a3[0] + rk45_coeffs_a3[1]) * h, 
                       y + h * (rk45_coeffs_a3[0] * k1 + rk45_coeffs_a3[1] * k2));
        real_t k4 = f(t + (rk45_coeffs_a4[0] + rk45_coeffs_a4[1] + rk45_coeffs_a4[2]) * h,
                       y + h * (rk45_coeffs_a4[0] * k1 + rk45_coeffs_a4[1] * k2 + rk45_coeffs_a4[2] * k3));
        real_t k5 = f(t + (rk45_coeffs_a5[0] + rk45_coeffs_a5[1] + rk45_coeffs_a5[2] + rk45_coeffs_a5[3]) * h,
                       y + h * (rk45_coeffs_a5[0] * k1 + rk45_coeffs_a5[1] * k2 + rk45_coeffs_a5[2] * k3 + rk45_coeffs_a5[3] * k4));
        real_t k6 = f(t + (rk45_coeffs_a6[0] + rk45_coeffs_a6[1] + rk45_coeffs_a6[2] + rk45_coeffs_a6[3] + rk45_coeffs_a6[4]) * h,
                       y + h * (rk45_coeffs_a6[0] * k1 + rk45_coeffs_a6[1] * k2 + rk45_coeffs_a6[2] * k3 + rk45_coeffs_a6[3] * k4 + rk45_coeffs_a6[4] * k5));
        
        real_t y4 = y + h * (rk45_b4[0] * k1 + rk45_b4[1] * k2 + rk45_b4[2] * k3 + rk45_b4[3] * k4 + rk45_b4[4] * k5);
        real_t y5 = y + h * (rk45_b5[0] * k1 + rk45_b5[1] * k2 + rk45_b5[2] * k3 + rk45_b5[3] * k4 + rk45_b5[4] * k5 + rk45_b5[5] * k6);
        
        real_t error = std::abs(y5 - y4);
        real_t scale = tol * (std::abs(y) + std::abs(h * k1));
        
        real_t factor;
        if (error <= scale) {
            t += h;
            y = y5;
            factor = (error > 0.0) ? 0.9 * std::pow(scale / error, 0.2) : 5.0;
        } else {
            factor = 0.9 * std::pow(scale / error, 0.25);
        }
        
        factor = std::min(std::max(factor, 0.2), 5.0);
        h *= factor;
        
        if (std::abs(h) < h_min) {
            break;
        }
    }
    
    return y;
}

ODE1DResult odeSolve1D(ODEFunction1D f, real_t t0, real_t y0, real_t t_end,
                       real_t h, ODEMethod method) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    ODE1DResult result;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    real_t step = h * direction;
    
    size_t steps = static_cast<size_t>(std::ceil(std::abs((t_end - t0) / step)));
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    
    result.t.push_back(t0);
    result.y.push_back(y0);
    
    real_t t = t0;
    real_t y = y0;
    
    for (size_t i = 0; i < steps; ++i) {
        real_t h_actual = step;
        if ((direction > 0 && t + step > t_end) || (direction < 0 && t + step < t_end)) {
            h_actual = t_end - t;
        }
        
        switch (method) {
            case ODEMethod::Euler:
                y += h_actual * f(t, y);
                break;
            case ODEMethod::ImprovedEuler: {
                real_t k1 = f(t, y);
                real_t k2 = f(t + h_actual, y + h_actual * k1);
                y += 0.5 * h_actual * (k1 + k2);
                break;
            }
            case ODEMethod::RK4:
            case ODEMethod::RK45: {
                real_t k1 = f(t, y);
                real_t k2 = f(t + 0.5 * h_actual, y + 0.5 * h_actual * k1);
                real_t k3 = f(t + 0.5 * h_actual, y + 0.5 * h_actual * k2);
                real_t k4 = f(t + h_actual, y + h_actual * k3);
                y += (h_actual / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                break;
            }
        }
        
        t += h_actual;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    
    return result;
}

Vector odeEulerSystem(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    Vector y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        y = y + f(t, y) * h;
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        y = y + f(t, y) * remaining;
    }
    
    return y;
}

Vector odeImprovedEulerSystem(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    Vector y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        Vector k1 = f(t, y);
        Vector k2 = f(t + h, y + k1 * h);
        y = y + (k1 + k2) * (0.5 * h);
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        Vector k1 = f(t, y);
        Vector k2 = f(t + remaining, y + k1 * remaining);
        y = y + (k1 + k2) * (0.5 * remaining);
    }
    
    return y;
}

Vector odeRK4System(ODEFunction f, real_t t0, const Vector& y0, real_t t_end, real_t h) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    real_t t = t0;
    Vector y = y0;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    size_t steps = static_cast<size_t>(std::abs((t_end - t0) / h));
    for (size_t i = 0; i < steps; ++i) {
        Vector k1 = f(t, y);
        Vector k2 = f(t + 0.5 * h, y + k1 * (0.5 * h));
        Vector k3 = f(t + 0.5 * h, y + k2 * (0.5 * h));
        Vector k4 = f(t + h, y + k3 * h);
        y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
        t += h;
    }
    
    real_t remaining = t_end - t;
    if (std::abs(remaining) > EPS) {
        Vector k1 = f(t, y);
        Vector k2 = f(t + 0.5 * remaining, y + k1 * (0.5 * remaining));
        Vector k3 = f(t + 0.5 * remaining, y + k2 * (0.5 * remaining));
        Vector k4 = f(t + remaining, y + k3 * remaining);
        y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (remaining / 6.0);
    }
    
    return y;
}

Vector odeRK45System(ODEFunction f, real_t t0, const Vector& y0, real_t t_end,
                     real_t tol, real_t h_init, real_t h_min) {
    if (h_init <= 0.0) {
        throw std::invalid_argument("Initial step size must be positive");
    }
    if (tol <= 0.0) {
        throw std::invalid_argument("Tolerance must be positive");
    }
    
    real_t t = t0;
    Vector y = y0;
    real_t h = h_init;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    h *= direction;
    
    while ((direction > 0 && t < t_end) || (direction < 0 && t > t_end)) {
        if (std::abs(h) > std::abs(t_end - t)) {
            h = t_end - t;
        }
        
        if (std::abs(h) < h_min) {
            h = h_min * direction;
        }
        
        Vector k1 = f(t, y);
        Vector k2 = f(t + rk45_coeffs_a2 * h, y + k1 * (rk45_coeffs_a2 * h));
        Vector k3 = f(t + (rk45_coeffs_a3[0] + rk45_coeffs_a3[1]) * h,
                       y + k1 * (rk45_coeffs_a3[0] * h) + k2 * (rk45_coeffs_a3[1] * h));
        Vector k4 = f(t + (rk45_coeffs_a4[0] + rk45_coeffs_a4[1] + rk45_coeffs_a4[2]) * h,
                       y + k1 * (rk45_coeffs_a4[0] * h) + k2 * (rk45_coeffs_a4[1] * h) + k3 * (rk45_coeffs_a4[2] * h));
        Vector k5 = f(t + (rk45_coeffs_a5[0] + rk45_coeffs_a5[1] + rk45_coeffs_a5[2] + rk45_coeffs_a5[3]) * h,
                       y + k1 * (rk45_coeffs_a5[0] * h) + k2 * (rk45_coeffs_a5[1] * h) + k3 * (rk45_coeffs_a5[2] * h) + k4 * (rk45_coeffs_a5[3] * h));
        Vector k6 = f(t + (rk45_coeffs_a6[0] + rk45_coeffs_a6[1] + rk45_coeffs_a6[2] + rk45_coeffs_a6[3] + rk45_coeffs_a6[4]) * h,
                       y + k1 * (rk45_coeffs_a6[0] * h) + k2 * (rk45_coeffs_a6[1] * h) + k3 * (rk45_coeffs_a6[2] * h) + k4 * (rk45_coeffs_a6[3] * h) + k5 * (rk45_coeffs_a6[4] * h));
        
        Vector y4 = y + k1 * (rk45_b4[0] * h) + k2 * (rk45_b4[1] * h) + k3 * (rk45_b4[2] * h) + k4 * (rk45_b4[3] * h) + k5 * (rk45_b4[4] * h);
        Vector y5 = y + k1 * (rk45_b5[0] * h) + k2 * (rk45_b5[1] * h) + k3 * (rk45_b5[2] * h) + k4 * (rk45_b5[3] * h) + k5 * (rk45_b5[4] * h) + k6 * (rk45_b5[5] * h);
        
        real_t max_error = 0.0;
        real_t max_scale = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            real_t error = std::abs(y5[i] - y4[i]);
            real_t scale = tol * (std::abs(y[i]) + std::abs(h * k1[i]));
            if (error > max_error) max_error = error;
            if (scale > max_scale) max_scale = scale;
        }
        
        real_t factor;
        if (max_error <= max_scale) {
            t += h;
            y = y5;
            factor = (max_error > 0.0) ? 0.9 * std::pow(max_scale / max_error, 0.2) : 5.0;
        } else {
            factor = 0.9 * std::pow(max_scale / max_error, 0.25);
        }
        
        factor = std::min(std::max(factor, 0.2), 5.0);
        h *= factor;
        
        if (std::abs(h) < h_min) {
            break;
        }
    }
    
    return y;
}

ODEResult odeSolve(ODEFunction f, real_t t0, const Vector& y0, real_t t_end,
                    real_t h, ODEMethod method) {
    if (h <= 0.0) {
        throw std::invalid_argument("Step size must be positive");
    }
    
    ODEResult result;
    real_t direction = (t_end > t0) ? 1.0 : -1.0;
    real_t step = h * direction;
    
    size_t steps = static_cast<size_t>(std::ceil(std::abs((t_end - t0) / step)));
    result.t.reserve(steps + 1);
    result.y.reserve(steps + 1);
    
    result.t.push_back(t0);
    result.y.push_back(y0);
    
    real_t t = t0;
    Vector y = y0;
    
    for (size_t i = 0; i < steps; ++i) {
        real_t h_actual = step;
        if ((direction > 0 && t + step > t_end) || (direction < 0 && t + step < t_end)) {
            h_actual = t_end - t;
        }
        
        switch (method) {
            case ODEMethod::Euler:
                y = y + f(t, y) * h_actual;
                break;
            case ODEMethod::ImprovedEuler: {
                Vector k1 = f(t, y);
                Vector k2 = f(t + h_actual, y + k1 * h_actual);
                y = y + (k1 + k2) * (0.5 * h_actual);
                break;
            }
            case ODEMethod::RK4:
            case ODEMethod::RK45: {
                Vector k1 = f(t, y);
                Vector k2 = f(t + 0.5 * h_actual, y + k1 * (0.5 * h_actual));
                Vector k3 = f(t + 0.5 * h_actual, y + k2 * (0.5 * h_actual));
                Vector k4 = f(t + h_actual, y + k3 * h_actual);
                y = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h_actual / 6.0);
                break;
            }
        }
        
        t += h_actual;
        result.t.push_back(t);
        result.y.push_back(y);
    }
    
    return result;
}

std::vector<Vector> odeSolveMultiple(const std::vector<ODEFunction>& systems,
                               real_t t0, const std::vector<Vector>& y0s,
                               real_t t_end, real_t h,
                               ODEMethod method) {
    if (systems.size() != y0s.size()) {
        throw std::invalid_argument("Number of systems and initial conditions must match");
    }
    
    size_t n = systems.size();
    std::vector<Vector> results(n);
    
#ifdef NUMPCPP_HAVE_THREADS
    unsigned int nthreads = getNumThreads();
    if (nthreads > 1 && n > 1) {
        parallelFor(0, n, [&](size_t i) {
            switch (method) {
                case ODEMethod::Euler:
                    results[i] = odeEulerSystem(systems[i], t0, y0s[i], t_end, h);
                    break;
                case ODEMethod::ImprovedEuler:
                    results[i] = odeImprovedEulerSystem(systems[i], t0, y0s[i], t_end, h);
                    break;
                case ODEMethod::RK4:
                case ODEMethod::RK45:
                    results[i] = odeRK4System(systems[i], t0, y0s[i], t_end, h);
                    break;
            }
        });
        return results;
    }
#endif
    
    for (size_t i = 0; i < n; ++i) {
        switch (method) {
            case ODEMethod::Euler:
                results[i] = odeEulerSystem(systems[i], t0, y0s[i], t_end, h);
                break;
            case ODEMethod::ImprovedEuler:
                results[i] = odeImprovedEulerSystem(systems[i], t0, y0s[i], t_end, h);
                break;
            case ODEMethod::RK4:
            case ODEMethod::RK45:
                results[i] = odeRK4System(systems[i], t0, y0s[i], t_end, h);
                break;
        }
    }
    
    return results;
}

}
