#include "numcpp/pde/fdm.h"
#include "numcpp/core/types.h"
#include "test_utils.h"
#include <cmath>

std::vector<TestCase> get_pde_tests() {
    return {
        {"PDE Poisson 1D - constant source", []() {
            auto f = [](real_t x) { return -2.0; };
            auto bc_left = [](real_t x) { return 0.0; };
            auto bc_right = [](real_t x) { return 0.0; };
            auto result = numcpp::solvePoisson1D(0.0, 1.0, 21, f, bc_left, bc_right);
            real_t mid_val = result.u[10];
            real_t expected = 0.5 * 0.5;
            return std::abs(mid_val - expected) < 0.001;
        }},
        
        {"PDE Laplace 1D - linear solution", []() {
            auto bc_left = [](real_t x) { return 0.0; };
            auto bc_right = [](real_t x) { return 1.0; };
            auto result = numcpp::solveLaplace1D(0.0, 1.0, 11, bc_left, bc_right);
            bool ok = true;
            for (size_t i = 0; i < result.x.size(); ++i) {
                real_t expected = result.x[i];
                if (std::abs(result.u[i] - expected) > 1e-10) {
                    ok = false;
                    break;
                }
            }
            return ok;
        }},
        
        {"PDE Poisson 2D - constant source", []() {
            auto f = [](real_t x, real_t y) { return -2.0; };
            auto bc_left = [](real_t y) { return 0.0; };
            auto bc_right = [](real_t y) { return 0.0; };
            auto bc_bottom = [](real_t x) { return 0.0; };
            auto bc_top = [](real_t x) { return 0.0; };
            auto result = numcpp::solvePoisson2D(0.0, 1.0, 11,
                                                  0.0, 1.0, 11,
                                                  f, bc_left, bc_right, bc_bottom, bc_top,
                                                  1e-6, 2000);
            return result.u(5, 5) > 0.0 && result.u(5, 5) < 0.5;
        }},
        
        {"PDE Laplace 2D - boundary conditions", []() {
            auto bc_left = [](real_t y) { return 0.0; };
            auto bc_right = [](real_t y) { return 1.0; };
            auto bc_bottom = [](real_t x) { return x; };
            auto bc_top = [](real_t x) { return x; };
            auto result = numcpp::solveLaplace2D(0.0, 1.0, 11,
                                                  0.0, 1.0, 11,
                                                  bc_left, bc_right, bc_bottom, bc_top,
                                                  1e-6, 2000);
            return approx_equal(result.u(0, 0), 0.0) &&
                   approx_equal(result.u(10, 0), 0.0) &&
                   approx_equal(result.u(0, 10), 1.0) &&
                   approx_equal(result.u(10, 10), 1.0);
        }},
        
        {"PDE Heat equation 1D - initial condition", []() {
            auto u0 = [](real_t x) { return std::sin(numcpp::PI * x); };
            auto bc_left = [](real_t t) { return 0.0; };
            auto bc_right = [](real_t t) { return 0.0; };
            auto result = numcpp::solveHeatEquation1D(0.0, 1.0, 21,
                                                       0.0, 0.1, 100,
                                                       0.01,
                                                       u0, bc_left, bc_right);
            return result.t.size() == 101 &&
                   result.u.size() == 101 &&
                   result.u[0][10] > result.u[100][10];
        }},
        
        {"PDE Heat equation 2D - initial condition", []() {
            auto u0 = [](real_t x, real_t y) { return std::sin(numcpp::PI * x) * std::sin(numcpp::PI * y); };
            auto bc_left = [](real_t y) { return 0.0; };
            auto bc_right = [](real_t y) { return 0.0; };
            auto bc_bottom = [](real_t x) { return 0.0; };
            auto bc_top = [](real_t x) { return 0.0; };
            auto result = numcpp::solveHeatEquation2D(0.0, 1.0, 11,
                                                       0.0, 1.0, 11,
                                                       0.0, 0.01, 10,
                                                       0.001,
                                                       u0, bc_left, bc_right, bc_bottom, bc_top);
            return result.t.size() == 11 &&
                   result.u.size() == 11 &&
                   result.u[0](5, 5) > result.u[10](5, 5);
        }},
        
        {"PDE Convection-Diffusion 1D", []() {
            auto f = [](real_t x) { return 0.0; };
            auto bc_left = [](real_t x) { return 1.0; };
            auto bc_right = [](real_t x) { return 0.0; };
            auto result = numcpp::solveConvectionDiffusion1D(0.0, 1.0, 51,
                                                               1.0, 0.1, f,
                                                               bc_left, bc_right);
            return approx_equal(result.u[0], 1.0) &&
                   approx_equal(result.u[50], 0.0) &&
                   result.u[25] > 0.0 && result.u[25] < 1.0;
        }},
        
        {"PDE FDMResult1D structure", []() {
            auto bc_left = [](real_t x) { return 0.0; };
            auto bc_right = [](real_t x) { return 1.0; };
            auto result = numcpp::solveLaplace1D(0.0, 1.0, 11, bc_left, bc_right);
            return result.x.size() == 11 &&
                   result.u.size() == 11 &&
                   result.x.size() == result.u.size();
        }},
        
        {"PDE FDMResult2D structure", []() {
            auto bc_left = [](real_t y) { return 0.0; };
            auto bc_right = [](real_t y) { return 0.0; };
            auto bc_bottom = [](real_t x) { return 0.0; };
            auto bc_top = [](real_t x) { return 0.0; };
            auto f = [](real_t x, real_t y) { return 1.0; };
            auto result = numcpp::solvePoisson2D(0.0, 1.0, 10,
                                                  0.0, 1.0, 10,
                                                  f, bc_left, bc_right, bc_bottom, bc_top,
                                                  1e-3, 100);
            return result.x.size() == 10 &&
                   result.y.size() == 10 &&
                   result.u.rows() == 10 &&
                   result.u.cols() == 10;
        }},
    };
}
