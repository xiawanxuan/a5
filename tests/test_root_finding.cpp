#include "numcpp/nonlinear/root_finding.h"
#include "test_utils.h"
#include <cmath>

std::vector<TestCase> get_root_finding_tests() {
    return {
        {"Bisection method", []() {
            auto f = [](double x) { return x * x - 4.0; };
            auto result = numcpp::bisection(f, 0.0, 5.0, 1e-8, 100);
            return result.converged && approx_equal(result.root, 2.0, 1e-6);
        }},
        {"Newton method with analytical derivative", []() {
            auto f = [](double x) { return x * x - 4.0; };
            auto df = [](double x) { return 2.0 * x; };
            auto result = numcpp::newtonMethod(f, df, 3.0, 1e-8, 100);
            return result.converged && approx_equal(result.root, 2.0, 1e-8);
        }},
        {"Newton method with numerical derivative", []() {
            auto f = [](double x) { return x * x - 4.0; };
            auto result = numcpp::newtonMethod(f, 3.0, 1e-8, 100);
            return result.converged && approx_equal(result.root, 2.0, 1e-6);
        }},
        {"Secant method", []() {
            auto f = [](double x) { return x * x - 4.0; };
            auto result = numcpp::secantMethod(f, 1.0, 3.0, 1e-8, 100);
            return result.converged && approx_equal(result.root, 2.0, 1e-6);
        }},
        {"Brent method", []() {
            auto f = [](double x) { return x * x - 4.0; };
            auto result = numcpp::brentMethod(f, 0.0, 5.0, 1e-8, 100);
            return result.converged && approx_equal(result.root, 2.0, 1e-8);
        }},
        {"Fixed point iteration", []() {
            auto g = [](double x) { return std::sqrt(4.0 + x) - x + x; };
            auto f = [](double x) { return x * x - x - 4.0; };
            numcpp::ScalarFunction g2 = [](double x) {
                return (x + 4.0 / x) / 2.0;
            };
            auto result = numcpp::fixedPointIteration(g2, 3.0, 1e-8, 100);
            return approx_equal(result.root, 2.0, 1e-6);
        }},
        {"Newton system of equations", []() {
            auto F = [](const numcpp::Vector& x) {
                numcpp::Vector fx(2);
                fx[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
                fx[1] = x[0] - x[1];
                return fx;
            };
            numcpp::Vector x0({1.5, 1.0});
            auto result = numcpp::newtonSystem(F, x0, 1e-8, 50);
            return result.converged &&
                   approx_equal(result.root[0], std::sqrt(2.0), 1e-6) &&
                   approx_equal(result.root[1], std::sqrt(2.0), 1e-6);
        }},
    };
}
