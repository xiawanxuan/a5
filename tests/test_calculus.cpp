#include "numcpp/calculus/calculus.h"
#include "test_utils.h"
#include <cmath>

std::vector<TestCase> get_calculus_tests() {
    return {
        {"Forward differentiation", []() {
            auto f = [](double x) { return x * x; };
            double df = numcpp::differentiate(f, 2.0, numcpp::DiffMethod::Forward, 1e-5);
            return approx_equal(df, 4.0, 1e-4);
        }},
        {"Central differentiation", []() {
            auto f = [](double x) { return x * x; };
            double df = numcpp::differentiate(f, 2.0, numcpp::DiffMethod::Central, 1e-5);
            return approx_equal(df, 4.0, 1e-8);
        }},
        {"Second derivative", []() {
            auto f = [](double x) { return x * x * x; };
            double d2f = numcpp::differentiateSecond(f, 2.0, 1e-4);
            return approx_equal(d2f, 12.0, 1e-4);
        }},
        {"Gradient of multi-variable function", []() {
            auto f = [](const numcpp::Vector& x) {
                return x[0] * x[0] + x[1] * x[1];
            };
            numcpp::Vector x({2.0, 3.0});
            auto grad = numcpp::gradient(f, x, 1e-5);
            return approx_equal(grad[0], 4.0, 1e-4) &&
                   approx_equal(grad[1], 6.0, 1e-4);
        }},
        {"Hessian matrix", []() {
            auto f = [](const numcpp::Vector& x) {
                return x[0] * x[0] + x[1] * x[1] + x[0] * x[1];
            };
            numcpp::Vector x({1.0, 1.0});
            auto hess = numcpp::hessian(f, x, 1e-4);
            return approx_equal(hess(0, 0), 2.0, 1e-3) &&
                   approx_equal(hess(1, 1), 2.0, 1e-3) &&
                   approx_equal(hess(0, 1), 1.0, 1e-3);
        }},
        {"Trapezoidal integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateTrapezoidal(f, 0.0, 1.0, 10000);
            return approx_equal(integral, 1.0 / 3.0, 1e-4);
        }},
        {"Simpson integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateSimpson(f, 0.0, 1.0, 1000);
            return approx_equal(integral, 1.0 / 3.0, 1e-6);
        }},
        {"Romberg integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateRomberg(f, 0.0, 1.0, 1e-10, 20);
            return approx_equal(integral, 1.0 / 3.0, 1e-8);
        }},
        {"Adaptive Simpson integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateAdaptiveSimpson(f, 0.0, 1.0, 1e-10, 20);
            return approx_equal(integral, 1.0 / 3.0, 1e-8);
        }},
        {"Gauss-Legendre integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateGaussLegendre(f, 0.0, 1.0, 10);
            return approx_equal(integral, 1.0 / 3.0, 1e-8);
        }},
        {"Gauss-Legendre nodes and weights", []() {
            auto nodes = numcpp::gaussLegendreNodes(4);
            auto weights = numcpp::gaussLegendreWeights(4);
            double sum = 0.0;
            for (size_t i = 0; i < 4; ++i) sum += weights[i];
            return nodes.size() == 4 && weights.size() == 4 &&
                   approx_equal(sum, 2.0, 1e-10);
        }},
        {"Monte Carlo integration", []() {
            auto f = [](double x) { return x * x; };
            double integral = numcpp::integrateMonteCarlo(f, 0.0, 1.0, 100000, 42);
            return std::abs(integral - 1.0 / 3.0) < 0.01;
        }},
    };
}
