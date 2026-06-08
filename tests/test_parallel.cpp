#include "numcpp/parallel/parallel.h"
#include "numcpp/core/matrix.h"
#include "test_utils.h"
#include <iostream>
#include <chrono>

std::vector<TestCase> get_parallel_tests() {
    return {
        {"Get number of threads", []() {
            unsigned int n = numcpp::getNumThreads();
            return n > 0;
        }},
        {"Set number of threads", []() {
            unsigned int old = numcpp::getNumThreads();
            numcpp::setNumThreads(2);
            bool ok = numcpp::getNumThreads() == 2;
            numcpp::setNumThreads(old);
            return ok;
        }},
        {"Parallel for loop", []() {
            std::vector<double> data(1000, 0.0);
            numcpp::parallelFor(0, 1000, [&](size_t i) {
                data[i] = static_cast<double>(i) * 2.0;
            });
            bool ok = true;
            for (size_t i = 0; i < 1000; ++i) {
                if (!approx_equal(data[i], static_cast<double>(i) * 2.0)) {
                    ok = false;
                }
            }
            return ok;
        }},
        {"Parallel vector addition", []() {
            numcpp::Vector a(1000);
            numcpp::Vector b(1000);
            for (size_t i = 0; i < 1000; ++i) {
                a[i] = static_cast<double>(i);
                b[i] = static_cast<double>(i * 2);
            }
            auto c = numcpp::vecAddParallel(a, b);
            bool ok = true;
            for (size_t i = 0; i < 1000; ++i) {
                if (!approx_equal(c[i], a[i] + b[i])) {
                    ok = false;
                }
            }
            return ok;
        }},
        {"Parallel vector scalar multiply", []() {
            numcpp::Vector a(1000);
            for (size_t i = 0; i < 1000; ++i) {
                a[i] = static_cast<double>(i);
            }
            auto c = numcpp::vecMulParallel(a, 3.0);
            bool ok = true;
            for (size_t i = 0; i < 1000; ++i) {
                if (!approx_equal(c[i], a[i] * 3.0)) {
                    ok = false;
                }
            }
            return ok;
        }},
        {"Parallel matrix multiplication", []() {
            size_t n = 50;
            numcpp::Matrix A = numcpp::Matrix::random(n, n, 0.0, 1.0);
            numcpp::Matrix B = numcpp::Matrix::random(n, n, 0.0, 1.0);
            numcpp::Matrix C_serial = A * B;
            numcpp::Matrix C_parallel = numcpp::matMulParallel(A, B);
            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    double diff = C_serial(i, j) - C_parallel(i, j);
                    err += diff * diff;
                }
            }
            return std::sqrt(err) < 1e-10;
        }},
        {"Parallel integration", []() {
            double (*f)(double) = [](double x) { return x * x; };
            double integral = numcpp::integrateTrapezoidalParallel(f, 0.0, 1.0, 100000);
            return approx_equal(integral, 1.0 / 3.0, 1e-4);
        }},
    };
}
