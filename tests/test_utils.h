#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <string>
#include <functional>
#include <cmath>

struct TestCase {
    std::string name;
    std::function<bool()> func;
};

inline bool approx_equal(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

#endif
