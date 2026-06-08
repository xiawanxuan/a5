#ifndef NUMPCPP_CORE_TYPES_H
#define NUMPCPP_CORE_TYPES_H

#include <complex>
#include <cstddef>
#include <vector>
#include <functional>
#include <limits>

namespace numcpp {

using real_t = double;
using complex_t = std::complex<double>;
using index_t = std::ptrdiff_t;
using size_t = std::size_t;

constexpr real_t PI = 3.14159265358979323846264338327950288;
constexpr real_t EPS = std::numeric_limits<real_t>::epsilon() * 100;
constexpr real_t INF = std::numeric_limits<real_t>::max();

enum class DecompType {
    LU,
    QR,
    Cholesky,
    SVD
};

enum class EigenMethod {
    Power,
    QR,
    Jacobi
};

enum class FFTDirection {
    Forward,
    Inverse
};

enum class DiffMethod {
    Forward,
    Backward,
    Central
};

using ScalarFunction = std::function<real_t(real_t)>;

}

#endif
