#include "numcpp/calculus/calculus.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <random>
#include <algorithm>

namespace numcpp {

real_t differentiate(ScalarFunction f, real_t x, DiffMethod method, real_t h) {
    switch (method) {
        case DiffMethod::Forward:
            return (f(x + h) - f(x)) / h;
        case DiffMethod::Backward:
            return (f(x) - f(x - h)) / h;
        case DiffMethod::Central:
        default:
            return (f(x + h) - f(x - h)) / (2.0 * h);
    }
}

real_t differentiateSecond(ScalarFunction f, real_t x, real_t h) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

Vector gradient(MultiVarFunction f, const Vector& x, real_t h) {
    size_t n = x.size();
    Vector grad(n);
    
    for (size_t i = 0; i < n; ++i) {
        Vector x_plus = x;
        Vector x_minus = x;
        x_plus[i] += h;
        x_minus[i] -= h;
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h);
    }
    
    return grad;
}

Matrix hessian(MultiVarFunction f, const Vector& x, real_t h) {
    size_t n = x.size();
    Matrix hess(n, n);
    real_t f0 = f(x);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                Vector xp = x; xp[i] += h;
                Vector xm = x; xm[i] -= h;
                hess(i, j) = (f(xp) - 2.0 * f0 + f(xm)) / (h * h);
            } else {
                Vector xpp = x; xpp[i] += h; xpp[j] += h;
                Vector xpm = x; xpm[i] += h; xpm[j] -= h;
                Vector xmp = x; xmp[i] -= h; xmp[j] += h;
                Vector xmm = x; xmm[i] -= h; xmm[j] -= h;
                hess(i, j) = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h);
            }
        }
    }
    
    return hess;
}

real_t integrateTrapezoidal(ScalarFunction f, real_t a, real_t b, size_t n) {
    if (n == 0)
        throw std::invalid_argument("Number of intervals must be positive");
    
    real_t h = (b - a) / n;
    real_t sum = 0.5 * (f(a) + f(b));
    
    for (size_t i = 1; i < n; ++i) {
        sum += f(a + i * h);
    }
    
    return sum * h;
}

real_t integrateSimpson(ScalarFunction f, real_t a, real_t b, size_t n) {
    if (n == 0)
        throw std::invalid_argument("Number of intervals must be positive");
    if (n % 2 != 0) {
        n += 1;
    }
    
    real_t h = (b - a) / n;
    real_t sum = f(a) + f(b);
    
    for (size_t i = 1; i < n; ++i) {
        real_t x = a + i * h;
        if (i % 2 == 0) {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }
    
    return sum * h / 3.0;
}

real_t integrateRomberg(ScalarFunction f, real_t a, real_t b, real_t tol, size_t maxIter) {
    std::vector<std::vector<real_t>> R(maxIter + 1, std::vector<real_t>(maxIter + 1, 0.0));
    
    R[0][0] = 0.5 * (b - a) * (f(a) + f(b));
    
    for (size_t i = 1; i <= maxIter; ++i) {
        size_t n = 1 << i;
        real_t h = (b - a) / n;
        real_t sum = 0.0;
        
        for (size_t k = 1; k < n; k += 2) {
            sum += f(a + k * h);
        }
        
        R[i][0] = 0.5 * R[i - 1][0] + h * sum;
        
        for (size_t j = 1; j <= i; ++j) {
            real_t factor = std::pow(4.0, j);
            R[i][j] = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0);
        }
        
        if (i > 0 && std::abs(R[i][i] - R[i - 1][i - 1]) < tol) {
            return R[i][i];
        }
    }
    
    return R[maxIter][maxIter];
}

namespace {

real_t adaptiveSimpsonHelper(ScalarFunction f, real_t a, real_t b, real_t tol,
                              real_t fa, real_t fb, real_t fc, size_t depth, size_t maxDepth) {
    real_t c = (a + b) / 2.0;
    real_t h = b - a;
    real_t fd = f((a + c) / 2.0);
    real_t fe = f((c + b) / 2.0);
    
    real_t Simpson = h / 6.0 * (fa + 4.0 * fc + fb);
    real_t Simpson2 = h / 12.0 * (fa + 4.0 * fd + 2.0 * fc + 4.0 * fe + fb);
    
    if (depth >= maxDepth || std::abs(Simpson2 - Simpson) <= 15.0 * tol) {
        return Simpson2 + (Simpson2 - Simpson) / 15.0;
    }
    
    return adaptiveSimpsonHelper(f, a, c, tol / 2.0, fa, fc, fd, depth + 1, maxDepth) +
           adaptiveSimpsonHelper(f, c, b, tol / 2.0, fc, fb, fe, depth + 1, maxDepth);
}

}

real_t integrateAdaptiveSimpson(ScalarFunction f, real_t a, real_t b, real_t tol, size_t maxRecursion) {
    real_t fa = f(a);
    real_t fb = f(b);
    real_t fc = f((a + b) / 2.0);
    return adaptiveSimpsonHelper(f, a, b, tol, fa, fb, fc, 0, maxRecursion);
}

namespace {

void legendreRecurrence(size_t n, real_t x, real_t& pn, real_t& pn_1, real_t& dpn) {
    pn = 1.0;
    pn_1 = 0.0;
    real_t dpn_1 = 0.0;
    dpn = 0.0;
    
    for (size_t k = 1; k <= n; ++k) {
        real_t pn_2 = pn_1;
        pn_1 = pn;
        real_t dpn_2 = dpn_1;
        dpn_1 = dpn;
        
        if (k == 1) {
            pn = x;
            dpn = 1.0;
        } else {
            pn = ((2.0 * k - 1.0) * x * pn_1 - (k - 1.0) * pn_2) / k;
            dpn = ((2.0 * k - 1.0) * (pn_1 + x * dpn_1) - (k - 1.0) * dpn_2) / k;
        }
    }
}

}

std::vector<real_t> gaussLegendreNodes(size_t n) {
    if (n == 0) return {};
    
    std::vector<real_t> nodes(n);
    real_t pn, pn_1, dpn;
    
    for (size_t i = 0; i < n / 2; ++i) {
        real_t x = std::cos(PI * (i + 0.75) / (n + 0.5));
        
        for (size_t iter = 0; iter < 100; ++iter) {
            legendreRecurrence(n, x, pn, pn_1, dpn);
            real_t dx = -pn / dpn;
            x += dx;
            if (std::abs(dx) < 1e-15) break;
        }
        
        nodes[i] = -x;
        nodes[n - 1 - i] = x;
    }
    
    if (n % 2 == 1) {
        nodes[n / 2] = 0.0;
    }
    
    return nodes;
}

std::vector<real_t> gaussLegendreWeights(size_t n) {
    if (n == 0) return {};
    
    std::vector<real_t> weights(n);
    std::vector<real_t> nodes = gaussLegendreNodes(n);
    real_t pn, pn_1, dpn;
    
    for (size_t i = 0; i < n; ++i) {
        legendreRecurrence(n, nodes[i], pn, pn_1, dpn);
        weights[i] = 2.0 / ((1.0 - nodes[i] * nodes[i]) * dpn * dpn);
    }
    
    return weights;
}

real_t integrateGaussLegendre(ScalarFunction f, real_t a, real_t b, size_t n) {
    std::vector<real_t> nodes = gaussLegendreNodes(n);
    std::vector<real_t> weights = gaussLegendreWeights(n);
    
    real_t half = 0.5 * (b - a);
    real_t mid = 0.5 * (a + b);
    
    real_t sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += weights[i] * f(half * nodes[i] + mid);
    }
    
    return half * sum;
}

real_t integrateMonteCarlo(ScalarFunction f, real_t a, real_t b, size_t samples, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real_t> dist(a, b);
    
    real_t sum = 0.0;
    for (size_t i = 0; i < samples; ++i) {
        sum += f(dist(gen));
    }
    
    return (b - a) * sum / samples;
}

}
