#include "numcpp/parallel/parallel.h"

#ifdef NUMPCPP_HAVE_THREADS
#include <thread>
#endif

#include <vector>
#include <algorithm>
#include <cmath>

namespace numcpp {

static unsigned int num_threads = 
#ifdef NUMPCPP_HAVE_THREADS
    (std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4);
#else
    1;
#endif

unsigned int getNumThreads() {
    return num_threads;
}

void setNumThreads(unsigned int n) {
    if (n > 0) {
        num_threads = 
#ifdef NUMPCPP_HAVE_THREADS
            n;
#else
            1;
#endif
    }
#ifdef NUMPCPP_HAVE_OPENMP
    omp_set_num_threads(n);
#endif
}

void parallelFor(size_t begin, size_t end, std::function<void(size_t)> func) {
    if (begin >= end) return;
    
    size_t total = end - begin;
    unsigned int nthreads = num_threads;
    
#ifdef NUMPCPP_HAVE_THREADS
    if (total < nthreads || nthreads <= 1) {
        for (size_t i = begin; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    size_t chunk = total / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    
    for (unsigned int t = 0; t < nthreads; ++t) {
        size_t start = begin + t * chunk;
        size_t stop = (t == nthreads - 1) ? end : begin + (t + 1) * chunk;
        
        threads.emplace_back([start, stop, &func]() {
            for (size_t i = start; i < stop; ++i) {
                func(i);
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
#else
    for (size_t i = begin; i < end; ++i) {
        func(i);
    }
#endif
}

Matrix matMulParallel(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    size_t m = A.rows();
    size_t n = B.cols();
    size_t k = A.cols();
    
    Matrix result(m, n);
    
    parallelFor(0, m, [&](size_t i) {
        for (size_t j = 0; j < n; ++j) {
            real_t sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += A(i, p) * B(p, j);
            }
            result(i, j) = sum;
        }
    });
    
    return result;
}

Vector vecAddParallel(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match for addition");
    }
    
    Vector result(a.size());
    
    parallelFor(0, a.size(), [&](size_t i) {
        result[i] = a[i] + b[i];
    });
    
    return result;
}

Vector vecMulParallel(const Vector& a, real_t scalar) {
    Vector result(a.size());
    
    parallelFor(0, a.size(), [&](size_t i) {
        result[i] = a[i] * scalar;
    });
    
    return result;
}

real_t integrateTrapezoidalParallel(real_t (*f)(real_t), real_t a, real_t b, size_t n) {
    if (n == 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    
    real_t h = (b - a) / n;
    real_t sum = 0.5 * (f(a) + f(b));
    
    unsigned int nthreads = num_threads;
    
#ifdef NUMPCPP_HAVE_THREADS
    if (nthreads <= 1 || n < nthreads * 2) {
        for (size_t i = 1; i < n; ++i) {
            sum += f(a + i * h);
        }
        return sum * h;
    }
    
    std::vector<real_t> partial_sums(nthreads, 0.0);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    
    size_t chunk = (n - 1) / nthreads;
    
    for (unsigned int t = 0; t < nthreads; ++t) {
        size_t start = 1 + t * chunk;
        size_t stop = (t == nthreads - 1) ? n : 1 + (t + 1) * chunk;
        
        threads.emplace_back([t, start, stop, &partial_sums, &f, a, h]() {
            real_t local_sum = 0.0;
            for (size_t i = start; i < stop; ++i) {
                local_sum += f(a + i * h);
            }
            partial_sums[t] = local_sum;
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    for (auto s : partial_sums) {
        sum += s;
    }
#else
    for (size_t i = 1; i < n; ++i) {
        sum += f(a + i * h);
    }
#endif
    
    return sum * h;
}

real_t integrateParallel(real_t (*f)(real_t), real_t a, real_t b, size_t n) {
    return integrateTrapezoidalParallel(f, a, b, n);
}

}
