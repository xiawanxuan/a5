#include <iostream>
#include <cmath>
#include <chrono>
#include "numcpp/numcpp.h"

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  NumCpp - High-Performance Numerical Library" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "1. Vector Operations" << std::endl;
    std::cout << "---------------------" << std::endl;
    numcpp::Vector a = numcpp::Vector::linspace(0.0, 1.0, 5);
    std::cout << "Vector a (linspace): " << a << std::endl;
    std::cout << "Norm of a: " << a.norm() << std::endl;
    std::cout << std::endl;
    
    std::cout << "2. Matrix Operations" << std::endl;
    std::cout << "---------------------" << std::endl;
    numcpp::Matrix A = {{4.0, 1.0, 1.0},
                         {1.0, 3.0, 0.0},
                         {1.0, 0.0, 2.0}};
    std::cout << "Matrix A:" << std::endl << A << std::endl;
    std::cout << "A is symmetric: " << (A.isSymmetric() ? "yes" : "no") << std::endl;
    std::cout << "Trace of A: " << A.trace() << std::endl;
    std::cout << "Determinant of A: " << A.determinant() << std::endl;
    std::cout << std::endl;
    
    std::cout << "3. Linear System Solver" << std::endl;
    std::cout << "-------------------------" << std::endl;
    numcpp::Vector b({6.0, 5.0, 4.0});
    numcpp::Vector x = numcpp::solveLU(A, b);
    std::cout << "Solve Ax = b, b = " << b << std::endl;
    std::cout << "Solution x = " << x << std::endl;
    std::cout << "Verification Ax = " << A * x << std::endl;
    std::cout << std::endl;
    
    std::cout << "4. Eigenvalue Computation" << std::endl;
    std::cout << "---------------------------" << std::endl;
    auto eigen_result = numcpp::eigenvalues(A, numcpp::EigenMethod::Jacobi);
    std::cout << "Eigenvalues of A: " << eigen_result.eigenvalues << std::endl;
    std::cout << std::endl;
    
    std::cout << "5. Fast Fourier Transform" << std::endl;
    std::cout << "--------------------------" << std::endl;
    numcpp::Vector signal(8);
    for (size_t i = 0; i < 8; ++i) {
        signal[i] = std::sin(2.0 * numcpp::PI * i / 8.0);
    }
    std::cout << "Input signal: " << signal << std::endl;
    auto spectrum = numcpp::fft(signal);
    auto mag = numcpp::magnitude(spectrum);
    std::cout << "FFT magnitude: [";
    for (size_t i = 0; i < mag.size(); ++i) {
        std::cout << mag[i];
        if (i < mag.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "6. Nonlinear Root Finding" << std::endl;
    std::cout << "--------------------------" << std::endl;
    auto f = [](double x) { return x * x - 2.0; };
    auto root = numcpp::newtonMethod(f, 1.0, 1e-10, 100);
    std::cout << "Root of x^2 - 2 = 0: " << root.root << std::endl;
    std::cout << "Converged: " << (root.converged ? "yes" : "no") << std::endl;
    std::cout << "Iterations: " << root.iterations << std::endl;
    std::cout << "sqrt(2) reference: " << std::sqrt(2.0) << std::endl;
    std::cout << std::endl;
    
    std::cout << "7. Numerical Integration" << std::endl;
    std::cout << "-------------------------" << std::endl;
    auto g = [](double x) { return x * x; };
    double integral = numcpp::integrateSimpson(g, 0.0, 1.0, 1000);
    std::cout << "Integral of x^2 from 0 to 1: " << integral << std::endl;
    std::cout << "Analytical result: " << 1.0 / 3.0 << std::endl;
    
    double integral_romberg = numcpp::integrateRomberg(g, 0.0, 1.0);
    std::cout << "Romberg integral: " << integral_romberg << std::endl;
    
    double integral_gauss = numcpp::integrateGaussLegendre(g, 0.0, 1.0, 10);
    std::cout << "Gauss-Legendre integral (10 nodes): " << integral_gauss << std::endl;
    std::cout << std::endl;
    
    std::cout << "8. Parallel Computing" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Number of threads: " << numcpp::getNumThreads() << std::endl;
    
    size_t n = 100;
    numcpp::Matrix M1 = numcpp::Matrix::random(n, n, 0.0, 1.0);
    numcpp::Matrix M2 = numcpp::Matrix::random(n, n, 0.0, 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    numcpp::Matrix M3 = M1 * M2;
    auto end = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    numcpp::Matrix M4 = numcpp::matMulParallel(M1, M2);
    end = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(end - start).count();
    
    std::cout << "Matrix multiplication (" << n << "x" << n << "):" << std::endl;
    std::cout << "  Serial time:   " << serial_time << "s" << std::endl;
    std::cout << "  Parallel time: " << parallel_time << "s" << std::endl;
    std::cout << "  Results match: " << (M3 == M4 ? "yes" : "no") << std::endl;
    std::cout << std::endl;
    
    std::cout << "=========================================" << std::endl;
    std::cout << "  All demos completed successfully!" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    return 0;
}
