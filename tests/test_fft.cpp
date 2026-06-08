#include "numcpp/fft/fft.h"
#include "test_utils.h"
#include <complex>
#include <cmath>

std::vector<TestCase> get_fft_tests() {
    return {
        {"FFT size is power of 2", []() {
            std::vector<std::complex<double>> input(8, 1.0);
            auto result = numcpp::fft(input);
            return result.size() == 8;
        }},
        {"FFT of constant signal", []() {
            std::vector<std::complex<double>> input(4, 1.0);
            auto result = numcpp::fft(input);
            return approx_equal(result[0].real(), 4.0) &&
                   approx_equal(result[0].imag(), 0.0);
        }},
        {"FFT and IFFT inverse", []() {
            std::vector<std::complex<double>> input(8);
            for (size_t i = 0; i < 8; ++i) {
                input[i] = std::complex<double>(std::sin(i * 0.5), std::cos(i * 0.3));
            }
            auto fft_result = numcpp::fft(input);
            auto ifft_result = numcpp::ifft(fft_result);
            double err = 0.0;
            for (size_t i = 0; i < 8; ++i) {
                double diff = std::abs(ifft_result[i] - input[i]);
                err += diff * diff;
            }
            return std::sqrt(err) < 1e-10;
        }},
        {"FFT of real vector", []() {
            numcpp::Vector v(8);
            for (size_t i = 0; i < 8; ++i) {
                v[i] = std::sin(i * 0.5);
            }
            auto result = numcpp::fft(v);
            return result.size() == 8;
        }},
        {"nextPowerOf2 function", []() {
            return numcpp::nextPowerOf2(1) == 1 &&
                   numcpp::nextPowerOf2(5) == 8 &&
                   numcpp::nextPowerOf2(8) == 8 &&
                   numcpp::nextPowerOf2(100) == 128;
        }},
        {"DFT matches FFT", []() {
            std::vector<std::complex<double>> input(8);
            for (size_t i = 0; i < 8; ++i) {
                input[i] = std::complex<double>(i * 1.0, 0.0);
            }
            auto fft_result = numcpp::fft(input);
            auto dft_result = numcpp::dft(input);
            double err = 0.0;
            for (size_t i = 0; i < 8; ++i) {
                double diff = std::abs(fft_result[i] - dft_result[i]);
                err += diff * diff;
            }
            return std::sqrt(err) < 1e-10;
        }},
        {"Magnitude and phase", []() {
            std::vector<std::complex<double>> input(4);
            input[0] = std::complex<double>(3.0, 4.0);
            input[1] = std::complex<double>(0.0, 1.0);
            input[2] = std::complex<double>(-1.0, 0.0);
            input[3] = std::complex<double>(1.0, 1.0);
            auto mag = numcpp::magnitude(input);
            auto phase = numcpp::phase(input);
            bool ok = approx_equal(mag[0], 5.0) && approx_equal(mag[1], 1.0);
            ok = ok && approx_equal(mag[2], 1.0);
            return ok;
        }},
        {"2D FFT basic", []() {
            numcpp::Matrix A = numcpp::Matrix::ones(4, 4);
            auto result = numcpp::fft2(A);
            return result.rows() == 4 && result.cols() == 4 &&
                   result(0, 0) > 0.0;
        }},
    };
}
