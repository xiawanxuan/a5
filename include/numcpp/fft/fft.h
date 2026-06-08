#ifndef NUMPCPP_FFT_FFT_H
#define NUMPCPP_FFT_FFT_H

#include "numcpp/core/types.h"
#include "numcpp/core/vector.h"
#include "numcpp/core/matrix.h"
#include <vector>
#include <complex>

namespace numcpp {

std::vector<complex_t> fft(const std::vector<complex_t>& input);
std::vector<complex_t> fft(const Vector& input);

std::vector<complex_t> ifft(const std::vector<complex_t>& input);
Vector ifftReal(const std::vector<complex_t>& input);

std::vector<complex_t> fft(const std::vector<complex_t>& input, FFTDirection direction);

Matrix fft2(const Matrix& input);
Matrix ifft2(const Matrix& input);

std::vector<complex_t> dft(const std::vector<complex_t>& input, FFTDirection direction = FFTDirection::Forward);

size_t nextPowerOf2(size_t n);

std::vector<real_t> magnitude(const std::vector<complex_t>& spectrum);
std::vector<real_t> phase(const std::vector<complex_t>& spectrum);

}

#endif
