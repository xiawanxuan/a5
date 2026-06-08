#include "numcpp/fft/fft.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef NUMPCPP_HAVE_THREADS
#include <thread>
#include <vector>
#endif

namespace numcpp {

namespace {

void bitReversalPermutation(std::vector<complex_t>& data) {
    size_t n = data.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }
}

void fftIterative(std::vector<complex_t>& data, FFTDirection direction) {
    size_t n = data.size();
    if (n == 0) return;
    
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("FFT input size must be a power of 2");
    }
    
    bitReversalPermutation(data);
    
    real_t sign = (direction == FFTDirection::Forward) ? -1.0 : 1.0;
    
    for (size_t len = 2; len <= n; len <<= 1) {
        real_t ang = 2.0 * PI / len * sign;
        complex_t wlen = std::polar(1.0, ang);
        
        for (size_t i = 0; i < n; i += len) {
            complex_t w = 1.0;
            for (size_t j = 0; j < len / 2; ++j) {
                complex_t u = data[i + j];
                complex_t v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    
    if (direction == FFTDirection::Inverse) {
        for (auto& x : data) {
            x /= static_cast<real_t>(n);
        }
    }
}

}

std::vector<complex_t> fft(const std::vector<complex_t>& input, FFTDirection direction) {
    size_t n = input.size();
    if (n == 0) return input;
    
    size_t powerOf2 = nextPowerOf2(n);
    std::vector<complex_t> data(powerOf2, 0.0);
    for (size_t i = 0; i < n; ++i) {
        data[i] = input[i];
    }
    
    fftIterative(data, direction);
    
    return data;
}

std::vector<complex_t> fft(const std::vector<complex_t>& input) {
    return fft(input, FFTDirection::Forward);
}

std::vector<complex_t> fft(const Vector& input) {
    std::vector<complex_t> complexInput(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        complexInput[i] = complex_t(input[i], 0.0);
    }
    return fft(complexInput, FFTDirection::Forward);
}

std::vector<complex_t> ifft(const std::vector<complex_t>& input) {
    return fft(input, FFTDirection::Inverse);
}

Vector ifftReal(const std::vector<complex_t>& input) {
    std::vector<complex_t> result = ifft(input);
    Vector realResult(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        realResult[i] = result[i].real();
    }
    return realResult;
}

Matrix fft2(const Matrix& input) {
    size_t rows = input.rows();
    size_t cols = input.cols();
    
    size_t padded_rows = nextPowerOf2(rows);
    size_t padded_cols = nextPowerOf2(cols);
    
    std::vector<std::vector<complex_t>> complexData(padded_rows, std::vector<complex_t>(padded_cols, 0.0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            complexData[i][j] = complex_t(input(i, j), 0.0);
        }
    }
    
    for (size_t i = 0; i < padded_rows; ++i) {
        complexData[i] = fft(complexData[i], FFTDirection::Forward);
    }
    
    for (size_t j = 0; j < padded_cols; ++j) {
        std::vector<complex_t> col(padded_rows);
        for (size_t i = 0; i < padded_rows; ++i) {
            col[i] = complexData[i][j];
        }
        col = fft(col, FFTDirection::Forward);
        for (size_t i = 0; i < padded_rows; ++i) {
            complexData[i][j] = col[i];
        }
    }
    
    Matrix result(padded_rows, padded_cols);
    for (size_t i = 0; i < padded_rows; ++i) {
        for (size_t j = 0; j < padded_cols; ++j) {
            result(i, j) = std::abs(complexData[i][j]);
        }
    }
    
    return result;
}

Matrix ifft2(const Matrix& input) {
    size_t rows = input.rows();
    size_t cols = input.cols();
    
    std::vector<std::vector<complex_t>> complexData(rows, std::vector<complex_t>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            complexData[i][j] = complex_t(input(i, j), 0.0);
        }
    }
    
    for (size_t i = 0; i < rows; ++i) {
        complexData[i] = fft(complexData[i], FFTDirection::Inverse);
    }
    
    for (size_t j = 0; j < cols; ++j) {
        std::vector<complex_t> col(rows);
        for (size_t i = 0; i < rows; ++i) {
            col[i] = complexData[i][j];
        }
        col = fft(col, FFTDirection::Inverse);
        for (size_t i = 0; i < rows; ++i) {
            complexData[i][j] = col[i];
        }
    }
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = complexData[i][j].real();
        }
    }
    
    return result;
}

std::vector<complex_t> dft(const std::vector<complex_t>& input, FFTDirection direction) {
    size_t n = input.size();
    std::vector<complex_t> result(n, 0.0);
    
    real_t sign = (direction == FFTDirection::Forward) ? -1.0 : 1.0;
    
    for (size_t k = 0; k < n; ++k) {
        complex_t sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            real_t ang = 2.0 * PI * k * t / n * sign;
            sum += input[t] * std::polar(1.0, ang);
        }
        if (direction == FFTDirection::Inverse) {
            sum /= static_cast<real_t>(n);
        }
        result[k] = sum;
    }
    
    return result;
}

size_t nextPowerOf2(size_t n) {
    if (n == 0) return 1;
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

std::vector<real_t> magnitude(const std::vector<complex_t>& spectrum) {
    std::vector<real_t> mag(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        mag[i] = std::abs(spectrum[i]);
    }
    return mag;
}

std::vector<real_t> phase(const std::vector<complex_t>& spectrum) {
    std::vector<real_t> ph(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
        ph[i] = std::arg(spectrum[i]);
    }
    return ph;
}

}
