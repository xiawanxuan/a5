#include "numcpp/core/vector.h"
#include <cmath>
#include <random>
#include <iomanip>
#include <stdexcept>

namespace numcpp {

Vector::Vector() : data_() {}

Vector::Vector(size_t n) : data_(n, 0.0) {}

Vector::Vector(size_t n, real_t value) : data_(n, value) {}

Vector::Vector(const std::vector<real_t>& data) : data_(data) {}

Vector::Vector(std::initializer_list<real_t> values) : data_(values) {}

size_t Vector::size() const { return data_.size(); }

bool Vector::empty() const { return data_.empty(); }

void Vector::resize(size_t n) { data_.resize(n, 0.0); }

void Vector::fill(real_t value) { std::fill(data_.begin(), data_.end(), value); }

void Vector::clear() { data_.clear(); }

real_t& Vector::operator()(size_t i) {
    if (i >= data_.size()) throw std::out_of_range("Vector index out of range");
    return data_[i];
}

const real_t& Vector::operator()(size_t i) const {
    if (i >= data_.size()) throw std::out_of_range("Vector index out of range");
    return data_[i];
}

real_t& Vector::operator[](size_t i) { return data_[i]; }

const real_t& Vector::operator[](size_t i) const { return data_[i]; }

Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size())
        throw std::invalid_argument("Vector sizes must match for addition");
    Vector result(size());
    for (size_t i = 0; i < size(); ++i)
        result[i] = data_[i] + other[i];
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size())
        throw std::invalid_argument("Vector sizes must match for subtraction");
    Vector result(size());
    for (size_t i = 0; i < size(); ++i)
        result[i] = data_[i] - other[i];
    return result;
}

Vector Vector::operator*(real_t scalar) const {
    Vector result(size());
    for (size_t i = 0; i < size(); ++i)
        result[i] = data_[i] * scalar;
    return result;
}

Vector Vector::operator/(real_t scalar) const {
    if (std::abs(scalar) < EPS)
        throw std::invalid_argument("Division by zero");
    Vector result(size());
    for (size_t i = 0; i < size(); ++i)
        result[i] = data_[i] / scalar;
    return result;
}

Vector& Vector::operator+=(const Vector& other) {
    if (size() != other.size())
        throw std::invalid_argument("Vector sizes must match for addition");
    for (size_t i = 0; i < size(); ++i)
        data_[i] += other[i];
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (size() != other.size())
        throw std::invalid_argument("Vector sizes must match for subtraction");
    for (size_t i = 0; i < size(); ++i)
        data_[i] -= other[i];
    return *this;
}

Vector& Vector::operator*=(real_t scalar) {
    for (auto& x : data_) x *= scalar;
    return *this;
}

Vector& Vector::operator/=(real_t scalar) {
    if (std::abs(scalar) < EPS)
        throw std::invalid_argument("Division by zero");
    for (auto& x : data_) x /= scalar;
    return *this;
}

bool Vector::operator==(const Vector& other) const {
    if (size() != other.size()) return false;
    for (size_t i = 0; i < size(); ++i)
        if (std::abs(data_[i] - other[i]) > EPS) return false;
    return true;
}

bool Vector::operator!=(const Vector& other) const {
    return !(*this == other);
}

real_t Vector::norm() const { return std::sqrt(norm2()); }

real_t Vector::norm2() const {
    real_t sum = 0.0;
    for (auto x : data_) sum += x * x;
    return sum;
}

real_t Vector::normInf() const {
    real_t maxVal = 0.0;
    for (auto x : data_) {
        real_t absX = std::abs(x);
        if (absX > maxVal) maxVal = absX;
    }
    return maxVal;
}

real_t Vector::dot(const Vector& other) const {
    if (size() != other.size())
        throw std::invalid_argument("Vector sizes must match for dot product");
    real_t sum = 0.0;
    for (size_t i = 0; i < size(); ++i)
        sum += data_[i] * other[i];
    return sum;
}

Vector Vector::normalize() const {
    real_t n = norm();
    if (n < EPS) return *this;
    return *this / n;
}

Vector Vector::zeros(size_t n) { return Vector(n, 0.0); }

Vector Vector::ones(size_t n) { return Vector(n, 1.0); }

Vector Vector::linspace(real_t start, real_t end, size_t n) {
    if (n == 0) return Vector();
    if (n == 1) return Vector(1, start);
    Vector result(n);
    real_t step = (end - start) / (n - 1);
    for (size_t i = 0; i < n; ++i)
        result[i] = start + i * step;
    return result;
}

Vector Vector::random(size_t n, real_t min, real_t max) {
    Vector result(n);
    static std::mt19937 gen(42);
    std::uniform_real_distribution<real_t> dist(min, max);
    for (size_t i = 0; i < n; ++i)
        result[i] = dist(gen);
    return result;
}

const std::vector<real_t>& Vector::data() const { return data_; }

std::vector<real_t>& Vector::data() { return data_; }

std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << std::setw(10) << std::setprecision(6) << v[i];
        if (i < v.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

Vector operator*(real_t scalar, const Vector& v) { return v * scalar; }

}
