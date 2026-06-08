#ifndef NUMPCPP_CORE_VECTOR_H
#define NUMPCPP_CORE_VECTOR_H

#include "numcpp/core/types.h"
#include <vector>
#include <initializer_list>
#include <ostream>

namespace numcpp {

class Vector {
public:
    Vector();
    explicit Vector(size_t n);
    Vector(size_t n, real_t value);
    Vector(const std::vector<real_t>& data);
    Vector(std::initializer_list<real_t> values);

    size_t size() const;
    bool empty() const;
    void resize(size_t n);
    void fill(real_t value);
    void clear();

    real_t& operator()(size_t i);
    const real_t& operator()(size_t i) const;
    real_t& operator[](size_t i);
    const real_t& operator[](size_t i) const;

    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector operator*(real_t scalar) const;
    Vector operator/(real_t scalar) const;
    Vector& operator+=(const Vector& other);
    Vector& operator-=(const Vector& other);
    Vector& operator*=(real_t scalar);
    Vector& operator/=(real_t scalar);

    bool operator==(const Vector& other) const;
    bool operator!=(const Vector& other) const;

    real_t norm() const;
    real_t norm2() const;
    real_t normInf() const;
    real_t dot(const Vector& other) const;
    Vector normalize() const;

    static Vector zeros(size_t n);
    static Vector ones(size_t n);
    static Vector linspace(real_t start, real_t end, size_t n);
    static Vector random(size_t n, real_t min = 0.0, real_t max = 1.0);

    const std::vector<real_t>& data() const;
    std::vector<real_t>& data();

    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

private:
    std::vector<real_t> data_;
};

Vector operator*(real_t scalar, const Vector& v);

}

#endif
