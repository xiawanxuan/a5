#include "numcpp/core/vector.h"
#include "test_utils.h"

std::vector<TestCase> get_vector_tests() {
    return {
        {"Vector creation and size", []() {
            numcpp::Vector v(5);
            return v.size() == 5;
        }},
        {"Vector zeros", []() {
            numcpp::Vector v = numcpp::Vector::zeros(3);
            return v.size() == 3 && v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0;
        }},
        {"Vector ones", []() {
            numcpp::Vector v = numcpp::Vector::ones(3);
            return v[0] == 1.0 && v[1] == 1.0 && v[2] == 1.0;
        }},
        {"Vector linspace", []() {
            numcpp::Vector v = numcpp::Vector::linspace(0.0, 10.0, 11);
            return v.size() == 11 && approx_equal(v[0], 0.0) && approx_equal(v[10], 10.0);
        }},
        {"Vector addition", []() {
            numcpp::Vector a({1.0, 2.0, 3.0});
            numcpp::Vector b({4.0, 5.0, 6.0});
            numcpp::Vector c = a + b;
            return approx_equal(c[0], 5.0) && approx_equal(c[1], 7.0) && approx_equal(c[2], 9.0);
        }},
        {"Vector subtraction", []() {
            numcpp::Vector a({4.0, 5.0, 6.0});
            numcpp::Vector b({1.0, 2.0, 3.0});
            numcpp::Vector c = a - b;
            return approx_equal(c[0], 3.0) && approx_equal(c[1], 3.0) && approx_equal(c[2], 3.0);
        }},
        {"Vector scalar multiplication", []() {
            numcpp::Vector a({1.0, 2.0, 3.0});
            numcpp::Vector b = a * 2.0;
            return approx_equal(b[0], 2.0) && approx_equal(b[1], 4.0) && approx_equal(b[2], 6.0);
        }},
        {"Vector dot product", []() {
            numcpp::Vector a({1.0, 2.0, 3.0});
            numcpp::Vector b({4.0, 5.0, 6.0});
            double dot = a.dot(b);
            return approx_equal(dot, 32.0);
        }},
        {"Vector norm", []() {
            numcpp::Vector a({3.0, 4.0});
            double norm = a.norm();
            return approx_equal(norm, 5.0);
        }},
        {"Vector normalize", []() {
            numcpp::Vector a({3.0, 4.0});
            numcpp::Vector n = a.normalize();
            return approx_equal(n.norm(), 1.0);
        }},
        {"Vector equality", []() {
            numcpp::Vector a({1.0, 2.0, 3.0});
            numcpp::Vector b({1.0, 2.0, 3.0});
            return a == b;
        }},
        {"Vector inequality", []() {
            numcpp::Vector a({1.0, 2.0, 3.0});
            numcpp::Vector b({1.0, 2.0, 4.0});
            return a != b;
        }},
    };
}
