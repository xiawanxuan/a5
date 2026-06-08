#include "numcpp/core/matrix.h"
#include "test_utils.h"

std::vector<TestCase> get_matrix_tests() {
    return {
        {"Matrix creation", []() {
            numcpp::Matrix m(3, 3);
            return m.rows() == 3 && m.cols() == 3;
        }},
        {"Matrix zeros", []() {
            numcpp::Matrix m = numcpp::Matrix::zeros(2, 3);
            return m.rows() == 2 && m.cols() == 3 && m(0, 0) == 0.0;
        }},
        {"Matrix identity", []() {
            numcpp::Matrix m = numcpp::Matrix::identity(3);
            return m(0, 0) == 1.0 && m(1, 1) == 1.0 && m(2, 2) == 1.0 && m(0, 1) == 0.0;
        }},
        {"Matrix addition", []() {
            numcpp::Matrix A = numcpp::Matrix::ones(2, 2);
            numcpp::Matrix B = numcpp::Matrix::ones(2, 2);
            numcpp::Matrix C = A + B;
            return approx_equal(C(0, 0), 2.0) && approx_equal(C(1, 1), 2.0);
        }},
        {"Matrix scalar multiplication", []() {
            numcpp::Matrix A = numcpp::Matrix::ones(2, 2);
            numcpp::Matrix B = A * 3.0;
            return approx_equal(B(0, 0), 3.0) && approx_equal(B(1, 1), 3.0);
        }},
        {"Matrix vector multiplication", []() {
            numcpp::Matrix A = numcpp::Matrix::identity(3);
            numcpp::Vector v({1.0, 2.0, 3.0});
            numcpp::Vector result = A * v;
            return approx_equal(result[0], 1.0) && approx_equal(result[1], 2.0) && approx_equal(result[2], 3.0);
        }},
        {"Matrix multiplication", []() {
            numcpp::Matrix A = numcpp::Matrix::identity(2);
            numcpp::Matrix B(2, 2);
            B(0, 0) = 1.0; B(0, 1) = 2.0;
            B(1, 0) = 3.0; B(1, 1) = 4.0;
            numcpp::Matrix C = A * B;
            return approx_equal(C(0, 0), 1.0) && approx_equal(C(0, 1), 2.0) &&
                   approx_equal(C(1, 0), 3.0) && approx_equal(C(1, 1), 4.0);
        }},
        {"Matrix transpose", []() {
            numcpp::Matrix A(2, 3);
            A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
            A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
            numcpp::Matrix AT = A.transpose();
            return AT.rows() == 3 && AT.cols() == 2 &&
                   approx_equal(AT(0, 0), 1.0) && approx_equal(AT(2, 1), 6.0);
        }},
        {"Matrix trace", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 1.0; A(1, 1) = 2.0; A(2, 2) = 3.0;
            return approx_equal(A.trace(), 6.0);
        }},
        {"Matrix determinant 2x2", []() {
            numcpp::Matrix A(2, 2);
            A(0, 0) = 1.0; A(0, 1) = 2.0;
            A(1, 0) = 3.0; A(1, 1) = 4.0;
            return approx_equal(A.determinant(), -2.0);
        }},
        {"Matrix isSquare", []() {
            numcpp::Matrix A(3, 3);
            numcpp::Matrix B(2, 3);
            return A.isSquare() && !B.isSquare();
        }},
        {"Matrix row and column access", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
            A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
            A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
            numcpp::Vector row0 = A.row(0);
            numcpp::Vector col2 = A.col(2);
            return approx_equal(row0[0], 1.0) && approx_equal(row0[2], 3.0) &&
                   approx_equal(col2[0], 3.0) && approx_equal(col2[2], 9.0);
        }},
    };
}
