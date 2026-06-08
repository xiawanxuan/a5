#include "numcpp/linalg/solver.h"
#include "test_utils.h"
#include <iostream>

std::vector<TestCase> get_solver_tests() {
    return {
        {"LU decomposition and solve", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 2.0; A(0, 1) = -1.0; A(0, 2) = 0.0;
            A(1, 0) = -1.0; A(1, 1) = 2.0; A(1, 2) = -1.0;
            A(2, 0) = 0.0; A(2, 1) = -1.0; A(2, 2) = 2.0;
            numcpp::Vector b({1.0, 0.0, 1.0});
            numcpp::Vector x = numcpp::solveLU(A, b);
            numcpp::Vector Ax = A * x;
            double err = (Ax - b).norm();
            return err < 1e-8;
        }},
        {"Gaussian elimination", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
            A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
            A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 10.0;
            numcpp::Vector b({6.0, 15.0, 25.0});
            numcpp::Vector x = numcpp::solveGaussElimination(A, b);
            numcpp::Vector Ax = A * x;
            double err = (Ax - b).norm();
            return err < 1e-8;
        }},
        {"Jacobi method", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 10.0; A(0, 1) = -1.0; A(0, 2) = 2.0;
            A(1, 0) = -1.0; A(1, 1) = 11.0; A(1, 2) = -1.0;
            A(2, 0) = 2.0; A(2, 1) = -1.0; A(2, 2) = 10.0;
            numcpp::Vector b({6.0, 25.0, -11.0});
            numcpp::Vector x = numcpp::solveJacobi(A, b, 1e-10, 100);
            numcpp::Vector Ax = A * x;
            double err = (Ax - b).norm() / b.norm();
            return err < 1e-6;
        }},
        {"Gauss-Seidel method", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 10.0; A(0, 1) = -1.0; A(0, 2) = 2.0;
            A(1, 0) = -1.0; A(1, 1) = 11.0; A(1, 2) = -1.0;
            A(2, 0) = 2.0; A(2, 1) = -1.0; A(2, 2) = 10.0;
            numcpp::Vector b({6.0, 25.0, -11.0});
            numcpp::Vector x = numcpp::solveGaussSeidel(A, b, 1e-10, 100);
            numcpp::Vector Ax = A * x;
            double err = (Ax - b).norm() / b.norm();
            return err < 1e-6;
        }},
        {"Conjugate gradient method", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
            A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 0.0;
            A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 2.0;
            numcpp::Vector b({3.0, 5.0, 4.0});
            numcpp::Vector x = numcpp::solveConjugateGradient(A, b, 1e-8, 100);
            numcpp::Vector Ax = A * x;
            double err = (Ax - b).norm() / b.norm();
            return err < 1e-6;
        }},
        {"Multiple RHS solve", []() {
            numcpp::Matrix A(2, 2);
            A(0, 0) = 1.0; A(0, 1) = 2.0;
            A(1, 0) = 3.0; A(1, 1) = 4.0;
            numcpp::Matrix B(2, 2);
            B(0, 0) = 5.0; B(0, 1) = 7.0;
            B(1, 0) = 11.0; B(1, 1) = 15.0;
            numcpp::Matrix X = numcpp::solveMultiple(A, B);
            numcpp::Matrix AX = A * X;
            double err = 0.0;
            for (size_t i = 0; i < 2; ++i)
                for (size_t j = 0; j < 2; ++j)
                    err += (AX(i, j) - B(i, j)) * (AX(i, j) - B(i, j));
            return std::sqrt(err) < 1e-8;
        }},
    };
}
