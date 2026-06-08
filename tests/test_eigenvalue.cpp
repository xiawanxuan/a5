#include "numcpp/linalg/eigenvalue.h"
#include "test_utils.h"
#include <iostream>
#include <cmath>

std::vector<TestCase> get_eigenvalue_tests() {
    return {
        {"Power method for dominant eigenvalue", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
            A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 0.0;
            A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 2.0;
            numcpp::Vector ev;
            double eig = numcpp::powerMethod(A, ev, 1e-8, 1000);
            numcpp::Vector Av = A * ev;
            double rel_err = (Av - eig * ev).norm() / std::abs(eig);
            return rel_err < 1e-6;
        }},
        {"QR decomposition", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 12.0; A(0, 1) = -51.0; A(0, 2) = 4.0;
            A(1, 0) = 6.0; A(1, 1) = 167.0; A(1, 2) = -68.0;
            A(2, 0) = -4.0; A(2, 1) = 24.0; A(2, 2) = -41.0;
            auto qr = numcpp::qrDecompose(A);
            numcpp::Matrix recon = qr.Q * qr.R;
            double err = 0.0;
            for (size_t i = 0; i < 3; ++i)
                for (size_t j = 0; j < 3; ++j)
                    err += (recon(i, j) - A(i, j)) * (recon(i, j) - A(i, j));
            return std::sqrt(err) < 1e-6;
        }},
        {"QR eigenvalues for symmetric matrix", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
            A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 0.0;
            A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 2.0;
            auto result = numcpp::qrEigenvalues(A, 1e-10, 1000);
            bool ok = true;
            for (size_t i = 0; i < 3; ++i) {
                numcpp::Vector ev = result.eigenvectors.col(i);
                numcpp::Vector Av = A * ev;
                numcpp::Vector lambda_v = ev * result.eigenvalues[i];
                double err = (Av - lambda_v).norm();
                if (err > 1e-6) ok = false;
            }
            return ok;
        }},
        {"Jacobi eigenvalues for symmetric matrix", []() {
            numcpp::Matrix A(3, 3);
            A(0, 0) = 4.0; A(0, 1) = 1.0; A(0, 2) = 1.0;
            A(1, 0) = 1.0; A(1, 1) = 3.0; A(1, 2) = 0.0;
            A(2, 0) = 1.0; A(2, 1) = 0.0; A(2, 2) = 2.0;
            auto result = numcpp::jacobiEigenvalues(A, 1e-10, 100);
            bool ok = true;
            for (size_t i = 0; i < 3; ++i) {
                numcpp::Vector ev = result.eigenvectors.col(i);
                numcpp::Vector Av = A * ev;
                numcpp::Vector lambda_v = ev * result.eigenvalues[i];
                double err = (Av - lambda_v).norm();
                if (err > 1e-6) ok = false;
            }
            return ok;
        }},
        {"Eigenvalues sorted in descending order", []() {
            numcpp::Matrix A = numcpp::Matrix::diagonal(numcpp::Vector({1.0, 3.0, 2.0}));
            auto result = numcpp::eigenvalues(A, numcpp::EigenMethod::QR, 1e-10, 500);
            return result.eigenvalues[0] >= result.eigenvalues[1] &&
                   result.eigenvalues[1] >= result.eigenvalues[2];
        }},
    };
}
