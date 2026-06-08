#include "numcpp/sparse/sparse_matrix.h"
#include "test_utils.h"

std::vector<TestCase> get_sparse_tests() {
    return {
        {"SparseMatrix creation", []() {
            numcpp::SparseMatrix m(3, 3);
            return m.rows() == 3 && m.cols() == 3 && m.nnz() == 0;
        }},
        
        {"SparseMatrix insert and access", []() {
            numcpp::SparseMatrix m(3, 3);
            m.insert(0, 0, 1.0);
            m.insert(1, 1, 2.0);
            m.insert(2, 2, 3.0);
            return approx_equal(m(0, 0), 1.0) && 
                   approx_equal(m(1, 1), 2.0) && 
                   approx_equal(m(2, 2), 3.0) && 
                   approx_equal(m(0, 1), 0.0);
        }},
        
        {"SparseMatrix identity", []() {
            auto m = numcpp::SparseMatrix::identity(4);
            return m.rows() == 4 && m.cols() == 4 && 
                   m.nnz() == 4 && 
                   approx_equal(m(0, 0), 1.0) &&
                   approx_equal(m(2, 2), 1.0);
        }},
        
        {"SparseMatrix diagonal", []() {
            numcpp::Vector diag({1.0, 2.0, 3.0, 4.0});
            auto m = numcpp::SparseMatrix::diagonal(diag);
            return m.nnz() == 4 && 
                   approx_equal(m(1, 1), 2.0) &&
                   approx_equal(m(3, 3), 4.0);
        }},
        
        {"SparseMatrix toDense", []() {
            auto sparse = numcpp::SparseMatrix::identity(3);
            auto dense = sparse.toDense();
            return dense.rows() == 3 && dense.cols() == 3 &&
                   approx_equal(dense(0, 0), 1.0) &&
                   approx_equal(dense(1, 1), 1.0) &&
                   approx_equal(dense(0, 1), 0.0);
        }},
        
        {"SparseMatrix fromDense", []() {
            numcpp::Matrix dense(3, 3, 0.0);
            dense(0, 0) = 1.0;
            dense(1, 1) = 2.0;
            dense(2, 2) = 3.0;
            numcpp::SparseMatrix sparse(dense);
            return sparse.nnz() == 3 &&
                   approx_equal(sparse(0, 0), 1.0) &&
                   approx_equal(sparse(2, 2), 3.0);
        }},
        
        {"SparseMatrix addition", []() {
            auto A = numcpp::SparseMatrix::identity(3);
            auto B = numcpp::SparseMatrix::identity(3);
            auto C = A + B;
            return C.nnz() == 3 && approx_equal(C(1, 1), 2.0);
        }},
        
        {"SparseMatrix scalar multiplication", []() {
            auto A = numcpp::SparseMatrix::identity(3);
            auto B = A * 3.0;
            return approx_equal(B(0, 0), 3.0) && approx_equal(B(2, 2), 3.0);
        }},
        
        {"SparseMatrix transpose", []() {
            numcpp::SparseMatrix m(2, 3);
            m.insert(0, 0, 1.0);
            m.insert(0, 2, 3.0);
            m.insert(1, 1, 2.0);
            auto mt = m.transpose();
            return mt.rows() == 3 && mt.cols() == 2 &&
                   approx_equal(mt(0, 0), 1.0) &&
                   approx_equal(mt(2, 0), 3.0) &&
                   approx_equal(mt(1, 1), 2.0);
        }},
        
        {"SparseMatrix vector multiplication", []() {
            auto A = numcpp::SparseMatrix::identity(3);
            numcpp::Vector v({1.0, 2.0, 3.0});
            numcpp::Vector result = A * v;
            return approx_equal(result[0], 1.0) &&
                   approx_equal(result[1], 2.0) &&
                   approx_equal(result[2], 3.0);
        }},
        
        {"SparseMatrix spmv", []() {
            auto A = numcpp::SparseMatrix::identity(3);
            numcpp::Vector v({2.0, 4.0, 6.0});
            numcpp::Vector result = numcpp::spmv(A, v);
            return approx_equal(result[0], 2.0) &&
                   approx_equal(result[1], 4.0) &&
                   approx_equal(result[2], 6.0);
        }},
        
        {"SparseMatrix trace", []() {
            numcpp::SparseMatrix m(3, 3);
            m.insert(0, 0, 1.0);
            m.insert(1, 1, 2.0);
            m.insert(2, 2, 3.0);
            return approx_equal(m.trace(), 6.0);
        }},
    };
}
