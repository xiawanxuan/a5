#include <iostream>
#include <vector>
#include "test_utils.h"

extern std::vector<TestCase> get_vector_tests();
extern std::vector<TestCase> get_matrix_tests();
extern std::vector<TestCase> get_solver_tests();
extern std::vector<TestCase> get_eigenvalue_tests();
extern std::vector<TestCase> get_fft_tests();
extern std::vector<TestCase> get_root_finding_tests();
extern std::vector<TestCase> get_calculus_tests();
extern std::vector<TestCase> get_parallel_tests();

int main() {
    std::vector<TestCase> all_tests;
    
    auto add_tests = [&](const std::vector<TestCase>& tests) {
        all_tests.insert(all_tests.end(), tests.begin(), tests.end());
    };
    
    add_tests(get_vector_tests());
    add_tests(get_matrix_tests());
    add_tests(get_solver_tests());
    add_tests(get_eigenvalue_tests());
    add_tests(get_fft_tests());
    add_tests(get_root_finding_tests());
    add_tests(get_calculus_tests());
    add_tests(get_parallel_tests());
    
    int passed = 0;
    int failed = 0;
    
    for (const auto& test : all_tests) {
        std::cout << "[TEST] " << test.name << " ... ";
        try {
            if (test.func()) {
                std::cout << "PASSED" << std::endl;
                passed++;
            } else {
                std::cout << "FAILED" << std::endl;
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "FAILED (exception: " << e.what() << ")" << std::endl;
            failed++;
        }
    }
    
    std::cout << "\n==========================" << std::endl;
    std::cout << "Total: " << all_tests.size() << " tests" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "==========================" << std::endl;
    
    return failed == 0 ? 0 : 1;
}
