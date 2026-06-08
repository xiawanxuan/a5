set(TEST_SOURCES
    test_main.cpp
    test_vector.cpp
    test_matrix.cpp
    test_solver.cpp
    test_eigenvalue.cpp
    test_fft.cpp
    test_root_finding.cpp
    test_calculus.cpp
    test_parallel.cpp
    test_sparse.cpp
    test_ode.cpp
    test_pde.cpp
)

add_executable(numcpp_tests ${TEST_SOURCES})

target_link_libraries(numcpp_tests PRIVATE numcpp)

add_test(NAME numcpp_tests COMMAND numcpp_tests)
