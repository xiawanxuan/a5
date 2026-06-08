set(NUMPCPP_SOURCES
    core/vector.cpp
    core/matrix.cpp
    linalg/solver.cpp
    linalg/eigenvalue.cpp
    fft/fft.cpp
    nonlinear/root_finding.cpp
    calculus/calculus.cpp
    parallel/parallel.cpp
)

add_library(numcpp ${NUMPCPP_SOURCES})

target_include_directories(numcpp
    PUBLIC
        $<BUILD_INTERFACE:${NUMPCPP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(numcpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(numcpp PUBLIC pthread)
endif()

install(TARGETS numcpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${NUMPCPP_INCLUDE_DIR}/numcpp
    DESTINATION include
)
