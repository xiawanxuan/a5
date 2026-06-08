#include "numcpp/ode/ode_solver.h"
#include "test_utils.h"
#include <cmath>

std::vector<TestCase> get_ode_tests() {
    return {
        {"ODE Euler method - exponential decay", []() {
            auto f = [](real_t t, real_t y) { return -y; };
            real_t result = numcpp::odeEuler(f, 0.0, 1.0, 1.0, 0.001);
            real_t expected = std::exp(-1.0);
            return std::abs(result - expected) < 0.005;
        }},
        
        {"ODE Improved Euler method - exponential decay", []() {
            auto f = [](real_t t, real_t y) { return -y; };
            real_t result = numcpp::odeImprovedEuler(f, 0.0, 1.0, 1.0, 0.01);
            real_t expected = std::exp(-1.0);
            return std::abs(result - expected) < 0.0001;
        }},
        
        {"ODE RK4 method - exponential decay", []() {
            auto f = [](real_t t, real_t y) { return -y; };
            real_t result = numcpp::odeRK4(f, 0.0, 1.0, 1.0, 0.1);
            real_t expected = std::exp(-1.0);
            return std::abs(result - expected) < 1e-6;
        }},
        
        {"ODE RK45 adaptive - exponential decay", []() {
            auto f = [](real_t t, real_t y) { return -y; };
            real_t result = numcpp::odeRK45(f, 0.0, 1.0, 1.0, 1e-8);
            real_t expected = std::exp(-1.0);
            return std::abs(result - expected) < 1e-6;
        }},
        
        {"ODE solve1D - RK4", []() {
            auto f = [](real_t t, real_t y) { return -y; };
            auto result = numcpp::odeSolve1D(f, 0.0, 1.0, 1.0, 0.1, numcpp::ODEMethod::RK4);
            return result.t.size() == result.y.size() &&
                   result.t.size() > 0 &&
                   approx_equal(result.t.front(), 0.0) &&
                   approx_equal(result.t.back(), 1.0) &&
                   std::abs(result.y.back() - std::exp(-1.0)) < 1e-6;
        }},
        
        {"ODE system - Euler method", []() {
            auto f = [](real_t t, const numcpp::Vector& y) {
                numcpp::Vector dy(2);
                dy[0] = y[1];
                dy[1] = -y[0];
                return dy;
            };
            numcpp::Vector y0({1.0, 0.0});
            auto result = numcpp::odeEulerSystem(f, 0.0, y0, 0.01, 0.001);
            return result.size() == 2 &&
                   std::abs(result[0] - 1.0) < 0.001 &&
                   std::abs(result[1] - (-0.01)) < 0.001;
        }},
        
        {"ODE system - RK4 method", []() {
            auto f = [](real_t t, const numcpp::Vector& y) {
                numcpp::Vector dy(2);
                dy[0] = y[1];
                dy[1] = -y[0];
                return dy;
            };
            numcpp::Vector y0({1.0, 0.0});
            auto result = numcpp::odeRK4System(f, 0.0, y0, 1.0, 0.1);
            real_t expected_y0 = std::cos(1.0);
            real_t expected_y1 = -std::sin(1.0);
            return std::abs(result[0] - expected_y0) < 1e-6 &&
                   std::abs(result[1] - expected_y1) < 1e-6;
        }},
        
        {"ODE system - RK45 adaptive", []() {
            auto f = [](real_t t, const numcpp::Vector& y) {
                numcpp::Vector dy(2);
                dy[0] = y[1];
                dy[1] = -y[0];
                return dy;
            };
            numcpp::Vector y0({1.0, 0.0});
            auto result = numcpp::odeRK45System(f, 0.0, y0, 1.0, 1e-8);
            real_t expected_y0 = std::cos(1.0);
            real_t expected_y1 = -std::sin(1.0);
            return std::abs(result[0] - expected_y0) < 1e-6 &&
                   std::abs(result[1] - expected_y1) < 1e-6;
        }},
        
        {"ODE solve - RK4 system", []() {
            auto f = [](real_t t, const numcpp::Vector& y) {
                numcpp::Vector dy(2);
                dy[0] = y[1];
                dy[1] = -y[0];
                return dy;
            };
            numcpp::Vector y0({1.0, 0.0});
            auto result = numcpp::odeSolve(f, 0.0, y0, 0.5, 0.1, numcpp::ODEMethod::RK4);
            return result.t.size() == result.y.size() &&
                   result.t.size() > 0 &&
                   result.y[0].size() == 2;
        }},
        
        {"ODE method enum", []() {
            return static_cast<int>(numcpp::ODEMethod::Euler) == 0 &&
                   static_cast<int>(numcpp::ODEMethod::ImprovedEuler) == 1 &&
                   static_cast<int>(numcpp::ODEMethod::RK4) == 2 &&
                   static_cast<int>(numcpp::ODEMethod::RK45) == 3;
        }},
    };
}
