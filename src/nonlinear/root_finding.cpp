#include "numcpp/nonlinear/root_finding.h"
#include "numcpp/linalg/solver.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace numcpp {

RootResult bisection(ScalarFunction f, real_t a, real_t b, real_t tol, size_t maxIter) {
    RootResult result;
    result.converged = false;
    result.iterations = 0;
    
    real_t fa = f(a);
    real_t fb = f(b);
    
    if (fa * fb > 0) {
        throw std::invalid_argument("Function must have opposite signs at interval endpoints");
    }
    
    if (std::abs(fa) < tol) {
        result.root = a;
        result.residual = std::abs(fa);
        result.converged = true;
        return result;
    }
    if (std::abs(fb) < tol) {
        result.root = b;
        result.residual = std::abs(fb);
        result.converged = true;
        return result;
    }
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        real_t c = (a + b) / 2.0;
        real_t fc = f(c);
        
        if (std::abs(fc) < tol || (b - a) / 2.0 < tol) {
            result.root = c;
            result.residual = std::abs(fc);
            result.converged = true;
            return result;
        }
        
        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    result.root = (a + b) / 2.0;
    result.residual = std::abs(f(result.root));
    return result;
}

RootResult newtonMethod(ScalarFunction f, ScalarFunction df, real_t x0, real_t tol, size_t maxIter) {
    RootResult result;
    result.converged = false;
    result.root = x0;
    result.iterations = 0;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        real_t fx = f(result.root);
        real_t dfx = df(result.root);
        
        if (std::abs(dfx) < EPS) {
            result.residual = std::abs(fx);
            return result;
        }
        
        real_t dx = fx / dfx;
        result.root -= dx;
        
        if (std::abs(dx) < tol || std::abs(fx) < tol) {
            result.converged = true;
            result.residual = std::abs(f(result.root));
            return result;
        }
    }
    
    result.residual = std::abs(f(result.root));
    return result;
}

RootResult newtonMethod(ScalarFunction f, real_t x0, real_t tol, size_t maxIter, real_t h) {
    auto df = [f, h](real_t x) {
        return (f(x + h) - f(x - h)) / (2.0 * h);
    };
    return newtonMethod(f, df, x0, tol, maxIter);
}

RootResult secantMethod(ScalarFunction f, real_t x0, real_t x1, real_t tol, size_t maxIter) {
    RootResult result;
    result.converged = false;
    result.iterations = 0;
    
    real_t x_prev = x0;
    real_t x_curr = x1;
    real_t f_prev = f(x_prev);
    real_t f_curr = f(x_curr);
    
    if (std::abs(f_prev) < tol) {
        result.root = x_prev;
        result.residual = std::abs(f_prev);
        result.converged = true;
        return result;
    }
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        
        if (std::abs(f_curr - f_prev) < EPS) {
            break;
        }
        
        real_t x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
        real_t f_next = f(x_next);
        
        if (std::abs(x_next - x_curr) < tol || std::abs(f_next) < tol) {
            result.root = x_next;
            result.residual = std::abs(f_next);
            result.converged = true;
            return result;
        }
        
        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
        f_curr = f_next;
    }
    
    result.root = x_curr;
    result.residual = std::abs(f_curr);
    return result;
}

RootResult fixedPointIteration(ScalarFunction g, real_t x0, real_t tol, size_t maxIter) {
    RootResult result;
    result.converged = false;
    result.root = x0;
    result.iterations = 0;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        real_t x_new = g(result.root);
        
        if (std::abs(x_new - result.root) < tol) {
            result.root = x_new;
            result.converged = true;
            result.residual = std::abs(x_new - result.root);
            return result;
        }
        
        result.root = x_new;
    }
    
    result.residual = std::abs(g(result.root) - result.root);
    return result;
}

RootResult brentMethod(ScalarFunction f, real_t a, real_t b, real_t tol, size_t maxIter) {
    RootResult result;
    result.converged = false;
    result.iterations = 0;
    
    real_t fa = f(a);
    real_t fb = f(b);
    
    if (fa * fb > 0) {
        throw std::invalid_argument("Function must have opposite signs at interval endpoints");
    }
    
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    
    real_t c = a;
    real_t fc = fa;
    bool mflag = true;
    real_t s = b;
    real_t d = 0.0;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        
        if (std::abs(fb) < tol || std::abs(b - a) < tol) {
            result.root = b;
            result.residual = std::abs(fb);
            result.converged = true;
            return result;
        }
        
        if (std::abs(fa - fc) > EPS && std::abs(fb - fc) > EPS) {
            real_t sab = a * fb / (fa - fb);
            real_t sbc = b * fc / (fb - fc);
            real_t sca = c * fa / (fc - fa);
            s = sab + sbc + sca;
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }
        
        bool condition1 = (s < (3 * a + b) / 4.0 || s > b);
        bool condition2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2.0;
        bool condition3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2.0;
        bool condition4 = mflag && std::abs(b - c) < tol;
        bool condition5 = !mflag && std::abs(c - d) < tol;
        
        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            s = (a + b) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        
        real_t fs = f(s);
        d = c;
        c = b;
        fc = fb;
        
        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    
    result.root = b;
    result.residual = std::abs(fb);
    return result;
}

VectorRootResult newtonSystem(VectorFunction F, JacobianFunction J, const Vector& x0, real_t tol, size_t maxIter) {
    VectorRootResult result;
    result.converged = false;
    result.root = x0;
    result.iterations = 0;
    
    for (size_t iter = 0; iter < maxIter; ++iter) {
        result.iterations = iter + 1;
        Vector fx = F(result.root);
        Matrix jac = J(result.root);
        
        Vector delta = solveLU(jac, fx * (-1.0));
        
        result.root = result.root + delta;
        
        if (delta.norm() < tol || fx.norm() < tol) {
            result.converged = true;
            result.residual = F(result.root).norm();
            return result;
        }
    }
    
    result.residual = F(result.root).norm();
    return result;
}

VectorRootResult newtonSystem(VectorFunction F, const Vector& x0, real_t tol, size_t maxIter, real_t h) {
    JacobianFunction J = [F, h](const Vector& x) {
        size_t n = x.size();
        Vector fx = F(x);
        Matrix jac(n, n);
        
        for (size_t j = 0; j < n; ++j) {
            Vector x_perturbed = x;
            x_perturbed[j] += h;
            Vector fx_perturbed = F(x_perturbed);
            
            for (size_t i = 0; i < n; ++i) {
                jac(i, j) = (fx_perturbed[i] - fx[i]) / h;
            }
        }
        
        return jac;
    };
    
    return newtonSystem(F, J, x0, tol, maxIter);
}

}
