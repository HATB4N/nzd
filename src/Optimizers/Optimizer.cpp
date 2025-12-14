#include "Optimizers/Optimizer.h"
#include "Optimizers/SGD.h"
#include "Optimizers/Adam.h"

OptFunc resolve_opt(OptType f,
                    Matrix_T<fp32> &W,
                    Matrix_T<fp32> &b,
                    Matrix_T<fp32> &dW,
                    Matrix_T<fp32> &db) {
    switch (f) {
        case OptType::SGD:
            return std::make_unique<SGD>(W, b, dW, db);
        case OptType::ADAM:
            return std::make_unique<Adam>(W, b, dW, db);
        default:
            return nullptr; // fallback -> req nullptr exception (should I?)
    }
}