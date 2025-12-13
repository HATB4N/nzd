#include "Optimizers/Optimizer.h"
#include "Optimizers/SGD.h"
#include "Optimizers/Adam.h"

OptFunc resolve_opt(OptType f) {
    switch (f) {
        case OptType::SGD:
            return std::make_unique<SGD>();
        case OptType::ADAM:
            return std::make_unique<Adam>();
        default:
            return nullptr;
    }
}