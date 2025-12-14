#ifndef NZD_OPTIMIZER_H
#define NZD_OPTIMIZER_H

#include "Optimizers/IOptimizer.h"
#include "Common/Struct.h"
#include "Common/Types.h"
#include <memory>

enum class OptType : uint8_t {
    SGD,
    ADAM,
};

using OptFunc = std::unique_ptr<IOptimizer>;

OptFunc resolve_opt(OptType f,
                    Matrix_T<fp32> &W,
                    Matrix_T<fp32> &b,
                    Matrix_T<fp32> &dW,
                    Matrix_T<fp32> &db);

#endif