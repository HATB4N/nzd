#ifndef NZD_OPTIMIZER_H
#define NZD_OPTIMIZER_H

#include "Optimizers/IOptimizer.h"
#include <memory>

enum class OptType : uint8_t {
    SGD,
    ADAM
};

using OptFunc = std::unique_ptr<IOptimizer>;

OptFunc resolve_opt(OptType f);

#endif