#ifndef NZD_INITIALIZER_H
#define NZD_INITIALIZER_H

#include "Initializers/IWeightInitializer.h"
#include <memory>

enum class InitType : uint8_t {
    HE,
    XAVIER
};

using InitFunc = std::shared_ptr<IWeightInitializer>;

InitFunc resolve_init(InitType f, uint32_t seed = 1);

#endif
