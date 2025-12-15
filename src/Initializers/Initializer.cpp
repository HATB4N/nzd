#include "Initializers/Initializer.h"
#include "Initializers/HeInitializer.h"
#include "Initializers/XavierInitializer.h"

InitFunc resolve_init(InitType f, uint32_t seed) {
    switch (f) {
        case InitType::HE: {
            static auto he_instance = std::make_shared<HeInitializer>(seed);
            return he_instance;
        }
        case InitType::XAVIER: {
            static auto xavier_instance = std::make_shared<XavierInitializer>(seed);
            return xavier_instance;
        }
        default:
            return nullptr;
    }
}
