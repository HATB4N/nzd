#include "Initializers/Initializer.h"
#include "Initializers/HeInitializer.h"
#include "Initializers/XavierInitializer.h"

InitFunc resolve_init(InitType f) {
    switch (f) {
        case InitType::HE: {
            static auto he_instance = std::make_shared<HeInitializer>(1);
            return he_instance;
        }
        case InitType::XAVIER: {
            static auto xavier_instance = std::make_shared<XavierInitializer>(1);
            return xavier_instance;
        }
        default:
            return nullptr;
    }
}