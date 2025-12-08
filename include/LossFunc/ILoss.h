#ifndef ILOSS_H
#define ILOSS_H

#include "Common/Struct.h"
class ILoss {
public:
    virtual ~ILoss() = default;
    virtual void getLoss(Matrix_T<fp16>& wiights, size_t input_dim, size_t output_dim) const = 0;  
};

#endif // ILOSS_H