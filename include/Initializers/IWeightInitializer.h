#ifndef IWEIGHTINITIALIZER_H
#define IWEIGHTINITIALIZER_H

#include "../Struct.h"
class IWeightInitializer {
public:
    virtual ~IWeightInitializer() = default;
    virtual void initialize(Matrix_T<fp16>& weights, size_t input_dim, size_t output_dim) const = 0;
    // 난 무려 가상함수를 배웠단 사실
};

#endif
