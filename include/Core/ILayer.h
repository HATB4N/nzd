#ifndef NZD_ILAYER_H
#define NZD_ILAYER_H

#include <vector>
#include <cstdint>

#include "Common/Struct.h"
#include "Common/Types.h"

class ILayer {
public:
    virtual ~ILayer() = default;

    virtual void forward(const Matrix_T<fp32> &X, Matrix_T<fp32> &Z) = 0;
    virtual void backward(Matrix_T<fp32> &dR, Matrix_T<fp32> &dX) = 0;
    virtual void update() = 0;

    virtual uint64_t get_in_dim() const = 0;
    virtual uint64_t get_out_dim() const = 0;
};

#endif  // NZD_ILAYER_H
