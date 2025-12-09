#ifndef ILOSS_H
#define ILOSS_H

#include "Common/Struct.h"
#include "Common/Types.h"

class ILoss {
public:
    virtual ~ILoss() = default;
    virtual fp32 calculate(const Matrix_T<fp32>& y_pred, const Matrix_T<fp32>& y_true) = 0;
    virtual void backward(const Matrix_T<fp32>& y_pred, const Matrix_T<fp32>& y_true, Matrix_T<fp32>& gradient) = 0;
};

#endif // ILOSS_H