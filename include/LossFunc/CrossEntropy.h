#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include "LossFunc/ILoss.h"

class CrossEntropy : public ILoss {
public:
    CrossEntropy() = default;

    fp32 calculate(const Matrix_T<fp32> &y_pred, const Matrix_T<fp32> &y_true) override;

    void backward(const Matrix_T<fp32> &y_pred, const Matrix_T<fp32> &y_true, Matrix_T<fp32> &gradient) override;
};

#endif // CROSSENTROPY_H
