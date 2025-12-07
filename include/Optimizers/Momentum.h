#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include "Common/Struct.h"
#include <vector>
#include <memory>

class DenseLayer;

class Momentum : public IOptimizer {
public:
    Momentum(std::vector<std::unique_ptr<DenseLayer>>& layers, fp32 learning_rate = 0.01f, fp32 beta = 0.9f);

    void step() override;

private:
    void init();
    fp32 _lr;
    fp32 _beta;

    std::vector<Matrix_T<fp32>> _w_velocities;
    std::vector<Matrix_T<fp32>> _b_velocities;
    bool _initialized = false;
};

#endif // MOMENTUM_H
