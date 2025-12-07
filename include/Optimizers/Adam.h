#ifndef ADAM_H
#define ADAM_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include "Common/Struct.h"
#include <vector>
#include <memory>

class DenseLayer;

class Adam : public IOptimizer {
public:
    Adam(std::vector<std::unique_ptr<DenseLayer>>& layers, 
         fp32 learning_rate = 0.001f, 
         fp32 beta1 = 0.9f, 
         fp32 beta2 = 0.999f, 
         fp32 epsilon = 1e-8f);

    void step() override;

private:
    void init();
    fp32 _lr;
    fp32 _beta1;
    fp32 _beta2;
    fp32 _epsilon;
    int _timestep;

    std::vector<Matrix_T<fp32>> _w_m;
    std::vector<Matrix_T<fp32>> _w_v;
    std::vector<Matrix_T<fp32>> _b_m;
    std::vector<Matrix_T<fp32>> _b_v;
    bool _initialized = false;
};

#endif // ADAM_H
