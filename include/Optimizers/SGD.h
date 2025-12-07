#ifndef SGD_H
#define SGD_H

#include "Optimizers/IOptimizer.h"
#include "Common/Types.h"
#include <vector>
#include <memory>

class DenseLayer;

class SGD : public IOptimizer {
public:
    SGD(std::vector<std::unique_ptr<DenseLayer>>& layers, fp32 learning_rate = 0.01f);
    
    void step() override;

private:
    fp32 _lr;
};

#endif
