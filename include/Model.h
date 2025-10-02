#ifndef MODEL_H
#define MODEL_H

#include "DenseLayer.h"

class Model {
public:
    Model(size_t num_of_layers);
    void init();

private:
    size_t _nol;
    // std::vector<std::unique_ptr<DenseLayer>> _layers;

};

#endif