#include "Model.h"

Model::Model(size_t num_of_layers) {
    _nol = num_of_layers;
}

void Model::init() {
    for (size_t i = 0; i< _nol-1; ++i) {
        // _layers.push_back(std::make_unique<DenseLayer>()); // fix args
    }
}


// _nol
// 0: input layer
// 1, 2, ..., _nol-1: hidden layer
// _nol: output layer