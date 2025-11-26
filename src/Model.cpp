#include "Model.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h" // Add this

Model::Model(size_t num_of_layers) {
    _nol = num_of_layers;
}

void Model::init() {
    unsigned int t = 14;
    // 테스트
    const size_t input_dim = 784;
    const size_t output_dim = 10;
    const size_t hidden_dim = 128;

    size_t last_dim = input_dim;

    for (size_t i = 0; i < _nol; ++i) {
        _layers.push_back(std::make_unique<DenseLayer>(t, &Act::relu, last_dim, hidden_dim, i,
            std::make_unique<HeInitializer>())); // Pass initializer
        last_dim = hidden_dim;
    }

    // Output layer
    _layers.push_back(std::make_unique<DenseLayer>(t, &Act::softmax, last_dim, output_dim, _nol,
        std::make_unique<HeInitializer>())); // Pass initializer
}


// _nol: only count hidden layers
// 0: input layer
// 1, 2, ..., _nol-1: hidden layer
// _nol: output layer