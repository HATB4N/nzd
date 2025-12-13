#include "Core/Model.h"
#include "Initializers/HeInitializer.h"
#include "Initializers/XavierInitializer.h"
#include <stdexcept>
#include "Utils/Activation.h" // enum매칭해?
#include "Common/Struct.h"

Model::Model() {}

int Model::init(uint64_t num_of_layers, // denselayer 기준
                uint64_t input_dim,
                uint64_t output_dim,
                uint64_t hidden_dim,
                uint64_t batch_size) {
    _nol = num_of_layers;
    _input_dim = input_dim;
    _output_dim =output_dim;
    _hidden_dim =hidden_dim;
    _batch_size =batch_size;
    unsigned int t = 14;
    auto init = std::make_shared<HeInitializer>(1); // seed injectin req
    uint64_t last_dim = _input_dim;
    // index = (0, _nol-1) | hidden layer | act
    for (uint64_t i = 0; i< _nol; i++) {
        _layers.push_back(
            std::make_unique<DenseLayer>(ActFunc::RELU, 
                                        last_dim, 
                                        _hidden_dim, 
                                        init, 
                                        _layers.size()));
        last_dim = _hidden_dim;
        }

    _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::SOFTMAX,
                                     last_dim, 
                                     _output_dim, 
                                     init, 
                                     _layers.size()));
    
    return 0; // add error accumulation (ret+=...)
}

Matrix_T<fp32> Model::forward_batch(const Matrix_T<fp32>& x) {
    const uint64_t current_batch_size = x.row();
    if (current_batch_size == 0) return Matrix_T<fp32>(0,0);

    Matrix_T<fp32> current_input = x;
    Matrix_T<fp32> layer_output(0, 0);

    for (uint64_t i = 0; i< _layers.size(); i++) {
        auto& layer = _layers[i];

        uint64_t current_output_dim = (i == _layers.size() - 1) ? _output_dim : _hidden_dim;

        if (layer_output.row() != current_batch_size || layer_output.col() != current_output_dim) {
            layer_output = Matrix_T<fp32>(current_batch_size, current_output_dim);
        }

        layer->forward(current_input, layer_output);

        if (i < _layers.size() - 1) {
            if (current_input.row() != current_batch_size || current_input.col() != current_output_dim) {
                current_input = Matrix_T<fp32>(current_batch_size, current_output_dim);
            }
            const auto& out_data  = layer_output.data(View::NT);
            auto& next_data = current_input.data(View::NT);
            #pragma omp parallel for
            for (uint64_t j = 0; j < out_data.size(); ++j) {
                next_data[j] = static_cast<fp32>(out_data[j]);
            }
        }
    }
    return layer_output;
}

Matrix_T<fp32> Model::backward_batch(const Matrix_T<fp32>& y) { // loss를 받음
    const uint64_t current_batch_size = y.row();
    if (current_batch_size == 0) return Matrix_T<fp32>(0,0);

    Matrix_T<fp32> current_grad = y;
    Matrix_T<fp32> grad_output(0,0);

    for (int i = _layers.size() - 1; i>= 0; i--) {
        auto& layer = _layers[i];

        // Determine the input dimension for the current layer to correctly size grad_output.
        // Based on the init() logic, layer 0 has `_input_dim` and all others have `_hidden_dim`.
        uint64_t input_dim_for_this_layer = (i == 0) ? _input_dim : _hidden_dim;

        if (grad_output.row() != current_batch_size || grad_output.col() != input_dim_for_this_layer) {
            grad_output = Matrix_T<fp32>(current_batch_size, input_dim_for_this_layer);
        }

        layer->backward(current_grad, grad_output);
        
        std::swap(current_grad, grad_output);
    }
    for (auto& layer : _layers) {
        layer->update();
    }
    return current_grad;
}